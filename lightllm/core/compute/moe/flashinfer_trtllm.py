from importlib import import_module

import torch

from lightllm.core.compute.moe.routing import MoeRouter
from lightllm.core.compute.moe.spec import MoeSpec
from lightllm.core.formats.weights.moe import validate_moe_weights
from lightllm.core.formats.activations.fp8 import quantize_fp8
from lightllm.core.contracts.provider import MoeOffer, ProbeResult


class FlashInferTrtllmMoeBackend:
    name = "flashinfer_trtllm"

    def probe(self, spec, format, device, *, fused_shared_experts=0):
        if not isinstance(spec, MoeSpec):
            return ProbeResult(reason="FlashInfer TRT-LLM MoE requires standard routed expert semantics")
        if not device.is_nvidia or device.capability not in ((10, 0), (10, 3)):
            return ProbeResult(reason="FlashInfer TRT-LLM MoE requires NVIDIA SM100/SM103 (Blackwell)")
        if spec.dtype != torch.bfloat16 or spec.round_intermediates:
            return ProbeResult(reason="FlashInfer TRT-LLM MoE requires BF16 without intermediate rounding")
        quantized = format.encoding == "fp8_block128"
        fields = ("gate_up", "down", "gate_up_scale", "down_scale") if quantized else ("gate_up", "down")
        if (
            format.encoding not in ("plain", "fp8_block128")
            or format.layout != "expert_output_input"
            or format.fields != fields
            or format.version != 1
            or format.dtype != (torch.float8_e4m3fn if quantized else torch.bfloat16)
            or format.group_size != (128 if quantized else None)
            or format.scale_dtype != (torch.float32 if quantized else None)
        ):
            return ProbeResult(reason="FlashInfer TRT-LLM MoE needs BF16 or FP8 block-128 expert weights without bias")
        if spec.hidden_size % 128 or spec.intermediate_size % (128 * format.shard[1]):
            return ProbeResult(reason="FlashInfer TRT-LLM MoE requires complete 128-element hidden and TP groups")
        if spec.limit is not None or spec.alpha != 1:
            return ProbeResult(reason="FlashInfer TRT-LLM MoE currently supports unclamped SwiGLU")
        if spec.group_score_topk not in (1, 2) or spec.groups & (spec.groups - 1):
            return ProbeResult(reason="MoE routing needs power-of-two groups and top-1/top-2 group scores")
        if not 0 <= fused_shared_experts <= spec.shared_experts or (
            spec.gated_shared and fused_shared_experts != spec.shared_experts
        ):
            return ProbeResult(reason="shared expert views must match the model declaration")
        try:
            api = import_module("flashinfer.fused_moe")
            prepare = import_module("flashinfer.fused_moe.prepare")
            getattr(api, "trtllm_fp8_block_scale_routed_moe" if quantized else "trtllm_bf16_routed_moe")
            getattr(prepare, "prepare_trtllm_bf16_weights")
        except (ImportError, AttributeError) as exc:
            return ProbeResult(reason=f"FlashInfer 0.6.18+ routed MoE API is unavailable: {exc}")
        return ProbeResult(offers=(MoeOffer(self.name, spec, format, fused_shared_experts),))

    def bind(
        self, offer, gate_up, down, *, gate_up_scale=None, down_scale=None, correction_bias=None, expert_scale=None
    ):
        if offer.backend != self.name:
            raise ValueError("MoE offer belongs to another backend")
        from lightllm.core.contracts.device import DeviceCaps

        device = gate_up.device
        caps = DeviceCaps(
            device.type, device.index, torch.cuda.get_device_capability(device) if gate_up.is_cuda else None
        )
        result = self.probe(offer.spec, offer.format, caps, fused_shared_experts=offer.fused_shared_experts)
        if not result.offers:
            raise ValueError(result.reason)
        validate_moe_weights(offer, gate_up, down, gate_up_scale, down_scale, correction_bias, expert_scale)
        return BoundFlashInferTrtllmMoe(offer, gate_up, down, gate_up_scale, down_scale, correction_bias, expert_scale)


class BoundFlashInferTrtllmMoe:
    def __init__(self, offer, gate_up, down, gate_up_scale, down_scale, correction_bias, expert_scale):
        self.offer, self.spec, self.device = offer, offer.spec, gate_up.device
        self.router = MoeRouter(offer, self.device, correction_bias, expert_scale)
        self.quantized = offer.format.encoding == "fp8_block128"
        self.width = self.spec.intermediate_size // offer.format.shard[1]
        self.count = self.spec.experts + offer.fused_shared_experts
        up_gate = torch.cat(gate_up.chunk(2, dim=1)[::-1], dim=1)
        api = import_module("flashinfer.fused_moe")
        if self.quantized:
            self.run_kernel = api.trtllm_fp8_block_scale_routed_moe
            self.weights = dict(
                gemm1_weights=up_gate,
                gemm1_weights_scale=torch.cat(gate_up_scale.chunk(2, dim=1)[::-1], dim=1),
                gemm2_weights=down,
                gemm2_weights_scale=down_scale,
                use_shuffled_weight=False,
                weight_layout=0,
                routing_bias=None,
            )
        else:
            prepare = import_module("flashinfer.fused_moe.prepare")
            self.run_kernel = api.trtllm_bf16_routed_moe
            self.weights = prepare.prepare_trtllm_bf16_weights(
                up_gate,
                down,
                num_local_experts=self.count,
                hidden_size=self.spec.hidden_size,
                intermediate_size=self.width,
                permute_cache={},
            )

    def route(self, logits, shared_gate=None, capture=None):
        return self.router.run(logits, shared_gate, capture)

    def run(self, inputs, router_logits, *, shared_gate=None, capture=None):
        if (
            inputs.ndim != 2
            or router_logits.ndim != 2
            or inputs.shape[1] != self.spec.hidden_size
            or inputs.dtype != self.spec.dtype
            or inputs.device != self.device
            or not inputs.is_contiguous()
            or router_logits.shape[0] != inputs.shape[0]
        ):
            raise ValueError("MoE inputs do not match the bound native weights")
        weights, ids = self.route(router_logits, shared_gate, capture)
        if not inputs.shape[0]:
            return inputs
        destination = inputs
        kwargs = {}
        if self.quantized:
            inputs, scales = quantize_fp8(inputs, 128, torch.float8_e4m3fn)
            kwargs["hidden_states_scale"] = scales.T.contiguous()
        output = self.run_kernel(
            topk_ids=(ids.to(torch.int32), weights.float().contiguous()),
            hidden_states=inputs,
            num_experts=self.count,
            top_k=ids.shape[1],
            n_group=None,
            topk_group=None,
            intermediate_size=self.width,
            local_expert_offset=0,
            local_num_experts=self.count,
            routed_scaling_factor=None,
            routing_method_type=0,
            do_finalize=True,
            **self.weights,
            **kwargs,
        )
        output = output[0] if isinstance(output, (tuple, list)) else output
        destination.copy_(output)
        return destination
