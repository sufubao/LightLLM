import torch

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import PostLayerOutput
from .adapter import DeepseekV41Adapter


class DeepseekV41TpPartModel(TpPartBaseModel):
    is_hybrid_att_model = True
    has_vision = False
    model_adapter_class = DeepseekV41Adapter

    @classmethod
    def normalize_model_config(cls, model_config):
        result = super().normalize_model_config(model_config)
        if "quantization_config" in model_config:
            result["quantization_config"] = model_config["quantization_config"]
        return result

    def _verify_params(self):
        if self.load_way != "HF" or self.data_type != torch.bfloat16:
            raise ValueError("DeepSeek-V4.1 requires an HF checkpoint and BF16 activations")
        if not self.disable_cudagraph or not self.args.disable_dynamic_prompt_cache:
            raise ValueError("DeepSeek-V4.1 currently requires --disable_cudagraph --disable_dynamic_prompt_cache")
        for flag in (
            "enable_cpu_cache",
            "enable_prefill_cudagraph",
            "enable_tpsp_mix_mode",
            "enable_ep_moe",
            "enable_decode_microbatch_overlap",
            "enable_prefill_microbatch_overlap",
            "enable_dp_prefill_balance",
            "enable_multimodal",
            "enable_prefill_decode_mixed",
        ):
            if getattr(self.args, flag, False):
                raise ValueError(f"DeepSeek-V4.1 does not yet support {flag}")
        if self.args.mtp_mode is not None or self.run_mode in ("prefill", "decode"):
            raise ValueError("DeepSeek-V4.1 does not yet support speculative decoding or PD separation")
        if self.args.llm_kv_type != "None":
            raise ValueError("DeepSeek-V4.1 uses its declared MX cache format; do not select a generic KV encoding")

    def _init_quant(self):
        self.quant_cfg = None

    def _create_model_adapter(self, adapter_class):
        from lightllm.core.device.cuda import CudaRuntime
        from lightllm.distributed.communication_op import dist_group_manager

        if self.execution_device is None:
            self.execution_device = CudaRuntime()
        group = dist_group_manager.get_group(0).device_group if self.tp_world_size_ > 1 else None
        adapter = adapter_class(
            self.config,
            role=self.att_state_role,
            runtime=self.execution_device,
            rank=self.tp_rank_,
            world=self.tp_world_size_,
            request_capacity=self.max_req_num + 1,
            max_seq_len=self.get_req_manager_max_seq_len(),
            group=group,
        )
        self.execution_providers = adapter.providers(self.execution_providers)
        self.execution_formats = self.execution_formats if self.execution_formats is not None else adapter.formats
        return adapter

    def _forward_v41(self, state):
        phase = "prefill" if state.is_prefill else "decode"
        metadata = state.execution_metadata[(self.att_state_role, "shared_compressed", phase)]
        logits = self.model_adapter.forward(
            state.input_ids, metadata, phase, all_logits=state.is_prefill and state.return_all_prompt_logics
        )
        if state.is_prefill and state.return_all_prompt_logics:
            state.prompt_logics = logits
            logits = logits[[offset + count - 1 for _, _, count, offset in metadata.requests]]
        return self._create_model_output(PostLayerOutput(logits=logits), state)

    _context_forward = _forward_v41
    _token_forward = _forward_v41
