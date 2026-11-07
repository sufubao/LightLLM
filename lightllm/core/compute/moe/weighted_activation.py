from dataclasses import dataclass
import torch
import torch.nn.functional as F

from lightllm.core.compute.linear.backend import LinearBackend
from lightllm.core.compute.linear.mxfp import MXLinearBackend
from lightllm.core.compute.linear.spec import LinearSpec
from lightllm.core.contracts.provider import MoeOffer, ProbeResult
from lightllm.core.contracts.state import FormatSpec


@dataclass(frozen=True)
class WeightedActivationMoeSpec:
    hidden_size: int
    intermediate_size: int
    experts: int
    top_k: int
    rank: int
    world_size: int
    dtype: object
    temperature: float
    renormalize: bool
    routed_scale: float
    limit: float


class WeightedActivationMoeBackend:
    name = "tilelang.moe.weighted_activation"

    def probe(self, spec, fmt, caps):
        if not isinstance(spec, WeightedActivationMoeSpec):
            return ProbeResult(reason="requires sqrt-softplus routing with weighted intermediate activations")
        if not caps.is_nvidia or caps.capability is None or caps.capability < (9, 0):
            return ProbeResult(reason="weighted MX MoE requires NVIDIA Hopper or newer")
        if fmt.encoding not in ("plain", "fp4_group32_e8m0") or fmt.layout != "local_expert_output_input":
            return ProbeResult(reason="weighted MoE requires native BF16 or group-32 FP4 expert weights")
        return ProbeResult((MoeOffer(self.name, spec, fmt),))

    def bind(self, offer, gate_up, down, gate_up_scale=None, down_scale=None, correction_bias=None):
        return BoundWeightedActivationMoe(offer, gate_up, down, gate_up_scale, down_scale, correction_bias)


class BoundWeightedActivationMoe:
    def __init__(self, offer, gate_up, down, gate_up_scale, down_scale, correction_bias):
        self.spec, self.name = offer.spec, offer.backend
        s, fmt = self.spec, offer.format
        self.bias = correction_bias
        self.layers = []
        quantized = fmt.encoding != "plain"
        backend = MXLinearBackend(fmt.encoding) if quantized else LinearBackend("torch.linear")
        linear_format = FormatSpec(
            fmt.encoding,
            fmt.dtype,
            "output_input",
            ("weight", "scale") if quantized else ("weight",),
            1,
            group_size=32 if quantized else None,
            scale_dtype=fmt.scale_dtype,
        )
        from lightllm.core.contracts.provider import LinearOffer

        for i in range(s.experts // s.world_size):
            bindings = []
            for weight, scale, n, k in (
                (
                    gate_up[i, : s.intermediate_size],
                    None if gate_up_scale is None else gate_up_scale[i, : s.intermediate_size],
                    s.intermediate_size,
                    s.hidden_size,
                ),
                (
                    gate_up[i, s.intermediate_size :],
                    None if gate_up_scale is None else gate_up_scale[i, s.intermediate_size :],
                    s.intermediate_size,
                    s.hidden_size,
                ),
                (down[i], None if down_scale is None else down_scale[i], s.hidden_size, s.intermediate_size),
            ):
                binding = backend.bind(
                    LinearOffer(backend.name, LinearSpec(k, n, s.dtype), linear_format), weight, scale, None
                )
                bindings.append(binding)
            self.layers.append(tuple(bindings))

    def run(self, inputs, router_logits):
        s = self.spec
        scores = F.softplus(router_logits.float() / s.temperature).sqrt()
        selected = (scores + self.bias).topk(s.top_k, -1).indices
        weights = scores.gather(1, selected)
        if s.renormalize and s.top_k > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * s.routed_scale
        output = torch.zeros_like(inputs, dtype=torch.float32)
        counts = torch.bincount(selected.flatten(), minlength=s.experts).tolist()
        first = s.rank * len(self.layers)
        for local, (gate, up, down) in enumerate(self.layers):
            expert = first + local
            if not counts[expert]:
                continue
            rows, choices = torch.where(selected == expert)
            x = inputs[rows]
            g, u = gate.run(x).float(), up.run(x).float()
            if s.limit > 0:
                g, u = g.clamp(max=s.limit), u.clamp(-s.limit, s.limit)
            activation = (F.silu(g) * u * weights[rows, choices, None]).to(inputs.dtype)
            output[rows] += down.run(activation)

        return output
