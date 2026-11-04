import torch

from lightllm.core.contracts.provider import LinearOffer, ProbeResult
from lightllm.core.compute.linear.spec import LinearSpec


class MXLinearBackend:
    def __init__(self, encoding):
        if encoding not in ("fp8_block32_e8m0", "fp4_group32_e8m0"):
            raise ValueError("unsupported MX weight encoding")
        self.encoding, self.name = encoding, f"tilelang.linear.{encoding}"

    def probe(self, spec, fmt, caps):
        if not isinstance(spec, LinearSpec) or spec.batches is not None:
            return ProbeResult(reason="MX linear requires an unbatched matrix")
        if not caps.is_nvidia or caps.capability is None or caps.capability < (9, 0):
            return ProbeResult(reason="MX linear requires NVIDIA Hopper or newer")
        expected = torch.float8_e4m3fn if self.encoding == "fp8_block32_e8m0" else torch.float4_e2m1fn_x2
        if (
            fmt.encoding != self.encoding
            or fmt.dtype != expected
            or fmt.layout != "output_input"
            or fmt.fields != ("weight", "scale")
            or fmt.group_size != 32
            or fmt.scale_dtype != torch.float8_e8m0fnu
            or fmt.version != 1
        ):
            return ProbeResult(reason="MX linear requires matching block-32 native weights and UE8M0 scales")
        if spec.dtype != torch.bfloat16 or spec.in_features % 32 or spec.out_features % 32:
            return ProbeResult(reason="MX linear requires BF16 activations and matrix dimensions divisible by 32")
        return ProbeResult((LinearOffer(self.name, spec, fmt),))

    def bind(self, offer, weight, scale, bias=None, *, allocate=torch.empty):
        spec, fmt = offer.spec, offer.format
        packed = fmt.encoding == "fp4_group32_e8m0"
        shape = (spec.out_features, spec.in_features // (2 if packed else 1))
        scale_shape = (spec.out_features if packed else spec.out_features // 32, spec.in_features // 32)
        if offer.backend != self.name or bias is not None:
            raise ValueError("MX linear binding does not match its offer or has unsupported bias")
        if (
            weight.shape != shape
            or weight.dtype != fmt.dtype
            or not weight.is_contiguous()
            or scale.shape != scale_shape
            or scale.dtype != fmt.scale_dtype
            or not scale.is_contiguous()
            or scale.device != weight.device
        ):
            raise ValueError("MX linear native weights or scales disagree with their declared format")
        return BoundMXLinear(offer, weight, scale)


class BoundMXLinear:
    def __init__(self, offer, weight, scale):
        self.name, self.spec, self.format = offer.backend, offer.spec, offer.format
        self.weight, self.scale = weight, scale

    def run(self, input_tensor, out=None, use_custom_tensor_mananger=True):
        from lightllm.core.formats.kernels.mxfp import act_quant
        from .kernels.mxfp import fp4_gemm, fp8_gemm

        x = input_tensor
        if (
            x.ndim != 2
            or x.shape[1] != self.spec.in_features
            or x.dtype != self.spec.dtype
            or x.device != self.weight.device
        ):
            raise ValueError("MX linear input does not match its bound dimensions, dtype or device")
        if out is not None and (
            out.shape != (x.shape[0], self.spec.out_features) or out.dtype != x.dtype or out.device != x.device
        ):
            raise ValueError("MX linear output does not match its bound dimensions, dtype or device")
        if not x.shape[0]:
            return out if out is not None else x.new_empty(0, self.spec.out_features)
        encoded, scale = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        if self.format.encoding == "fp4_group32_e8m0":
            result = fp4_gemm(encoded, scale, self.weight, self.scale, torch.float8_e8m0fnu, act_block_size=32)
        else:
            result = fp8_gemm(encoded, scale, self.weight, self.scale, torch.float8_e8m0fnu, block_size=32)
        if out is not None:
            out.copy_(result)
            return out
        return result
