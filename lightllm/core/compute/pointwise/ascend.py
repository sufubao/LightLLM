import torch


def bind_rms_norm(spec, resources, device, communication):
    import torch_npu

    weight = resources["weight"]
    return lambda x: torch_npu.npu_rms_norm(x, weight, epsilon=spec.eps)[0]


def bind_rope(spec, resources, device, communication):
    import torch_npu

    if spec.interleaved:
        raise ValueError("Ascend rotary binding requires half rotation")

    def run(q, k, cos, sin):
        if cos.shape[-1] * 2 == spec.head_dim:
            cos, sin = torch.cat((cos, cos), -1), torch.cat((sin, sin), -1)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        for tensor in (q, k):
            tensor.copy_(torch_npu.npu_rotary_mul(tensor.unsqueeze(0), cos, sin, rotary_mode="half").squeeze(0))

    return run


def bind_swiglu(spec, resources, device, communication):
    import torch_npu

    def run(gate_up, out):
        out.copy_(torch_npu.npu_swiglu(gate_up, dim=-1))
        return out

    return run
