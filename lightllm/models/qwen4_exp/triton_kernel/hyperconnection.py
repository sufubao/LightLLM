"""Fused NVIDIA HyperConnection glue kernels for Qwen4-Exp.

The kernel layout follows the Qwen4 vLLM implementation, adapted to
LightLLM's direct Triton dispatch and tensor conventions.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    group_dim: tl.constexpr,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
):
    block_size: tl.constexpr = triton.next_power_of_2(group_dim)
    pid = tl.program_id(0)
    group_id = pid % num_groups
    row = pid // num_groups
    inner = tl.arange(0, block_size)
    mask = inner < group_dim
    offsets = group_id * group_dim + inner
    x = tl.load(x_ptr + row * stride_x + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x) / group_dim + eps)
    y = x * rrms
    y += y * w
    tl.store(y_ptr + row * stride_y + offsets, y, mask=mask)


def grouped_gemma_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    hidden_size: int,
    eps: float,
) -> torch.Tensor:
    states = hidden_states.view(-1, hidden_states.shape[-1])
    if states.shape[-1] != weight.numel():
        raise ValueError(
            f"hidden size {states.shape[-1]} does not match norm weight {weight.numel()}"
        )
    if states.shape[-1] % hidden_size != 0:
        raise ValueError(
            f"hyperconnection size {states.shape[-1]} is not divisible by hidden_size {hidden_size}"
        )
    if not states.is_cuda:
        grouped = states.float().unflatten(-1, (states.shape[-1] // hidden_size, hidden_size))
        variance = grouped.square().mean(dim=-1, keepdim=True)
        normalized = grouped * torch.rsqrt(variance + eps)
        scale = (1.0 + weight.float()).unflatten(-1, grouped.shape[-2:])
        return (normalized * scale).flatten(-2).to(states.dtype).view_as(hidden_states)

    if states.stride(-1) != 1 or weight.stride(-1) != 1:
        raise ValueError("grouped Gemma RMSNorm requires contiguous inner dimensions")
    num_groups = states.shape[-1] // hidden_size
    output = torch.empty_like(states)
    _grouped_gemma_rmsnorm_kernel[(states.shape[0] * num_groups,)](
        states,
        weight,
        output,
        states.stride(0),
        output.stride(0),
        group_dim=hidden_size,
        num_groups=num_groups,
        eps=eps,
        num_warps=8,
    )
    return output.view_as(hidden_states)


@triton.jit
def _hyperconnection_silu_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    dim: tl.constexpr,
    hc_count: tl.constexpr,
):
    block_size: tl.constexpr = triton.next_power_of_2(dim)
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < dim
    x = tl.load(x_ptr + row * stride_x + offsets, mask=mask, other=0.0).to(tl.float32)
    x /= hc_count
    tl.store(y_ptr + row * stride_y + offsets, x * tl.sigmoid(x), mask=mask)


def hyperconnection_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    if not x.is_cuda:
        return torch.nn.functional.silu(x / hc_count)
    states = x.view(-1, x.shape[-1])
    output = torch.empty_like(states)
    _hyperconnection_silu_kernel[(states.shape[0],)](
        states,
        output,
        states.stride(0),
        output.stride(0),
        dim=states.shape[-1],
        hc_count=hc_count,
        num_warps=4,
    )
    return output.view_as(x)


@triton.jit
def _hyperconnection_mix_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    stride_x,
    stride_gate,
    stride_out,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    inner = tile * block_size + tl.arange(0, block_size)
    mask = inner < hidden_size
    acc = tl.zeros([block_size], dtype=tl.float32)
    for stream in tl.static_range(hc_count):
        offsets = stream * hidden_size + inner
        gate = tl.load(gate_ptr + row * stride_gate + offsets, mask=mask, other=0.0)
        x = tl.load(x_ptr + row * stride_x + offsets, mask=mask, other=0.0)
        acc += tl.sigmoid(gate.to(tl.float32)) * x.to(tl.float32)
    tl.store(out_ptr + row * stride_out + inner, acc / hc_count, mask=mask)


def hyperconnection_mix(
    normalized_states: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    hc_count: int,
) -> torch.Tensor:
    if normalized_states.shape[-1] % hc_count != 0:
        raise ValueError(
            f"state width {normalized_states.shape[-1]} is not divisible by hc_count {hc_count}"
        )
    if normalized_states.shape != gate_logits.shape:
        raise ValueError(
            f"gate shape {tuple(gate_logits.shape)} does not match states {tuple(normalized_states.shape)}"
        )
    hidden_size = normalized_states.shape[-1] // hc_count
    if not normalized_states.is_cuda:
        gates = gate_logits.sigmoid().unflatten(-1, (hc_count, hidden_size))
        states = normalized_states.unflatten(-1, (hc_count, hidden_size))
        return (gates * states).mean(dim=-2)

    states = normalized_states.view(-1, normalized_states.shape[-1])
    gates = gate_logits.view_as(states)
    output = states.new_empty((states.shape[0], hidden_size))
    block_size = 512
    _hyperconnection_mix_kernel[(states.shape[0], triton.cdiv(hidden_size, block_size))](
        states,
        gates,
        output,
        states.stride(0),
        gates.stride(0),
        output.stride(0),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_size=block_size,
        num_warps=8,
    )
    return output.view(*normalized_states.shape[:-1], hidden_size)


@triton.jit
def _hyperconnection_combine_kernel(
    residual_ptr,
    block_ptr,
    injection_ptr,
    out_ptr,
    stride_residual,
    stride_block,
    stride_injection,
    stride_out,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_size: tl.constexpr,
):
    hc_pad: tl.constexpr = triton.next_power_of_2(hc_count)
    row = tl.program_id(0)
    tile = tl.program_id(1)
    inner = tile * block_size + tl.arange(0, block_size)
    inner_mask = inner < hidden_size
    streams = tl.arange(0, hc_pad)
    stream_mask = streams < hc_count
    offsets = streams[:, None] * hidden_size + inner[None, :]
    mask = stream_mask[:, None] & inner_mask[None, :]
    injection = tl.load(
        injection_ptr + row * stride_injection + streams, mask=stream_mask, other=0.0
    )
    block = tl.load(block_ptr + row * stride_block + inner, mask=inner_mask, other=0.0)
    residual = tl.load(residual_ptr + row * stride_residual + offsets, mask=mask, other=0.0)
    injection = 2.0 * tl.sigmoid(injection.to(tl.float32) / hc_count)
    output = residual.to(tl.float32) + block.to(tl.float32)[None, :] * injection[:, None]
    tl.store(out_ptr + row * stride_out + offsets, output, mask=mask)


def hyperconnection_combine(
    hidden_states: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    *,
    hc_count: int,
) -> torch.Tensor:
    if hidden_states.shape[-1] % hc_count != 0:
        raise ValueError(
            f"state width {hidden_states.shape[-1]} is not divisible by hc_count {hc_count}"
        )
    hidden_size = hidden_states.shape[-1] // hc_count
    if block_output.shape != (*hidden_states.shape[:-1], hidden_size):
        raise ValueError(
            f"block output shape {tuple(block_output.shape)} must end in {hidden_size}"
        )
    if injection_logits.shape != (*hidden_states.shape[:-1], hc_count):
        raise ValueError(
            f"injection shape {tuple(injection_logits.shape)} must end in {hc_count}"
        )
    if not hidden_states.is_cuda:
        injection = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
        combined = hidden_states.unflatten(-1, (hc_count, hidden_size)).float()
        combined = combined + block_output.float().unsqueeze(-2) * injection.unsqueeze(-1)
        return combined.flatten(-2).to(hidden_states.dtype)

    residual = hidden_states.view(-1, hidden_states.shape[-1])
    block = block_output.view(-1, hidden_size)
    injection = injection_logits.view(-1, hc_count)
    output = torch.empty_like(residual)
    block_size = 512
    _hyperconnection_combine_kernel[(residual.shape[0], triton.cdiv(hidden_size, block_size))](
        residual,
        block,
        injection,
        output,
        residual.stride(0),
        block.stride(0),
        injection.stride(0),
        output.stride(0),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_size=block_size,
        num_warps=8,
    )
    return output.view_as(hidden_states)


@triton.jit
def _hyperconnection_combine_norm_kernel(
    residual_ptr,
    block_ptr,
    injection_ptr,
    weight_ptr,
    out_ptr,
    norm_ptr,
    stride_residual,
    stride_block,
    stride_injection,
    stride_out,
    stride_norm,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    hc_pad: tl.constexpr = triton.next_power_of_2(hc_count)
    num_tiles: tl.constexpr = triton.cdiv(hidden_size, block_size)
    num_tiles_pad: tl.constexpr = triton.next_power_of_2(num_tiles)
    row = tl.program_id(0)
    stream = tl.program_id(1)
    streams = tl.arange(0, hc_pad)
    stream_mask = streams < hc_count
    tile_ids = tl.arange(0, num_tiles_pad)
    inner = tile_ids[:, None] * block_size + tl.arange(0, block_size)[None, :]
    inner_mask = inner < hidden_size
    offsets = stream * hidden_size + inner
    residual = tl.load(
        residual_ptr + row * stride_residual + offsets, mask=inner_mask, other=0.0
    )
    injection = tl.load(
        injection_ptr + row * stride_injection + streams, mask=stream_mask, other=0.0
    )
    block = tl.load(block_ptr + row * stride_block + inner, mask=inner_mask, other=0.0)
    injection = 2.0 * tl.sigmoid(injection.to(tl.float32) / hc_count)
    injection = tl.sum(tl.where(streams == stream, injection, 0.0))
    output = (
        residual.to(tl.float32) + block.to(tl.float32) * injection
    ).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride_out + offsets, output, mask=inner_mask)
    output_f32 = output.to(tl.float32)
    sum_sq = tl.sum(tl.sum(output_f32 * output_f32, axis=1), axis=0)
    rrms = tl.rsqrt(sum_sq / hidden_size + eps)
    weight = tl.load(weight_ptr + offsets, mask=inner_mask, other=0.0).to(tl.float32)
    normalized = output_f32 * rrms
    normalized += normalized * weight
    tl.store(norm_ptr + row * stride_norm + offsets, normalized, mask=inner_mask)


def hyperconnection_combine_norm(
    hidden_states: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    hidden_size: int,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hidden_states.is_cuda:
        combined = hyperconnection_combine(
            hidden_states, block_output, injection_logits, hc_count=hc_count
        )
        normalized = grouped_gemma_rmsnorm(
            combined, norm_weight, hidden_size=hidden_size, eps=eps
        )
        return combined, normalized

    residual = hidden_states.view(-1, hidden_states.shape[-1])
    block = block_output.view(-1, hidden_size)
    injection = injection_logits.view(-1, hc_count)
    output = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    block_size = 512
    _hyperconnection_combine_norm_kernel[(residual.shape[0], hc_count)](
        residual,
        block,
        injection,
        norm_weight,
        output,
        normalized,
        residual.stride(0),
        block.stride(0),
        injection.stride(0),
        output.stride(0),
        normalized.stride(0),
        hidden_size=hidden_size,
        hc_count=hc_count,
        eps=eps,
        block_size=block_size,
        num_warps=8,
    )
    return output.view_as(hidden_states), normalized.view_as(hidden_states)
