# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused post-convolution preparation for FlashInfer GDN prefill.

This follows vLLM's Qwen GDN prefill path so that Q/K normalization and gate
materialization use the same floating-point operation order before FlashInfer.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_post_conv_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    stride_x_tok,
    stride_a_tok,
    stride_b_tok,
    stride_q_tok,
    stride_k_tok,
    stride_v_tok,
    L,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    APPLY_L2NORM: tl.constexpr,
    L2NORM_EPS: tl.constexpr,
    OUTPUT_G_EXP: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    token_block = tl.program_id(0)
    head = tl.program_id(1)
    hk: tl.constexpr = H * K

    token_offsets = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = token_offsets < L

    if head < H:
        feature_offsets = tl.arange(0, BK)
        feature_mask = feature_offsets < K
        mask = token_mask[:, None] & feature_mask[None, :]

        q_offsets = (
            token_offsets[:, None] * stride_x_tok + head * K + feature_offsets[None, :]
        )
        q = tl.load(mixed_qkv_ptr + q_offsets, mask=mask, other=0).to(tl.float32)
        k_offsets = (
            token_offsets[:, None] * stride_x_tok
            + hk
            + head * K
            + feature_offsets[None, :]
        )
        k = tl.load(mixed_qkv_ptr + k_offsets, mask=mask, other=0).to(tl.float32)

        if APPLY_L2NORM:
            # Preserve vLLM's exact floating-point operation order. Although
            # division by sqrt is mathematically equivalent, the reciprocal
            # followed by multiplication can round differently before Q/K
            # are stored as bfloat16 and change close greedy decisions.
            q_sq_sum = tl.sum(q * q, axis=1)
            q_inv = 1.0 / tl.sqrt(q_sq_sum + L2NORM_EPS)
            q = q * q_inv[:, None]

            k_sq_sum = tl.sum(k * k, axis=1)
            k_inv = 1.0 / tl.sqrt(k_sq_sum + L2NORM_EPS)
            k = k * k_inv[:, None]

        q_out_offsets = (
            token_offsets[:, None] * stride_q_tok + head * K + feature_offsets[None, :]
        )
        k_out_offsets = (
            token_offsets[:, None] * stride_k_tok + head * K + feature_offsets[None, :]
        )
        tl.store(q_ptr + q_out_offsets, q.to(q_ptr.dtype.element_ty), mask=mask)
        tl.store(k_ptr + k_out_offsets, k.to(k_ptr.dtype.element_ty), mask=mask)
    else:
        value_head = head - H
        feature_offsets = tl.arange(0, BV)
        feature_mask = feature_offsets < V
        mask = token_mask[:, None] & feature_mask[None, :]

        value_offset: tl.constexpr = 2 * H * K
        v_offsets = (
            token_offsets[:, None] * stride_x_tok
            + value_offset
            + value_head * V
            + feature_offsets[None, :]
        )
        v = tl.load(mixed_qkv_ptr + v_offsets, mask=mask, other=0)
        v_out_offsets = (
            token_offsets[:, None] * stride_v_tok
            + value_head * V
            + feature_offsets[None, :]
        )
        tl.store(v_ptr + v_out_offsets, v, mask=mask)

        A_log = tl.load(A_log_ptr + value_head).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + value_head).to(tl.float32)
        a_offsets = token_offsets * stride_a_tok + value_head
        b_offsets = token_offsets * stride_b_tok + value_head
        a = tl.load(a_ptr + a_offsets, mask=token_mask, other=0).to(tl.float32)
        b = tl.load(b_ptr + b_offsets, mask=token_mask, other=0).to(tl.float32)

        x = a + dt_bias
        softplus = tl.where(
            x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x))
        )
        softplus = tl.where(x <= SOFTPLUS_THRESHOLD, softplus, x)
        g = -tl.exp(A_log) * softplus
        if OUTPUT_G_EXP:
            g = tl.exp(g)

        gate_offsets = token_offsets * HV + value_head
        tl.store(g_ptr + gate_offsets, g, mask=token_mask)
        tl.store(beta_ptr + gate_offsets, tl.sigmoid(b), mask=token_mask)


def fused_gdn_prefill_post_conv(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce contiguous Q/K/V/G/beta tensors in FlashInfer's layout."""

    # LightLLM's causal-conv kernel returns [channels, tokens]. Transposing it
    # to [tokens, channels] leaves a non-contiguous view, whereas the vLLM
    # kernel receives a row-major tensor and indexes features with unit stride.
    conv_output = conv_output.contiguous()
    token_count, qkv_dim = conv_output.shape
    num_v_heads = A_log.shape[0]
    expected_qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    assert (
        qkv_dim == expected_qkv_dim
    ), f"qkv_dim={qkv_dim} != expected {expected_qkv_dim}"

    q = torch.empty(
        token_count,
        num_k_heads,
        head_k_dim,
        dtype=conv_output.dtype,
        device=conv_output.device,
    )
    k = torch.empty_like(q)
    v = torch.empty(
        token_count,
        num_v_heads,
        head_v_dim,
        dtype=conv_output.dtype,
        device=conv_output.device,
    )
    g = torch.empty(
        token_count, num_v_heads, dtype=torch.float32, device=conv_output.device
    )
    beta = torch.empty_like(g)
    if token_count == 0:
        return q, k, v, g, beta

    block_t = 16
    grid = (triton.cdiv(token_count, block_t), num_k_heads + num_v_heads)
    _fused_post_conv_kernel[grid](
        mixed_qkv_ptr=conv_output,
        a_ptr=a,
        b_ptr=b,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        g_ptr=g,
        beta_ptr=beta,
        stride_x_tok=conv_output.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_q_tok=q.stride(0),
        stride_k_tok=k.stride(0),
        stride_v_tok=v.stride(0),
        L=token_count,
        H=num_k_heads,
        HV=num_v_heads,
        K=head_k_dim,
        V=head_v_dim,
        APPLY_L2NORM=apply_l2norm,
        L2NORM_EPS=1e-6,
        OUTPUT_G_EXP=output_g_exp,
        SOFTPLUS_THRESHOLD=20.0,
        BLOCK_T=block_t,
        BK=triton.next_power_of_2(head_k_dim),
        BV=triton.next_power_of_2(head_v_dim),
        num_warps=4,
        num_stages=2,
    )
    return q, k, v, g, beta
