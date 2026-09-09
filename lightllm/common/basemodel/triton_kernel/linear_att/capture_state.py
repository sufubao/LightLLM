"""Freeze a bounded number of recurrent states without a GPU-to-CPU decision.

The output buffers are owned by the caller and must not be reused while a
checkpoint transfer still reads them. Candidate selection is stable across TP
ranks: logical input order, rather than the order of GPU atomics, chooses slots.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _select_capture_candidates(
    req_indices,
    state_rows,
    exact_lengths,
    capture_mask,
    selected_reqs,
    selected_rows,
    selected_lengths,
    source_rows,
    N: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_REQS: tl.constexpr,
    MTP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.arange(0, BLOCK)
    tl.store(selected_reqs + index, -1, index < CAPACITY)
    tl.store(selected_rows + index, 0, index < CAPACITY)
    tl.store(selected_lengths + index, 0, index < CAPACITY)
    tl.store(source_rows + index, -1, index < CAPACITY)
    req = tl.load(req_indices + index, index < N, other=-1)
    row = tl.load(state_rows + index, index < N, other=-1)
    length = tl.load(exact_lengths + index, index < N, other=0)
    enabled = tl.load(capture_mask + index, index < N, other=0)
    valid = enabled & (req >= 0) & (req < MAX_REQS) & (row >= 0) & (row < MTP_SIZE) & (length > 0)
    slot = tl.cumsum(valid.to(tl.int32)) - 1
    keep = valid & (slot < CAPACITY)
    tl.debug_barrier()
    tl.store(selected_reqs + slot, req, keep)
    tl.store(selected_rows + slot, row, keep)
    tl.store(selected_lengths + slot, length, keep)
    tl.store(source_rows + slot, index, keep)


@triton.jit
def _freeze_linear_states(
    conv,
    ssm,
    selected_reqs,
    selected_rows,
    out_conv,
    out_ssm,
    CONV_LAYER_STRIDE: tl.constexpr,
    CONV_REQ_STRIDE: tl.constexpr,
    CONV_DIM_STRIDE: tl.constexpr,
    CONV_WIDTH_STRIDE: tl.constexpr,
    SSM_LAYER_STRIDE: tl.constexpr,
    SSM_REQ_STRIDE: tl.constexpr,
    LAYERS: tl.constexpr,
    CONV_WIDTH: tl.constexpr,
    CONV_ELEMENTS: tl.constexpr,
    SSM_ELEMENTS: tl.constexpr,
    MTP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot, layer, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req = tl.load(selected_reqs + slot)
    if req < 0:
        return
    row = tl.load(selected_rows + slot)
    index = block * BLOCK + tl.arange(0, BLOCK)
    conv_source = (
        layer * CONV_LAYER_STRIDE
        + req * CONV_REQ_STRIDE
        + (index // CONV_WIDTH) * CONV_DIM_STRIDE
        + (row + index % CONV_WIDTH) * CONV_WIDTH_STRIDE
    )
    conv_value = tl.load(conv + conv_source, index < CONV_ELEMENTS, other=0)
    tl.store(out_conv + (slot * LAYERS + layer) * CONV_ELEMENTS + index, conv_value, index < CONV_ELEMENTS)
    ssm_source = layer * SSM_LAYER_STRIDE + (req * MTP_SIZE + row) * SSM_REQ_STRIDE + index
    ssm_value = tl.load(ssm + ssm_source, index < SSM_ELEMENTS, other=0)
    tl.store(out_ssm + (slot * LAYERS + layer) * SSM_ELEMENTS + index, ssm_value, index < SSM_ELEMENTS)


def freeze_linear_states(
    conv: torch.Tensor,
    ssm: torch.Tensor,
    req_indices: torch.Tensor,
    state_rows: torch.Tensor,
    exact_lengths: torch.Tensor,
    capture_mask: torch.Tensor,
    selected_reqs: torch.Tensor,
    selected_rows: torch.Tensor,
    selected_lengths: torch.Tensor,
    source_rows: torch.Tensor,
    out_conv: torch.Tensor,
    out_ssm: torch.Tensor,
    mtp_size: int,
) -> None:
    """Gather at most ``out_conv.shape[0]`` masked candidates into owned slots.

    ``state_rows`` contains request-local MTP row numbers, not flattened model
    output rows. A row r is the state *after processing input row r*, before the
    token sampled from that row. The caller supplies that exact prefix length.
    No candidate metadata is read back to the CPU by this function.
    """
    count = req_indices.numel()
    capacity, layers, conv_dim, conv_width = out_conv.shape
    assert capacity > 0 and mtp_size > 0
    for tensor in (req_indices, state_rows, exact_lengths, capture_mask):
        assert tensor.is_cuda and tensor.ndim == 1 and tensor.numel() == count and tensor.is_contiguous()
        assert tensor.device == conv.device
    for tensor in (selected_reqs, selected_rows, selected_lengths, source_rows):
        assert tensor.is_cuda and tensor.shape == (capacity,) and tensor.is_contiguous()
        assert tensor.device == conv.device and tensor.dtype in (torch.int32, torch.int64)
    assert req_indices.dtype in (torch.int32, torch.int64)
    assert state_rows.dtype in (torch.int32, torch.int64)
    assert exact_lengths.dtype in (torch.int32, torch.int64)
    assert capture_mask.dtype == torch.bool
    assert conv.is_cuda and ssm.is_cuda and conv.device == ssm.device
    assert out_conv.device == conv.device and out_ssm.device == conv.device
    assert conv.ndim == 4 and conv.shape[0] == layers and conv.shape[2] == conv_dim
    assert conv.shape[-1] == conv_width + mtp_size - 1
    assert ssm.shape[0] == layers and ssm.shape[1] == conv.shape[1] * mtp_size
    assert ssm.is_contiguous() and out_conv.is_contiguous() and out_ssm.is_contiguous()
    assert out_ssm.shape == (capacity, layers, *ssm.shape[2:])
    assert conv.dtype == out_conv.dtype and ssm.dtype == out_ssm.dtype
    _select_capture_candidates[(1,)](
        req_indices,
        state_rows,
        exact_lengths,
        capture_mask,
        selected_reqs,
        selected_rows,
        selected_lengths,
        source_rows,
        N=count,
        CAPACITY=capacity,
        MAX_REQS=conv.shape[1] - 1,  # The final request slot is graph padding.
        MTP_SIZE=mtp_size,
        BLOCK=triton.next_power_of_2(max(count, capacity)),
    )
    ssm_elements = ssm[0, 0].numel()
    conv_elements = conv_dim * conv_width
    _freeze_linear_states[(capacity, layers, triton.cdiv(max(conv_elements, ssm_elements), 256))](
        conv,
        ssm,
        selected_reqs,
        selected_rows,
        out_conv,
        out_ssm,
        *conv.stride(),
        ssm.stride(0),
        ssm.stride(1),
        LAYERS=layers,
        CONV_WIDTH=conv_width,
        CONV_ELEMENTS=conv_elements,
        SSM_ELEMENTS=ssm_elements,
        MTP_SIZE=mtp_size,
        BLOCK=256,
    )
