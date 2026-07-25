import torch
import triton
import triton.language as tl


@triton.jit
def _reset_mtp_state_index_kernel(state_index, req_idx, req_num: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < req_num
    requests = tl.load(req_idx + offsets, mask=mask).to(tl.int64)
    tl.store(state_index + requests, 0, mask=mask)


@triton.jit
def _commit_compact_ssm_state_kernel(
    state,
    req_idx,
    workspace_idx,
    accepted_idx,
    state_stride_l,
    state_stride_s,
    canonical_size: tl.constexpr,
    mtp_step: tl.constexpr,
    cell_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_pos = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    block = tl.program_id(2)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < cell_size

    request = tl.load(req_idx + req_pos).to(tl.int64)
    workspace_request = tl.load(workspace_idx + req_pos).to(tl.int64)
    accepted = tl.load(accepted_idx + request).to(tl.int64)
    workspace_slot = canonical_size + workspace_request * mtp_step + accepted - 1
    copy_mask = mask & (accepted > 0)
    src = state + layer * state_stride_l + workspace_slot * state_stride_s + offsets
    dst = state + layer * state_stride_l + request * state_stride_s + offsets

    values = tl.load(src, mask=copy_mask)
    tl.store(dst, values, mask=copy_mask)


@triton.jit
def _copy_contiguous_ssm_state_kernel(
    state,
    req_idx,
    workspace_idx,
    accepted_idx,
    state_stride_l,
    state_stride_s,
    canonical_size: tl.constexpr,
    mtp_size: tl.constexpr,
    cell_size: tl.constexpr,
    commit: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_pos = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    block = tl.program_id(2)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < cell_size

    request = tl.load(req_idx + req_pos).to(tl.int64)
    workspace_request = tl.load(workspace_idx + req_pos).to(tl.int64)
    workspace_slot = canonical_size + workspace_request * mtp_size
    if commit:
        workspace_slot += tl.load(accepted_idx + request).to(tl.int64)
        src_slot = workspace_slot
        dst_slot = request
    else:
        src_slot = request
        dst_slot = workspace_slot

    src = state + layer * state_stride_l + src_slot * state_stride_s + offsets
    dst = state + layer * state_stride_l + dst_slot * state_stride_s + offsets
    values = tl.load(src, mask=mask)
    tl.store(dst, values, mask=mask)


@triton.jit
def _copy_conv_state_kernel(
    canonical,
    workspace,
    req_idx,
    workspace_idx,
    accepted_idx,
    canonical_stride_l,
    canonical_stride_s,
    canonical_stride_c,
    workspace_stride_l,
    workspace_stride_s,
    workspace_stride_c,
    conv_width: tl.constexpr,
    conv_dim: tl.constexpr,
    commit: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_pos = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    block = tl.program_id(2)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    cell_size = conv_dim * conv_width
    mask = offsets < cell_size

    request = tl.load(req_idx + req_pos).to(tl.int64)
    workspace_request = tl.load(workspace_idx + req_pos).to(tl.int64)
    conv_row = offsets // conv_width
    conv_col = offsets % conv_width
    workspace_col = conv_col
    if commit:
        workspace_col += tl.load(accepted_idx + request).to(tl.int64)
        src = (
            workspace
            + layer * workspace_stride_l
            + workspace_request * workspace_stride_s
            + conv_row * workspace_stride_c
            + workspace_col
        )
        dst = (
            canonical
            + layer * canonical_stride_l
            + request * canonical_stride_s
            + conv_row * canonical_stride_c
            + conv_col
        )
    else:
        src = (
            canonical
            + layer * canonical_stride_l
            + request * canonical_stride_s
            + conv_row * canonical_stride_c
            + conv_col
        )
        dst = (
            workspace
            + layer * workspace_stride_l
            + workspace_request * workspace_stride_s
            + conv_row * workspace_stride_c
            + workspace_col
        )

    values = tl.load(src, mask=mask)
    tl.store(dst, values, mask=mask)


def copy_canonical_to_mtp_workspace(
    canonical_conv: torch.Tensor,
    workspace_conv: torch.Tensor,
    req_idx: torch.Tensor,
    workspace_idx: torch.Tensor,
    ssm_state: torch.Tensor = None,
    canonical_ssm_size: int = None,
    mtp_step: int = None,
    use_contiguous_ssm_workspace: bool = False,
) -> None:
    if use_contiguous_ssm_workspace:
        assert ssm_state is not None
        assert canonical_ssm_size is not None
        assert mtp_step is not None and mtp_step > 0
        _copy_contiguous_ssm_state(
            ssm_state=ssm_state,
            req_idx=req_idx,
            workspace_idx=workspace_idx,
            accepted_idx=None,
            canonical_ssm_size=canonical_ssm_size,
            mtp_step=mtp_step,
            commit=False,
        )
    _copy_conv_state(
        canonical_conv=canonical_conv,
        workspace_conv=workspace_conv,
        req_idx=req_idx,
        workspace_idx=workspace_idx,
        accepted_idx=None,
        commit=False,
    )


def copy_mtp_workspace_to_canonical(
    ssm_state: torch.Tensor,
    canonical_conv: torch.Tensor,
    workspace_conv: torch.Tensor,
    req_idx: torch.Tensor,
    workspace_idx: torch.Tensor,
    accepted_idx: torch.Tensor,
    canonical_ssm_size: int,
    mtp_step: int,
    use_contiguous_ssm_workspace: bool = False,
) -> None:
    assert req_idx.is_cuda and req_idx.dtype == torch.int32
    assert workspace_idx.is_cuda and workspace_idx.dtype == torch.int32
    assert workspace_idx.shape == req_idx.shape
    assert accepted_idx.is_cuda
    assert ssm_state.shape[1] >= canonical_ssm_size + mtp_step

    if use_contiguous_ssm_workspace:
        _copy_contiguous_ssm_state(
            ssm_state=ssm_state,
            req_idx=req_idx,
            workspace_idx=workspace_idx,
            accepted_idx=accepted_idx,
            canonical_ssm_size=canonical_ssm_size,
            mtp_step=mtp_step,
            commit=True,
        )
    else:
        layer_num = ssm_state.shape[0]
        ssm_cell_size = ssm_state[0, 0].numel()
        block = 256
        _commit_compact_ssm_state_kernel[(req_idx.numel(), layer_num, triton.cdiv(ssm_cell_size, block))](
            ssm_state,
            req_idx,
            workspace_idx,
            accepted_idx,
            ssm_state.stride(0),
            ssm_state.stride(1),
            canonical_size=canonical_ssm_size,
            mtp_step=mtp_step,
            cell_size=ssm_cell_size,
            BLOCK=block,
        )
    _copy_conv_state(
        canonical_conv=canonical_conv,
        workspace_conv=workspace_conv,
        req_idx=req_idx,
        workspace_idx=workspace_idx,
        accepted_idx=accepted_idx,
        commit=True,
    )
    block = triton.next_power_of_2(req_idx.numel())
    _reset_mtp_state_index_kernel[(1,)](
        accepted_idx,
        req_idx,
        req_num=req_idx.numel(),
        BLOCK=block,
    )


def _copy_contiguous_ssm_state(
    ssm_state: torch.Tensor,
    req_idx: torch.Tensor,
    workspace_idx: torch.Tensor,
    accepted_idx: torch.Tensor,
    canonical_ssm_size: int,
    mtp_step: int,
    commit: bool,
) -> None:
    assert req_idx.is_cuda and req_idx.dtype == torch.int32
    assert workspace_idx.is_cuda and workspace_idx.dtype == torch.int32
    assert workspace_idx.shape == req_idx.shape
    if commit:
        assert accepted_idx is not None and accepted_idx.is_cuda
    else:
        accepted_idx = req_idx

    layer_num = ssm_state.shape[0]
    cell_size = ssm_state[0, 0].numel()
    block = 256
    _copy_contiguous_ssm_state_kernel[(req_idx.numel(), layer_num, triton.cdiv(cell_size, block))](
        ssm_state,
        req_idx,
        workspace_idx,
        accepted_idx,
        ssm_state.stride(0),
        ssm_state.stride(1),
        canonical_size=canonical_ssm_size,
        mtp_size=mtp_step + 1,
        cell_size=cell_size,
        commit=commit,
        BLOCK=block,
    )


def _copy_conv_state(
    canonical_conv: torch.Tensor,
    workspace_conv: torch.Tensor,
    req_idx: torch.Tensor,
    workspace_idx: torch.Tensor,
    accepted_idx: torch.Tensor,
    commit: bool,
) -> None:
    assert req_idx.is_cuda and req_idx.dtype == torch.int32
    assert workspace_idx.is_cuda and workspace_idx.dtype == torch.int32
    assert workspace_idx.shape == req_idx.shape
    assert canonical_conv.shape[0] == workspace_conv.shape[0]
    if commit:
        assert accepted_idx is not None and accepted_idx.is_cuda
    else:
        accepted_idx = req_idx

    layer_num = canonical_conv.shape[0]
    block = 256
    conv_dim = canonical_conv.shape[-2]
    conv_width = canonical_conv.shape[-1]
    _copy_conv_state_kernel[(req_idx.numel(), layer_num, triton.cdiv(conv_dim * conv_width, block))](
        canonical_conv,
        workspace_conv,
        req_idx,
        workspace_idx,
        accepted_idx,
        canonical_conv.stride(0),
        canonical_conv.stride(1),
        canonical_conv.stride(2),
        workspace_conv.stride(0),
        workspace_conv.stride(1),
        workspace_conv.stride(2),
        conv_width=conv_width,
        conv_dim=conv_dim,
        commit=commit,
        BLOCK=block,
    )
