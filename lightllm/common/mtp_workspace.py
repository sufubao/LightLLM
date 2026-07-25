from typing import List, Tuple

import torch


def get_mtp_workspace_request_capacity(workspace_rows: int, runtime_mtp_step: int) -> int:
    assert workspace_rows > 0
    assert runtime_mtp_step > 0
    return workspace_rows // runtime_mtp_step


def select_runtime_mtp_step(logical_batch_size: int, workspace_rows: int, max_mtp_step: int) -> int:
    assert logical_batch_size > 0
    assert workspace_rows >= logical_batch_size
    assert max_mtp_step > 0
    return min(max_mtp_step, workspace_rows // logical_batch_size)


def get_dynamic_mtp_decode_token_num(logical_batch_size: int, workspace_rows: int, max_mtp_step: int) -> int:
    if logical_batch_size == 0:
        return 0
    runtime_mtp_step = select_runtime_mtp_step(
        logical_batch_size=logical_batch_size,
        workspace_rows=workspace_rows,
        max_mtp_step=max_mtp_step,
    )
    return logical_batch_size * (runtime_mtp_step + 1) * 2


def get_dynamic_mtp_decode_token_delta(next_logical_batch_size: int, workspace_rows: int, max_mtp_step: int) -> int:
    assert next_logical_batch_size > 0
    return get_dynamic_mtp_decode_token_num(
        next_logical_batch_size, workspace_rows, max_mtp_step
    ) - get_dynamic_mtp_decode_token_num(next_logical_batch_size - 1, workspace_rows, max_mtp_step)


def compact_mtp_ssm_size(max_request_num: int, workspace_rows: int, max_mtp_step: int) -> int:
    assert max_request_num > 0
    assert workspace_rows > 0
    assert max_mtp_step > 0
    return max_request_num + 1 + workspace_rows + max_mtp_step


def get_contiguous_mtp_workspace_request_capacity(workspace_rows: int, max_mtp_step: int, runtime_mtp_step: int) -> int:
    assert workspace_rows > 0
    assert max_mtp_step > 0
    assert runtime_mtp_step > 0
    storage_rows = workspace_rows + max_mtp_step
    return storage_rows // (runtime_mtp_step + 1) - 1


def can_use_contiguous_mtp_ssm_workspace(
    logical_batch_size: int,
    workspace_rows: int,
    max_mtp_step: int,
    runtime_mtp_step: int,
) -> bool:
    return logical_batch_size <= get_contiguous_mtp_workspace_request_capacity(
        workspace_rows=workspace_rows,
        max_mtp_step=max_mtp_step,
        runtime_mtp_step=runtime_mtp_step,
    )


def get_mtp_padding_workspace_idx(
    workspace_rows: int,
    max_mtp_step: int,
    runtime_mtp_step: int,
    use_contiguous_ssm_workspace: bool,
) -> int:
    if use_contiguous_ssm_workspace:
        return get_contiguous_mtp_workspace_request_capacity(
            workspace_rows=workspace_rows,
            max_mtp_step=max_mtp_step,
            runtime_mtp_step=runtime_mtp_step,
        )
    return workspace_rows // runtime_mtp_step


def build_compact_mtp_ssm_indices(
    canonical_req_idx: torch.Tensor,
    workspace_idx: torch.Tensor,
    canonical_size: int,
    mtp_step: int,
) -> torch.Tensor:
    assert mtp_step > 0
    assert canonical_req_idx.shape == workspace_idx.shape
    extra_start = canonical_size + workspace_idx.view(-1, 1) * mtp_step
    extra_offsets = torch.arange(mtp_step, dtype=workspace_idx.dtype, device=workspace_idx.device).view(1, -1)
    return torch.cat((canonical_req_idx.view(-1, 1), extra_start + extra_offsets), dim=1)


def build_contiguous_mtp_ssm_indices(
    workspace_idx: torch.Tensor,
    canonical_size: int,
    mtp_step: int,
) -> torch.Tensor:
    assert mtp_step > 0
    assert canonical_size > 0
    mtp_size = mtp_step + 1
    workspace_start = canonical_size + workspace_idx.view(-1, 1) * mtp_size
    offsets = torch.arange(mtp_size, dtype=workspace_idx.dtype, device=workspace_idx.device).view(1, -1)
    return workspace_start + offsets


def build_runtime_mtp_conv_state_view(
    storage: torch.Tensor,
    request_capacity: int,
    conv_state_shape: Tuple[int, int],
) -> torch.Tensor:
    assert storage.dim() == 4
    assert storage.is_contiguous()
    assert request_capacity > 0
    conv_dim, conv_width = conv_state_shape
    assert conv_dim == storage.shape[2]
    assert 0 < conv_width <= storage.shape[3]
    rows = request_capacity + 1
    cell_size = conv_dim * conv_width
    assert rows * cell_size <= storage.shape[1] * storage.shape[2] * storage.shape[3]
    return storage.as_strided(
        size=(storage.shape[0], rows, conv_dim, conv_width),
        stride=(storage.stride(0), cell_size, conv_width, 1),
    )


class MTPWorkspaceAllocator:
    def __init__(self, capacity: int):
        assert capacity > 0
        self.capacity = capacity
        self.req_to_workspace = {}
        self.workspace_to_req = [None for _ in range(capacity)]
        self.pending_reuse_events = {}

    def prepare(self, req_indices: List[int]):
        assert len(req_indices) == len(set(req_indices))
        assert len(req_indices) <= self.capacity
        selected = set(req_indices)
        evicted = []
        for workspace_idx, req_idx in enumerate(self.workspace_to_req):
            if req_idx is not None and req_idx not in selected:
                evicted.append((req_idx, workspace_idx))
                del self.req_to_workspace[req_idx]
                self.workspace_to_req[workspace_idx] = None

        staged = []
        for req_idx in req_indices:
            if req_idx in self.req_to_workspace:
                continue
            workspace_idx = self.workspace_to_req.index(None)
            self.req_to_workspace[req_idx] = workspace_idx
            self.workspace_to_req[workspace_idx] = req_idx
            staged.append((req_idx, workspace_idx))

        workspace_indices = [self.req_to_workspace[req_idx] for req_idx in req_indices]
        return workspace_indices, evicted, staged

    def release(self, req_indices: List[int]):
        released = []
        for req_idx in req_indices:
            workspace_idx = self.req_to_workspace.pop(req_idx, None)
            if workspace_idx is not None:
                assert self.workspace_to_req[workspace_idx] == req_idx
                self.workspace_to_req[workspace_idx] = None
                released.append((req_idx, workspace_idx))
        return released

    def workspace_for(self, req_idx: int):
        return self.req_to_workspace.get(req_idx)

    def defer_reuse(self, released, event):
        for _, workspace_idx in released:
            assert self.workspace_to_req[workspace_idx] is None
            assert workspace_idx not in self.pending_reuse_events
            self.pending_reuse_events[workspace_idx] = event

    def take_reuse_event(self, workspace_idx: int):
        return self.pending_reuse_events.pop(workspace_idx, None)

    def reset(self):
        self.req_to_workspace.clear()
        self.workspace_to_req = [None for _ in range(self.capacity)]
        self.pending_reuse_events.clear()
