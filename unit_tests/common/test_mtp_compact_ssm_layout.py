from types import SimpleNamespace

import torch

import lightllm.common.basemodel.cuda_graph as cuda_graph_module
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.mtp_workspace import (
    build_compact_mtp_ssm_indices,
    build_contiguous_mtp_ssm_indices,
    build_runtime_mtp_conv_state_view,
    can_use_contiguous_mtp_ssm_workspace,
    compact_mtp_ssm_size,
    get_contiguous_mtp_workspace_request_capacity,
    get_dynamic_mtp_decode_token_delta,
    get_dynamic_mtp_decode_token_num,
    get_mtp_padding_workspace_idx,
    get_mtp_workspace_request_capacity,
    select_runtime_mtp_step,
)


def test_compact_ssm_indices_reuse_canonical_slot_zero():
    indices = build_compact_mtp_ssm_indices(
        canonical_req_idx=torch.tensor([4, 1], dtype=torch.int32),
        workspace_idx=torch.tensor([1, 0], dtype=torch.int32),
        canonical_size=6,
        mtp_step=3,
    )

    assert torch.equal(
        indices,
        torch.tensor(
            [
                [4, 9, 10, 11],
                [1, 6, 7, 8],
            ],
            dtype=torch.int32,
        ),
    )


def test_compact_ssm_storage_has_one_canonical_and_mtp_step_extra_rows():
    max_request_num = 128
    workspace_rows = 128
    max_mtp_step = 4

    compact_rows = compact_mtp_ssm_size(
        max_request_num=max_request_num,
        workspace_rows=workspace_rows,
        max_mtp_step=max_mtp_step,
    )

    assert compact_rows == 261
    assert compact_rows == (max_request_num + 1) + workspace_rows + max_mtp_step


def test_contiguous_ssm_indices_use_workspace_slot_zero():
    indices = build_contiguous_mtp_ssm_indices(
        workspace_idx=torch.tensor([1, 0], dtype=torch.int32),
        canonical_size=6,
        mtp_step=3,
    )

    assert torch.equal(
        indices,
        torch.tensor(
            [
                [10, 11, 12, 13],
                [6, 7, 8, 9],
            ],
            dtype=torch.int32,
        ),
    )


def test_contiguous_ssm_workspace_reuses_existing_tail_for_mtp3_batch32():
    workspace_rows = 128
    max_mtp_step = 4

    assert (
        get_contiguous_mtp_workspace_request_capacity(
            workspace_rows=workspace_rows,
            max_mtp_step=max_mtp_step,
            runtime_mtp_step=3,
        )
        == 32
    )
    assert can_use_contiguous_mtp_ssm_workspace(
        logical_batch_size=32,
        workspace_rows=workspace_rows,
        max_mtp_step=max_mtp_step,
        runtime_mtp_step=3,
    )
    assert not can_use_contiguous_mtp_ssm_workspace(
        logical_batch_size=33,
        workspace_rows=workspace_rows,
        max_mtp_step=max_mtp_step,
        runtime_mtp_step=3,
    )
    assert (
        get_mtp_padding_workspace_idx(
            workspace_rows=workspace_rows,
            max_mtp_step=max_mtp_step,
            runtime_mtp_step=3,
            use_contiguous_ssm_workspace=True,
        )
        == 32
    )


def test_runtime_conv_view_uses_exact_step_width_without_extra_allocation():
    storage = torch.arange(2 * 5 * 3 * 6).reshape(2, 5, 3, 6)

    runtime_view = build_runtime_mtp_conv_state_view(
        storage=storage,
        request_capacity=4,
        conv_state_shape=(3, 4),
    )

    assert runtime_view.shape == (2, 5, 3, 4)
    assert runtime_view.stride() == (90, 12, 4, 1)
    assert runtime_view.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    runtime_view[0, 1, 0, 0] = -1
    assert storage.view(2, -1)[0, 12] == -1


def test_dynamic_cuda_graph_has_contiguous_layout_boundary(monkeypatch):
    args = SimpleNamespace(
        mtp_step=4,
        graph_split_batch_size=16,
        graph_grow_step_size=32,
        dynamic_mtp=True,
        mtp_workspace_rows=128,
        max_mtp_step=4,
        enable_tpsp_mix_mode=False,
    )
    monkeypatch.setattr(cuda_graph_module, "get_env_start_args", lambda: args)

    batch_sizes = CudaGraph.gen_cuda_graph_batch_sizes(
        max_batch_size=42 * 4,
        tp_world_size=4,
        mtp_step=3,
    )

    assert 32 * 4 in batch_sizes
    assert batch_sizes[batch_sizes.index(32 * 4) + 1] == 42 * 4


def test_runtime_mtp_step_uses_fixed_workspace_row_budget():
    workspace_rows = 128
    max_mtp_step = 4

    expected = {
        1: 4,
        32: 4,
        33: 3,
        42: 3,
        43: 2,
        64: 2,
        65: 1,
        128: 1,
    }

    for batch_size, runtime_step in expected.items():
        assert (
            select_runtime_mtp_step(
                logical_batch_size=batch_size,
                workspace_rows=workspace_rows,
                max_mtp_step=max_mtp_step,
            )
            == runtime_step
        )
        assert batch_size * runtime_step <= workspace_rows


def test_workspace_request_capacity_and_padding_block_depend_on_runtime_step():
    workspace_rows = 128

    assert get_mtp_workspace_request_capacity(workspace_rows, runtime_mtp_step=1) == 128
    assert get_mtp_workspace_request_capacity(workspace_rows, runtime_mtp_step=2) == 64
    assert get_mtp_workspace_request_capacity(workspace_rows, runtime_mtp_step=3) == 42
    assert get_mtp_workspace_request_capacity(workspace_rows, runtime_mtp_step=4) == 32


def test_dynamic_decode_token_reservation_follows_runtime_step_boundaries():
    workspace_rows = 128
    max_mtp_step = 4

    assert get_dynamic_mtp_decode_token_num(32, workspace_rows, max_mtp_step) == 320
    assert get_dynamic_mtp_decode_token_num(33, workspace_rows, max_mtp_step) == 264
    assert get_dynamic_mtp_decode_token_delta(33, workspace_rows, max_mtp_step) == -56
