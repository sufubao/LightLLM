from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lightllm.common.context_parallel.execution import ContextParallelExecutor
from lightllm.common.context_parallel_runtime import ContextParallelLayout, create_context_parallel_groups


class _State:
    def __init__(self):
        self.k = torch.zeros(4, 1, 1)
        self.v = torch.zeros(4, 1, 1)

    def get_layer_view(self, logical_layer_id):
        return self.k, self.v

    def write_full_att_kv(self, logical_layer_id, token_indices, kv):
        self.k[token_indices, 0] = kv[:, 0]
        self.v[token_indices, 0] = kv[:, 1]


class _StateManager:
    def __init__(self, state):
        self.state = state

    def get_state(self, **kwargs):
        return self.state


class _Layer:
    layer_num_ = 0
    tp_q_head_num_ = 1
    head_dim_ = 1


def _state(*, prefill, mem_index, sequence_length=4):
    state = SimpleNamespace(
        is_prefill=prefill,
        input_ids=torch.arange(4),
        mem_index=mem_index,
        total_token_num=4,
        b_seq_len=torch.tensor([sequence_length], dtype=torch.int32),
        b_ready_cache_len=torch.tensor([0], dtype=torch.int32),
        b_req_idx=torch.tensor([0], dtype=torch.int32),
        req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(4).view(1, 4)),
        att_state_role="main",
        att_state_manager=_StateManager(_State()),
    )
    return state


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_gloo_pcp_attention_replicates_kv_and_restores_prompt_order(tmp_path):
    mp.spawn(_pcp_worker, args=(str(tmp_path / "pcp"),), nprocs=2, join=True)


def _pcp_worker(rank, init_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    groups = None
    try:
        groups = create_context_parallel_groups(
            ContextParallelLayout(global_rank=rank, dp_size=1, context_size=2, tp_size=1), backend="gloo"
        )
        executor = ContextParallelExecutor(groups, pcp_size=2, dcp_size=1)
        state = _state(prefill=True, mem_index=torch.arange(rank * 2, rank * 2 + 2))
        attached = executor.attach(state)
        attached.prepare_prefill(state)
        local_kv = torch.tensor([[[0.0], [1.0 + rank * 2]], [[0.0], [2.0 + rank * 2]]])
        query = torch.zeros(2, 1)
        output = attached.prefill_attention(_Layer(), query, local_kv, state)
        expected = torch.tensor([[1.0], [1.5]]) if rank == 0 else torch.tensor([[2.0], [2.5]])
        torch.testing.assert_close(output, expected)
        restored = attached.finish_prefill(state, output)
        torch.testing.assert_close(restored, torch.tensor([[1.0], [1.5], [2.0], [2.5]]))
        assert state.input_ids.shape == (4,)
        assert state.mem_index.tolist() == list(range(rank * 2, rank * 2 + 2))
    finally:
        if groups is not None:
            groups.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_gloo_dcp_attention_merges_logsumexp_partials(tmp_path):
    mp.spawn(_dcp_worker, args=(str(tmp_path / "dcp"),), nprocs=2, join=True)


def _dcp_worker(rank, init_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    groups = None
    try:
        groups = create_context_parallel_groups(
            ContextParallelLayout(global_rank=rank, dp_size=1, context_size=2, tp_size=1), backend="gloo"
        )
        executor = ContextParallelExecutor(groups, pcp_size=1, dcp_size=2)
        state = _state(prefill=False, mem_index=torch.tensor([3]))
        state.att_state_manager.state.k[:, 0, 0] = torch.arange(1, 5)
        state.att_state_manager.state.v[:, 0, 0] = torch.arange(1, 5)
        attached = executor.attach(state)
        output = attached.decode_attention(_Layer(), torch.zeros(1, 1), state)
        torch.testing.assert_close(output, torch.tensor([[2.5]]))
    finally:
        if groups is not None:
            groups.close()
        dist.destroy_process_group()
