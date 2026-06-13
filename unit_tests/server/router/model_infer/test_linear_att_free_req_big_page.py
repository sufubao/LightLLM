"""Regression tests for the big-page state-buffer release on pause/abort mid-prefill.

Bug: in big-page mode (--linear_att_page_block_num set), a request whose chunked
prefill crossed a big-page boundary (filling req.linear_att_len_to_big_page_id) and
was then paused/aborted before completing took _linear_att_free_req's fallback branch,
which freed tokens but never released the accumulated big-page state-buffer ids ->
free_a_req_mem's `assert len(req.linear_att_len_to_big_page_id) == 0` crashed the
worker (or leaked big-page slots with asserts disabled).
"""
import types

import torch
from sortedcontainers import SortedDict

import lightllm.common.basemodel  # noqa: F401  (import first to break a circular-import cycle)
from lightllm.server.router.model_infer import infer_batch as IB


class _BigPool:
    def __init__(self):
        self.freed = []

    def free_state_cache(self, ids):
        self.freed.extend(ids)


class _RadixCache:
    def __init__(self):
        self.linear_att_big_page_buffers = _BigPool()
        self.deced = []

    def dec_node_ref_counter(self, node):
        self.deced.append(node)


def test_release_helper_frees_and_clears():
    ctx = IB.InferenceContext.__new__(IB.InferenceContext)
    ctx.radix_cache = _RadixCache()
    req = types.SimpleNamespace(linear_att_len_to_big_page_id=SortedDict({8: 101, 16: 102}))

    ctx._release_pending_linear_att_big_page_ids(req)
    assert sorted(ctx.radix_cache.linear_att_big_page_buffers.freed) == [101, 102]
    assert len(req.linear_att_len_to_big_page_id) == 0

    # idempotent: a second call on an empty dict frees nothing more
    ctx._release_pending_linear_att_big_page_ids(req)
    assert sorted(ctx.radix_cache.linear_att_big_page_buffers.freed) == [101, 102]


def _make_ctx_and_req(monkeypatch, cur_kv_len, cache_len, pending):
    ctx = IB.InferenceContext.__new__(IB.InferenceContext)
    ctx.is_linear_att_mixed_model = True
    ctx.radix_cache = _RadixCache()
    ctx.req_manager = types.SimpleNamespace(req_to_token_indexs=torch.arange(0, 200, dtype=torch.int64).reshape(1, 200))
    # _linear_att_free_req asserts on the *global* g_infer_context, and reads start args
    monkeypatch.setattr(IB, "g_infer_context", ctx)
    monkeypatch.setattr(
        IB,
        "get_env_start_args",
        lambda: types.SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2),
    )
    req = IB.InferReq.__new__(IB.InferReq)
    req.req_idx = 0
    req.shared_kv_node = None
    req.cur_kv_len = cur_kv_len
    req.linear_att_cache_len = cache_len
    req.tail_linear_att_small_page_buffer_id = None
    req.linear_att_len_to_big_page_id = SortedDict(pending)
    return ctx, req


def test_branch_c_releases_pending_big_pages(monkeypatch):
    # big_page_token_num = page(4)*block(2) = 8; cache_len=16 -> tail_big=16.
    # cur_kv_len=8 < tail_big=16 -> branch A and B skipped, fallback branch C taken.
    # pending dict holds the big-page id saved at boundary 8 during prefill.
    ctx, req = _make_ctx_and_req(monkeypatch, cur_kv_len=8, cache_len=16, pending={8: 777})
    free_idx = []
    ctx._linear_att_free_req(free_idx, req)
    assert ctx.radix_cache.linear_att_big_page_buffers.freed == [777], "branch C must release pending big-page ids"
    assert len(req.linear_att_len_to_big_page_id) == 0, "pending dict must be empty after free (invariant)"


def test_cur_kv_len_zero_release(monkeypatch):
    # cur_kv_len == 0 early return must also leave the dict empty (defensive).
    ctx, req = _make_ctx_and_req(monkeypatch, cur_kv_len=0, cache_len=16, pending={8: 555})
    ctx._linear_att_free_req([], req)
    assert ctx.radix_cache.linear_att_big_page_buffers.freed == [555]
    assert len(req.linear_att_len_to_big_page_id) == 0
