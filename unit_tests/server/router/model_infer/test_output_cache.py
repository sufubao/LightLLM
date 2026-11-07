from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch

from lightllm.server.router.model_infer.infer_batch import InferReq, InferenceContext, g_infer_context
from lightllm.server.router.model_infer import infer_batch
from lightllm.utils.kv_cache_utils import compute_token_list_hash


class _Allocator:
    def __init__(self, slots):
        self.slots = list(slots)
        self.freed = []

    def alloc(self, count):
        if len(self.slots) < count:
            return None
        result = self.slots[:count]
        del self.slots[:count]
        return result

    def free(self, slots):
        self.freed.extend(slots)


class _Event:
    def __init__(self):
        self.recorded = False
        self.synchronized = False

    def record(self):
        self.recorded = True

    def synchronize(self):
        self.synchronized = True


def _make_req(*, input_len=8, cur_kv_len=20, hybrid_cache_len=8):
    req = InferReq.__new__(InferReq)
    req.req_idx = 0
    req.cur_kv_len = cur_kv_len
    req.cur_output_len = cur_kv_len - input_len
    req.hybrid_cache_len = hybrid_cache_len
    req.hybrid_len_to_big_page_id = {}
    req.tail_small_page_buffer_id = None
    req.cache_snapshot_event = None
    req.decode_cache_blocked = False
    req.shm_req = SimpleNamespace(input_len=input_len, shm_prompt_ids=SimpleNamespace(arr=np.arange(64)))
    return req


def test_generated_prefix_hashes_include_all_executed_pages():
    req = _make_req(input_len=8, cur_kv_len=12)

    hashes = req.get_cache_hashes(page_size=4, token_len=12)
    expected = compute_token_list_hash(np.arange(13), 4)

    assert hashes == expected


def test_decode_snapshot_publishes_big_checkpoint_and_reuses_tail(monkeypatch):
    req = _make_req(input_len=8, cur_kv_len=20, hybrid_cache_len=8)
    req.tail_small_page_buffer_id = None
    calls = []
    big_allocator = _Allocator([21])
    small_allocator = _Allocator([31, 32])

    context = InferenceContext()
    context.args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2, mtp_mode=None)
    context.is_hybrid_att_model = True
    context.radix_cache = SimpleNamespace(free_one_small_page_buffer=lambda: None)
    context.get_overlap_stream = lambda: None
    context.att_state_manager = SimpleNamespace(
        bounded_att_big_page_allocator=big_allocator,
        bounded_att_small_page_allocator=small_allocator,
        save_bounded_att_runtime_to_pages=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _Event)

    context.save_decode_state_to_cache([req, req], {req.req_idx: 11})

    assert [call["state_token_lens"] for call in calls] == [[12], [16], [20]]
    assert [call["page_kind"] for call in calls] == ["small", "big", "small"]
    assert req.hybrid_cache_len == 20
    assert req.hybrid_len_to_big_page_id == {16: 21}
    assert req.tail_small_page_buffer_id == 32
    assert small_allocator.freed == [31]
    assert req.cache_snapshot_event.recorded


def test_decode_snapshot_stops_after_missing_big_checkpoint(monkeypatch):
    req = _make_req(input_len=8, cur_kv_len=24, hybrid_cache_len=12)
    big_allocator = _Allocator([])
    small_allocator = _Allocator([31])
    calls = []

    context = InferenceContext()
    context.args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2, mtp_mode=None)
    context.is_hybrid_att_model = True
    context.radix_cache = SimpleNamespace(free_one_small_page_buffer=lambda: None)
    context.get_overlap_stream = lambda: None
    context.att_state_manager = SimpleNamespace(
        bounded_att_big_page_allocator=big_allocator,
        bounded_att_small_page_allocator=small_allocator,
        save_bounded_att_runtime_to_pages=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())

    context.save_decode_state_to_cache([req], {req.req_idx: 23})

    assert calls == []
    assert req.hybrid_cache_len == 12
    assert req.decode_cache_blocked


def test_decode_snapshot_skips_speculative_requests(monkeypatch):
    req = _make_req(input_len=8, cur_kv_len=20, hybrid_cache_len=8)
    calls = []
    context = InferenceContext()
    context.args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2, mtp_mode="eagle_with_att")
    context.is_hybrid_att_model = True
    context.radix_cache = SimpleNamespace(free_one_small_page_buffer=lambda: None)
    context.att_state_manager = SimpleNamespace(
        bounded_att_big_page_allocator=_Allocator([21]),
        bounded_att_small_page_allocator=_Allocator([31]),
        save_bounded_att_runtime_to_pages=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())

    context.save_decode_state_to_cache([req], {req.req_idx: 15})

    assert calls == []
    assert req.hybrid_cache_len == 8


def test_hybrid_release_uses_generated_hashes(monkeypatch):
    captured = {}

    class _Radix:
        def insert(self, key, value, *, block_hashs, block_state_idxs, len_to_big_page_id):
            captured["key"] = key
            captured["block_hashs"] = block_hashs
            captured["state_idxs"] = block_state_idxs
            return len(key), None

        def dec_node_ref_counter(self, node):
            raise AssertionError("no shared node expected")

    req = _make_req(input_len=8, cur_kv_len=12, hybrid_cache_len=12)
    req.hold_kv_len = 12
    req.tail_small_page_buffer_id = 31
    req.shared_kv_node = None
    req.get_cache_hashes = lambda page_size, token_len: [101, 102, 103]
    req.shm_req.hybrid_token_hash_list = SimpleNamespace(get_all=lambda: [1, 2, 3])

    context = InferenceContext()
    context.args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2)
    context.is_hybrid_att_model = True
    context.radix_cache = _Radix()
    context.req_manager = SimpleNamespace(req_to_token_indexs=torch.arange(32).view(1, -1))
    context.att_state_manager = SimpleNamespace()
    monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: context.args)
    monkeypatch.setattr("lightllm.server.router.model_infer.infer_batch.g_infer_context", context)

    released = []
    context._hybrid_att_free_req(released, req)

    assert captured["key"].tolist() == list(range(12))
    assert captured["block_hashs"] == [101, 102, 103]
    assert captured["state_idxs"] == [None, None, 31]
    assert req.tail_small_page_buffer_id is None


def test_request_release_waits_for_snapshot_event():
    event = _Event()
    req = _make_req(input_len=0, cur_kv_len=0, hybrid_cache_len=0)
    req.hold_kv_len = 0
    req.cache_snapshot_event = event

    context = InferenceContext()
    context.radix_cache = None
    context.req_manager = SimpleNamespace(req_to_token_indexs=torch.empty((1, 0), dtype=torch.int64))
    context.is_hybrid_att_model = False
    context.free_a_req_mem([], req)

    assert event.synchronized
    assert req.cache_snapshot_event is None
