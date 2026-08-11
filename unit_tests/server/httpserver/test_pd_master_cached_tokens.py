import asyncio
import copy
import pickle
from types import SimpleNamespace

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
import lightllm.server.httpserver_for_pd_master.manager as pd_master_manager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster


def _make_manager(monkeypatch):
    monkeypatch.setattr(
        "lightllm.server.httpserver.manager.HttpServerManager._check_and_repair_length",
        classmethod(lambda cls, *a, **k: asyncio.sleep(0)),
    )
    monkeypatch.setattr(SamplingParams, "from_buffer_copy", classmethod(lambda cls, other: copy.copy(other)))
    mgr = object.__new__(HttpServerManagerForPDMaster)
    mgr.running_request_count = 0
    counter = [0]

    def gen_id():
        counter[0] += 1
        return counter[0]

    mgr.id_gen = SimpleNamespace(generate_id=gen_id)
    mgr.metric_client = SimpleNamespace(counter_inc=lambda *a, **k: None, histogram_observe=lambda *a, **k: None)
    mgr.tokens = lambda *a, **k: 10
    mgr._log_req_header = lambda *a, **k: asyncio.sleep(0)
    mgr.recorded_cache_hit_rates = []
    mgr.pd_manager = SimpleNamespace(
        selector=SimpleNamespace(record_prompt_cache_hit_rate=mgr.recorded_cache_hit_rates.append)
    )
    p_node = SimpleNamespace(dispatched_prompt_chars=0, dispatched_req_num=0)
    mgr.select_p_d_node = lambda *a, **k: asyncio.sleep(0, result=(p_node, 1))
    mgr.remove_req = lambda *a, **k: asyncio.sleep(0)
    return mgr


def _collect(mgr, sampling_params, monkeypatch, split):
    mgr._split_max_new_tokens = lambda *a, **k: list(split)

    async def fake_wait(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
        sub_req_id = sp.group_request_id
        hit = sp.max_new_tokens * 10
        yield sub_req_id, "x", {"prompt_tokens": 100, "prompt_cache_len": hit}, FinishStatus()
        for _ in range(2):
            yield sub_req_id, "y", {"prompt_tokens": 100, "prompt_cache_len": 0}, FinishStatus()
        yield sub_req_id, "z", {"prompt_tokens": 100, "prompt_cache_len": 0}, FinishStatus(FinishStatus.FINISHED_STOP)

    monkeypatch.setattr(mgr, "_wait_to_token_package", fake_wait)

    async def run():
        out = []
        async for sub_id, out_str, metadata, finish in mgr.generate(
            prompt="hello",
            sampling_params=sampling_params,
            multimodal_params=SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
            request=None,
        ):
            out.append(metadata.get("prompt_cache_len", -1))
        return out

    return asyncio.run(run())


def test_single_block_prefill_hit_persists_past_decode_zeros(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sp = SamplingParams()
    sp.n = 1
    sp.max_new_tokens = 3
    sp.best_of = 1
    sp.group_request_id = 0
    cached = _collect(mgr, sp, monkeypatch, split=[3])
    assert cached and all(c == 30 for c in cached), cached
    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.3)]


def test_multi_block_keeps_first_block_hit(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sp = SamplingParams()
    sp.n = 1
    sp.max_new_tokens = 5
    sp.best_of = 1
    sp.group_request_id = 0
    cached = _collect(mgr, sp, monkeypatch, split=[3, 2])
    assert cached[-1] == 30, cached
    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.3)]


def test_missing_prefill_token_flushes_while_decode_is_still_running(monkeypatch):
    group_request_id = 1
    clock = [0.0]

    def decode_token(index):
        return (
            group_request_id,
            str(index),
            {"count_output_tokens": index, "node_mode": "decode", "prompt_cache_len": 0},
            FinishStatus(),
        )

    class OutputBatches:
        def __init__(self):
            self.batches = [(0.0, [decode_token(1)]), (6.0, [decode_token(2)]), (7.0, [decode_token(3)])]

        async def wait_to_get_all_data(self, timeout):
            assert self.batches, "decode output was not released after the prefill-token deadline"
            clock[0], batch = self.batches.pop(0)
            return batch

    async def send_bytes(_):
        pass

    manager = object.__new__(HttpServerManagerForPDMaster)
    manager.args = SimpleNamespace(pd_node_id=7)
    manager.req_id_to_out_inf = {}
    p_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=send_bytes))
    d_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=send_bytes))

    async def wait_for_stage(event, request, timeout, group_request_id, stage):
        if stage == "prefill":
            event.prompt_ids = [1, 2, 3, 4]
        else:
            decode_node_info = SimpleNamespace(ready_kv_len=0)
            event.upkv_status = SimpleNamespace(pd_kv_trans_params=pickle.dumps(decode_node_info))
            manager.req_id_to_out_inf[group_request_id].out_tokens = OutputBatches()

    manager._wait_for_event_or_disconnect = wait_for_stage
    monkeypatch.setattr(pd_master_manager.time, "monotonic", lambda: clock[0])

    sampling_params = SamplingParams()
    sampling_params.group_request_id = group_request_id
    sampling_params.max_new_tokens = 2048

    async def is_disconnected():
        return False

    async def collect():
        stream = manager.fetch_pd_stream(
            p_node,
            d_node,
            prompt="prompt",
            sampling_params=sampling_params,
            multimodal_params=SimpleNamespace(),
            request=SimpleNamespace(is_disconnected=is_disconnected),
        )
        try:
            return [await stream.__anext__() for _ in range(3)]
        finally:
            await stream.aclose()

    outputs = asyncio.run(collect())
    assert [output for _, output, _, _ in outputs] == ["1", "2", "3"]
    assert all(not finish_status.is_finished() for _, _, _, finish_status in outputs)
