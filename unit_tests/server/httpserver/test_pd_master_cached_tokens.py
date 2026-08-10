import asyncio
import copy
import pickle
from contextlib import aclosing
from types import SimpleNamespace

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver_for_pd_master import manager as pd_master_manager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster


def _ignore(*args):
    pass


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


def _collect_fetch_pd_stream(monkeypatch, batches):
    class FakeReqStatus:
        def __init__(self):
            self.up_status_event = asyncio.Event()
            self.prefill_prompt_ids_event = asyncio.Event()
            self.batches = list(batches)

        async def wait_to_ready(self):
            return bool(self.batches)

        async def can_read(self, req_id_to_out_inf):
            return bool(self.batches)

        async def pop_all_tokens(self):
            return self.batches.pop(0)

    req_status = FakeReqStatus()
    monkeypatch.setattr(pd_master_manager, "ReqStatus", lambda *args: req_status)

    manager = object.__new__(HttpServerManagerForPDMaster)
    manager.args = SimpleNamespace(pd_node_id=0)
    manager.req_id_to_out_inf = {}

    async def wait_for_stage(event, request, timeout, group_request_id, stage):
        if stage == "prefill":
            event.prompt_ids = [1, 2, 3, 4]
        else:
            decode_node_info = SimpleNamespace(ready_kv_len=0)
            event.upkv_status = SimpleNamespace(pd_kv_trans_params=pickle.dumps(decode_node_info))

    manager._wait_for_event_or_disconnect = wait_for_stage
    websocket = SimpleNamespace(send_bytes=lambda data: asyncio.sleep(0))
    node = SimpleNamespace(websocket=websocket)
    sampling_params = SimpleNamespace(
        group_request_id=1,
        max_new_tokens=3,
        pd_master_node_id=SimpleNamespace(initialize=_ignore),
    )
    request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))

    async def run():
        results = []
        generator = manager.fetch_pd_stream(node, node, "prompt", sampling_params, None, request)
        async with aclosing(generator):
            async for result in generator:
                results.append(result)
                if result[3].is_finished():
                    break
        return results

    return asyncio.run(run())


def test_prefill_cache_hit_is_copied_to_decode_tokens(monkeypatch):
    batches = [
        [
            (1, "d1", {"count_output_tokens": 1, "node_mode": "decode", "prompt_cache_len": 1}, FinishStatus()),
            (1, "d2", {"count_output_tokens": 2, "node_mode": "decode", "prompt_cache_len": 1}, FinishStatus()),
        ],
        [
            (
                1,
                "p1",
                {"count_output_tokens": 1, "node_mode": "prefill", "prompt_cache_len": 8},
                FinishStatus(FinishStatus.FINISHED_LENGTH),
            )
        ],
        [
            (
                1,
                "d3",
                {"count_output_tokens": 3, "node_mode": "decode", "prompt_cache_len": 1},
                FinishStatus(FinishStatus.FINISHED_STOP),
            )
        ],
    ]

    results = _collect_fetch_pd_stream(monkeypatch, batches)

    assert [result[1] for result in results] == ["p1", "d2", "d3"]
    assert [result[2]["prompt_cache_len"] for result in results] == [8, 8, 8]


def test_finished_decode_is_released_when_prefill_token_is_missing(monkeypatch):
    batches = [
        [
            (1, "d1", {"count_output_tokens": 1, "node_mode": "decode", "prompt_cache_len": 1}, FinishStatus()),
            (
                1,
                "d2",
                {"count_output_tokens": 2, "node_mode": "decode", "prompt_cache_len": 1},
                FinishStatus(FinishStatus.FINISHED_STOP),
            ),
        ]
    ]

    results = _collect_fetch_pd_stream(monkeypatch, batches)

    assert [result[1] for result in results] == ["d1", "d2"]
    assert results[-1][3].is_finished()
