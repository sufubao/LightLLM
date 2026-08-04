import asyncio
import pickle
from types import SimpleNamespace

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver_for_pd_master import manager as pd_master_manager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster


class _FakeReqStatus:
    def __init__(self, req_id, p_node, d_node):
        self.prefill_prompt_ids_event = SimpleNamespace(prompt_ids=[1, 2, 3])
        self.up_status_event = SimpleNamespace(
            upkv_status=SimpleNamespace(pd_kv_trans_params=pickle.dumps(SimpleNamespace()))
        )
        self.token_batches = [
            [
                (
                    req_id,
                    "decode-first",
                    {"count_output_tokens": 1, "node_mode": "decode", "prompt_cache_len": 0},
                    FinishStatus(),
                )
            ],
            [(req_id, "decode-second", {"count_output_tokens": 2, "node_mode": "decode"}, FinishStatus())],
            [
                (
                    req_id,
                    "prefill-first",
                    {"count_output_tokens": 1, "node_mode": "prefill", "prompt_cache_len": 7},
                    FinishStatus(FinishStatus.FINISHED_LENGTH),
                )
            ],
        ]

    async def wait_to_ready(self):
        await asyncio.sleep(0)

    async def can_read(self, req_id_to_out_inf):
        return bool(self.token_batches)

    async def pop_all_tokens(self):
        return self.token_batches.pop(0)


class _FakeWebSocket:
    async def send_bytes(self, data):
        pass


def _make_manager(monkeypatch):
    monkeypatch.setattr(pd_master_manager, "ReqStatus", _FakeReqStatus)

    async def _noop(*a, **k):
        return None

    mgr = object.__new__(HttpServerManagerForPDMaster)
    mgr.args = SimpleNamespace(pd_node_id=1)
    mgr.req_id_to_out_inf = {}
    mgr._wait_for_event_or_disconnect = _noop
    return mgr


def test_decode_first_token_uses_later_prefill_cache_metadata(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sampling_params = SamplingParams()
    sampling_params.group_request_id = 11
    sampling_params.max_new_tokens = 2
    node = SimpleNamespace(websocket=_FakeWebSocket())
    request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))

    async def run():
        stream = mgr.fetch_pd_stream(node, node, "hello", sampling_params, SimpleNamespace(), request)
        first = await anext(stream)
        second = await anext(stream)
        await stream.aclose()
        return first, second

    first, second = asyncio.run(run())
    assert first[1] == "decode-first"
    assert first[2]["prompt_cache_len"] == 7
    assert second[1] == "decode-second"
