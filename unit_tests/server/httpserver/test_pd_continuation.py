import asyncio
import pickle
import time
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver_for_pd_master import manager as master_module
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, ReqStatus
from lightllm.server.pd_io_struct import ObjType
from lightllm.utils.error_utils import ClientDisconnected, ServerBusyError
from lightllm.utils.envs_utils import get_pd_node_continuation_resource_wait_timeout_seconds
from unit_tests.server.httpserver.test_pd_master_cached_tokens import _make_manager


def _params():
    params = SamplingParams()
    params.n = params.best_of = 1
    params.max_new_tokens = 3
    return params


def _token(params, text, finish=FinishStatus.NO_FINISH):
    return params.group_request_id, text, {"prompt_tokens": 10, "prompt_cache_len": 4}, FinishStatus(finish)


def _stream(manager):
    return manager.generate(
        "hello",
        _params(),
        SimpleNamespace(images=[], audios=[], verify_and_preload=AsyncMock()),
        SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
    )


def test_continuation_busy_retries_only_unstarted_segment(monkeypatch):
    manager = _make_manager(monkeypatch)
    manager.pd_node_continuation_resource_wait_timeout_seconds = -1
    p_node = SimpleNamespace(dispatched_prompt_chars=0, dispatched_req_num=0, websocket=AsyncMock())
    d_node = SimpleNamespace(websocket=AsyncMock())
    manager.req_id_to_out_inf = {}
    selection = master_module.PDSelectionExtraInfo()
    manager.select_p_d_node = AsyncMock(return_value=(p_node, d_node, selection))
    attempts, closed = [], []

    async def wait(_p, _d, _start, prompt, params, *_args):
        attempts.append(
            (
                prompt,
                params.max_new_tokens,
                params.pd_is_continuation,
                params.pd_node_resource_wait_timeout_seconds,
                params.group_request_id,
            )
        )
        attempt = len(attempts)
        try:
            if attempt == 1:
                yield _token(params, "A")
                yield _token(params, "", FinishStatus.FINISHED_PD_DECODE_CAPACITY)
            elif attempt == 2:
                raise ServerBusyError()
            else:
                yield _token(params, "B", FinishStatus.FINISHED_STOP)
        finally:
            closed.append(attempt)

    manager._wait_to_token_package = wait

    async def run():
        return [result async for result in _stream(manager)]

    results = asyncio.run(run())
    assert "".join(result[1] for result in results) == "AB"
    assert len({result[0] for result in results}) == 1
    assert [attempt[:4] for attempt in attempts] == [
        ("hello", 3, False, -1),
        ("helloA", 2, True, -1),
        ("helloA", 2, True, -1),
    ]
    assert len({attempt[4] for attempt in attempts}) == 3
    for node in (p_node, d_node):
        commands = [pickle.loads(call.args[0]) for call in node.websocket.send_bytes.await_args_list]
        assert commands == [(ObjType.ABORT, attempts[1][4])]
    assert closed == [1, 2, 3]
    assert p_node.dispatched_prompt_chars == p_node.dispatched_req_num == 0
    assert manager.running_request_count == 0
    manager.select_p_d_node.assert_awaited_once()


def test_busy_after_output_in_continuation_is_not_replayed(monkeypatch):
    manager = _make_manager(monkeypatch)
    manager.abort = AsyncMock()
    attempts = []
    delivered = []

    async def wait(_p, _d, _start, _prompt, params, *_args):
        attempts.append(params.group_request_id)
        yield _token(params, "A" if len(attempts) == 1 else "B")
        if len(attempts) == 1:
            yield _token(params, "", FinishStatus.FINISHED_PD_DECODE_CAPACITY)
        else:
            raise ServerBusyError()

    manager._wait_to_token_package = wait

    async def run():
        with pytest.raises(ServerBusyError):
            async for result in _stream(manager):
                delivered.append(result[1])

    asyncio.run(run())
    assert delivered == ["A", "B"]
    assert len(attempts) == 2
    manager.abort.assert_awaited_once()


@pytest.mark.parametrize("stop", ["deadline", "cancel", "close"])
def test_waiting_continuation_cleanup_on_request_end(monkeypatch, stop):
    manager = _make_manager(monkeypatch)
    manager.abort = AsyncMock()
    monkeypatch.setattr(master_module, "get_pd_request_timeout_seconds", lambda: 0.05 if stop == "deadline" else -1)
    entered = asyncio.Event()
    closed = []
    attempts = []

    async def wait(_p, _d, _start, _prompt, params, *_args):
        attempts.append(params.group_request_id)
        try:
            if len(attempts) == 1:
                yield _token(params, "A")
                yield _token(params, "", FinishStatus.FINISHED_PD_DECODE_CAPACITY)
            else:
                entered.set()
                await asyncio.Event().wait()
        finally:
            closed.append(params.group_request_id)

    manager._wait_to_token_package = wait

    async def run():
        stream = _stream(manager)
        assert (await stream.__anext__())[1] == "A"
        await asyncio.wait_for(entered.wait(), timeout=1)
        if stop == "close":
            await stream.aclose()
        else:
            next_token = asyncio.create_task(stream.__anext__())
            if stop == "cancel":
                await asyncio.sleep(0)
                next_token.cancel()
            with pytest.raises(asyncio.CancelledError if stop == "cancel" else TimeoutError):
                await next_token
            await stream.aclose()

    asyncio.run(run())
    assert closed == attempts
    assert len(attempts) == 2
    manager.abort.assert_awaited_once()
    assert manager.running_request_count == 0


def test_continuation_handshake_survives_original_60_second_limit(monkeypatch):
    manager = object.__new__(HttpServerManagerForPDMaster)
    clock = [0.0]
    monkeypatch.setattr(master_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def run():
        event = asyncio.Event()
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        task = asyncio.create_task(manager._wait_for_event_or_disconnect(event, request, None, 1, "prefill"))
        await asyncio.sleep(0)
        clock[0] = 61.0
        event.set()
        await task
        assert request.is_disconnected.await_count >= 2

    asyncio.run(run())


def test_disconnected_client_stops_continuation_handshake():
    manager = object.__new__(HttpServerManagerForPDMaster)

    async def run():
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
        with pytest.raises(ClientDisconnected):
            await manager._wait_for_event_or_disconnect(asyncio.Event(), request, None, 1, "decode")

    asyncio.run(run())


def test_node_disconnect_wakes_pending_continuation():
    manager = object.__new__(HttpServerManagerForPDMaster)
    manager.pd_manager = MagicMock()

    async def run():
        state = ReqStatus(1, SimpleNamespace(node_id=10), SimpleNamespace(node_id=20))
        manager.req_id_to_out_inf = {1: state}
        await manager.remove_pd({"node_id": 20})
        assert state.up_status_event.is_set()
        assert state.prefill_prompt_ids_event.is_set()
        assert state.event.is_set()
        with pytest.raises(RuntimeError, match="disconnected"):
            state.raise_if_error()

    asyncio.run(run())


@pytest.mark.parametrize("continuation, expected", [(False, [60, 180]), (True, [None, None])])
def test_master_applies_continuation_policy_to_both_handshakes(continuation, expected):
    manager = object.__new__(HttpServerManagerForPDMaster)
    manager.args = SimpleNamespace(pd_node_id=0)
    manager.req_id_to_out_inf = {}
    timeouts = []
    params = _params()
    params.pd_is_continuation = continuation
    params.group_request_id = 12

    async def wait(event, request, timeout, group_request_id, stage):
        timeouts.append(timeout)
        if stage == "prefill":
            event.prompt_ids = [1, 2, 3]
        else:
            await manager.req_id_to_out_inf[group_request_id].set_error("end of handshake test")

    manager._wait_for_event_or_disconnect = wait

    async def run():
        node = SimpleNamespace(websocket=AsyncMock())
        stream = manager.fetch_pd_stream(node, node, "hello", params, None, None)
        with pytest.raises(RuntimeError, match="end of handshake test"):
            await stream.__anext__()

    asyncio.run(run())
    assert timeouts == expected


def test_external_request_cannot_claim_continuation_priority():
    params = SamplingParams()
    params.init(MagicMock(), pd_is_continuation=True, pd_high_priority_request=True)
    assert not params.pd_is_continuation
    assert not params.pd_high_priority_request
    params.pd_is_continuation = True
    restored = pickle.loads(pickle.dumps(params))
    assert restored.pd_is_continuation


@pytest.mark.parametrize("value, expected", [(None, -1), ("-1", -1), ("60", 60)])
def test_continuation_resource_timeout_supports_unlimited_wait(monkeypatch, value, expected):
    name = "LIGHTLLM_PD_NODE_CONTINUATION_RESOURCE_WAIT_TIMEOUT_SECONDS"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    get_pd_node_continuation_resource_wait_timeout_seconds.cache_clear()
    try:
        assert get_pd_node_continuation_resource_wait_timeout_seconds() == expected
    finally:
        get_pd_node_continuation_resource_wait_timeout_seconds.cache_clear()
