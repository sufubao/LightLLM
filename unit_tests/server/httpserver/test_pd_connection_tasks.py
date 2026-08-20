import asyncio
from types import SimpleNamespace

import pytest

from lightllm.server.httpserver.async_queue import AsyncQueue
from lightllm.server.httpserver.pd_loop import (
    _pd_process_generate,
    _recv_or_raise_on_background_failure,
)


def test_background_failure_interrupts_blocked_pd_receive():
    async def run():
        recv_started = asyncio.Event()
        recv_cancelled = asyncio.Event()

        class BlockingWebsocket:
            async def recv(self):
                recv_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    recv_cancelled.set()
                    raise

        websocket = BlockingWebsocket()
        failure = asyncio.get_running_loop().create_future()
        receive_task = asyncio.create_task(
            _recv_or_raise_on_background_failure(websocket, (failure,))
        )
        await recv_started.wait()

        failure.set_exception(RuntimeError("generation failed"))
        with pytest.raises(RuntimeError, match="generation failed"):
            await receive_task
        assert recv_cancelled.is_set()

    asyncio.run(run())


def test_pd_generation_failure_is_not_swallowed():
    async def run():
        class FailingManager:
            args = SimpleNamespace(run_mode="prefill")

            async def generate(self, **_kwargs):
                yield 1, "token", {}, None
                raise RuntimeError("generation failed")

        sampling_params = SimpleNamespace(group_request_id=123)
        with pytest.raises(RuntimeError, match="generation failed"):
            await _pd_process_generate(
                manager=FailingManager(),
                prompt="prompt",
                sampling_params=sampling_params,
                multimodal_params={},
                forwarding_queue=AsyncQueue(),
                pd_upload_websocket=object(),
                pd_event=asyncio.Event(),
            )

    asyncio.run(run())
