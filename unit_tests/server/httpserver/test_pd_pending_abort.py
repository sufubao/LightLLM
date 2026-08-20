import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.httpserver.pd_loop import _abort_pd_request


def test_pd_abort_cancels_request_before_http_manager_registration():
    async def run():
        manager = MagicMock()
        manager.abort = AsyncMock(return_value=False)
        request_started = asyncio.Event()

        async def wait_for_request_slot():
            request_started.set()
            await asyncio.Future()

        generation_task = asyncio.create_task(wait_for_request_slot())
        generation_tasks = {123: generation_task}
        await request_started.wait()

        assert await _abort_pd_request(manager, 123, generation_tasks)
        assert generation_task.cancelling() == 1
        assert await _abort_pd_request(manager, 123, generation_tasks)
        assert generation_task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await generation_task

        assert manager.abort.await_count == 2
        manager.abort.assert_awaited_with(123)
        assert generation_task.cancelled()

    asyncio.run(run())


def test_cancelled_slot_allocation_releases_partially_allocated_indexes():
    async def run():
        manager = HttpServerManager.__new__(HttpServerManager)
        manager.shm_req_manager = MagicMock()
        manager.shm_req_manager.async_release_req_index = AsyncMock()
        manager.shm_req_manager.async_put_back_req_obj = AsyncMock()
        manager.tokenizer = MagicMock()
        manager.args = MagicMock(chunked_prefill_size=16)

        waiting_for_second_slot = asyncio.Event()
        allocation_count = 0

        async def alloc_req_index():
            nonlocal allocation_count
            allocation_count += 1
            if allocation_count == 1:
                return 7
            waiting_for_second_slot.set()
            return None

        manager.shm_req_manager.async_alloc_req_index = alloc_req_index
        sampling_params = MagicMock(n=2)
        allocation_task = asyncio.create_task(
            manager._alloc_req_objs(123, [1, 2], sampling_params)
        )
        await waiting_for_second_slot.wait()
        allocation_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await allocation_task

        manager.shm_req_manager.async_release_req_index.assert_awaited_once_with(7)
        manager.shm_req_manager.async_put_back_req_obj.assert_not_awaited()

    asyncio.run(run())
