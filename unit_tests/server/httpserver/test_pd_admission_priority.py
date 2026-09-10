import asyncio
import threading
import multiprocessing
from types import SimpleNamespace

import numpy as np
import pytest

from lightllm.server.core.objs import SamplingParams
from lightllm.server.core.objs.shm_req_manager import ShmReqManager, ReqLinkedListManager
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.error_utils import ServerBusyError


def _pool(capacity=2):
    pool = object.__new__(ShmReqManager)
    pool.max_req_num = capacity
    pool.manager_lock = threading.Lock()
    pool.priority_waiters = SimpleNamespace(arr=np.zeros(3, dtype=np.int32))
    pool.alloc_state_shm = SimpleNamespace(arr=np.zeros(capacity, dtype=np.int32))
    pool.linked_req_manager = object.__new__(ReqLinkedListManager)
    pool.linked_req_manager.size = capacity + 1
    pool.linked_req_manager._values = np.zeros((capacity + 1, 1), dtype=np.int64)
    pool.linked_req_manager._initialize_values()
    return pool


def _worker(pool):
    manager = object.__new__(HttpServerManager)
    manager.args = SimpleNamespace(run_mode="decode")
    manager.is_multinode_tp_slave = False
    manager.shm_req_manager = pool
    return manager


def _continuation_process(pool, registered, allocate, result):
    pool.register_pd_waiter(0)
    try:
        registered.set()
        if allocate.wait(5):
            result.put(pool.alloc_pd_req_indexes(1, 0))
    finally:
        pool.unregister_pd_waiter(0)


def test_priority_gate_is_visible_across_processes():
    ctx = multiprocessing.get_context("fork")
    pool = _pool(1)
    pool.manager_lock = ctx.Lock()
    pool.priority_waiters.arr = np.frombuffer(ctx.RawArray("i", 3), dtype=np.int32)
    pool.alloc_state_shm.arr = np.frombuffer(ctx.RawArray("i", 1), dtype=np.int32)
    pool.linked_req_manager._values = np.frombuffer(ctx.RawArray("q", 2), dtype=np.int64).reshape(2, 1)
    pool.linked_req_manager._initialize_values()
    registered, allocate, result = ctx.Event(), ctx.Event(), ctx.Queue()
    child = ctx.Process(target=_continuation_process, args=(pool, registered, allocate, result))
    child.start()
    try:
        assert registered.wait(5)
        assert pool.alloc_pd_req_indexes(1, 1) is None
        assert pool.alloc_pd_req_indexes(1, 2) is None
        allocate.set()
        indexes = result.get(timeout=5)
        child.join(timeout=5)
        assert child.exitcode == 0
        assert pool.alloc_state_shm.arr[indexes[0]] == 1
        assert pool.priority_waiters.arr.tolist() == [0, 0, 0]
        pool.release_req_index(indexes[0])
        assert pool.is_idle()
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        result.close()
        result.join_thread()


def test_shared_priority_blocks_new_requests_until_continuation_acquires_slot():
    pool = _pool(1)
    # Separate worker objects see the same allocator state, rather than local asyncio queues.
    other_pool = object.__new__(ShmReqManager)
    other_pool.__dict__.update(pool.__dict__)
    first, second = _worker(pool), _worker(other_pool)

    async def run():
        occupied = pool.alloc_req_index()
        normal = asyncio.create_task(first._alloc_shm_req_indexes(1))
        cached = asyncio.create_task(second._alloc_shm_req_indexes(1, pd_high_priority_request=True))
        continuation = asyncio.create_task(first._alloc_shm_req_indexes(1, pd_is_continuation=True))
        await asyncio.sleep(0)
        assert pool.priority_waiters.arr.tolist() == [1, 1, 1]
        pool.release_req_index(occupied)
        indexes = await asyncio.wait_for(continuation, 1)
        assert not cached.done() and not normal.done()
        pool.release_req_index(indexes[0])
        indexes = await asyncio.wait_for(cached, 1)
        assert not normal.done()
        pool.release_req_index(indexes[0])
        indexes = await asyncio.wait_for(normal, 1)
        pool.release_req_index(indexes[0])

    asyncio.run(run())
    assert pool.priority_waiters.arr.tolist() == [0, 0, 0]
    assert pool.is_idle()


def test_cancelled_continuation_releases_priority_gate():
    pool = _pool(1)
    manager = _worker(pool)

    async def run():
        occupied = pool.alloc_req_index()
        pending = asyncio.create_task(manager._alloc_shm_req_indexes(1, pd_is_continuation=True))
        await asyncio.sleep(0)
        assert pool.priority_waiters.arr[0] == 1
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        pool.release_req_index(occupied)
        indexes = await asyncio.wait_for(manager._alloc_shm_req_indexes(1), 1)
        pool.release_req_index(indexes[0])

    asyncio.run(run())
    assert pool.priority_waiters.arr.tolist() == [0, 0, 0]
    assert pool.is_idle()


def test_group_allocation_holds_no_partial_resources_when_waiting():
    pool = _pool(2)
    occupied = pool.alloc_req_index()
    assert pool.alloc_pd_req_indexes(2, 0) is None
    assert pool.alloc_state_shm.arr.sum() == 1
    # The other slot remains available even after a failed group allocation.
    available = pool.alloc_pd_req_indexes(1, 0)
    assert available is not None
    pool.release_req_index(occupied)
    pool.release_req_index(available[0])
    assert len(pool.alloc_pd_req_indexes(2, 0)) == 2


def test_new_request_still_times_out_and_releases_waiter():
    pool = _pool(1)
    manager = _worker(pool)
    pool.alloc_req_index()

    async def run():
        with pytest.raises(ServerBusyError):
            await manager._alloc_shm_req_indexes(1, pd_node_resource_wait_timeout_seconds=0)

    asyncio.run(run())
    assert pool.priority_waiters.arr.tolist() == [0, 0, 0]


def _req(name, continuation=False, cached=False):
    params = SamplingParams()
    params.pd_is_continuation = continuation
    params.pd_high_priority_request = cached
    return SimpleNamespace(name=name, sample_params=params)


def test_router_and_inference_prioritize_continuations_with_fifo_within_class():
    requests = [
        _req("normal"),
        _req("cache1", cached=True),
        _req("cont1", True, True),
        _req("cache2", cached=True),
        _req("cont2", True, True),
    ]
    queue = object.__new__(BaseQueue)
    queue.dp_index = 0
    queue.waiting_req_list = []
    for req in requests:
        queue.extend([req])
    expected = ["cont1", "cont2", "cache1", "cache2", "normal"]
    assert [req.name for req in queue.waiting_req_list] == expected
    backend = object.__new__(ModeBackend)
    ordered = backend._reorder_pd_high_priority_reqs([SimpleNamespace(shm_req=req) for req in requests])
    assert [req.shm_req.name for req in ordered] == expected
