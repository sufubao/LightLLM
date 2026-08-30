import asyncio
from types import SimpleNamespace

import zmq

from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.server.router.manager import RouterManager


class _QueuedSocket:
    def __init__(self, values):
        self.values = list(values)

    def recv_pyobj(self, flags):
        assert flags == zmq.NOBLOCK
        if not self.values:
            raise zmq.Again()
        return self.values.pop(0)


def test_drain_new_requests_returns_count_and_adds_every_group():
    groups = [GroupReqIndexes(index, None, [], 0.0) for index in range(3)]
    manager = RouterManager.__new__(RouterManager)
    manager.zmq_recv_socket = _QueuedSocket(groups)
    added = []
    manager._add_req = added.append

    assert manager._drain_new_requests() == 3
    assert [group.group_req_id for group in added] == [0, 1, 2]
    assert manager.recv_max_count == 64


def test_idle_server_waits_until_a_quiet_window_before_scheduling():
    manager = RouterManager.__new__(RouterManager)
    manager.running_batch = None
    manager.schedule_new_batch = None
    manager.idle_batch_coalesce_quiet_time = 0.001
    manager.idle_batch_coalesce_max_wait = 0.05
    manager.args = SimpleNamespace(enable_rl=False, running_max_req_size=3)
    manager.is_multinode_tp = False
    manager._get_paused_req_num = lambda: 0

    drained_counts = iter([1, 2])
    manager._drain_new_requests = lambda: next(drained_counts)
    scheduled = []
    manager._generate_new_batch = lambda: scheduled.append(True)

    asyncio.run(manager._recv_new_reqs_and_schedule())

    assert scheduled == [True]
    assert list(drained_counts) == []


def test_idle_single_request_stops_after_one_quiet_window():
    manager = RouterManager.__new__(RouterManager)
    manager.running_batch = None
    manager.schedule_new_batch = None
    manager.idle_batch_coalesce_quiet_time = 0.001
    manager.idle_batch_coalesce_max_wait = 0.05
    manager.args = SimpleNamespace(enable_rl=False, running_max_req_size=8)
    manager.is_multinode_tp = False
    manager._get_paused_req_num = lambda: 0

    drained_counts = iter([1, 0])
    manager._drain_new_requests = lambda: next(drained_counts)
    scheduled = []
    manager._generate_new_batch = lambda: scheduled.append(True)

    asyncio.run(manager._recv_new_reqs_and_schedule())

    assert scheduled == [True]
    assert list(drained_counts) == []


def test_step_filters_finished_batch_before_receiving_new_requests():
    manager = RouterManager.__new__(RouterManager)
    manager.schedule_new_batch = None
    manager.is_multinode_tp = False
    events = []

    manager._filter_reqs_from_running_batch = lambda: events.append("filter")

    async def recv_and_schedule():
        events.append("receive")

    async def write_profiler_cmds():
        events.append("profiler")

    manager._recv_new_reqs_and_schedule = recv_and_schedule
    manager._write_profiler_cmds = write_profiler_cmds
    manager._get_aborted_reqs_from_running_batch = lambda: []
    manager._get_stop_str_reqs_from_running_batch = lambda: []

    asyncio.run(manager._step())

    assert events == ["filter", "receive", "profiler"]
