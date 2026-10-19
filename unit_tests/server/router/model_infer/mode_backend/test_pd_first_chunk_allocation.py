import copy
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.server.pd_io_struct import PDChunckedTransTask, PDChunckedTransTaskGroup
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_impl
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_impl import PDDecodeNode
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_impl_for_dp import PDDPForDecodeNode
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_trans_process import _DecodeTransModule
from lightllm.server.router.req_queue.chunked_prefill.impl_for_pd_decode import PDDecodeQueue


def task(start=0):
    return PDChunckedTransTask(
        request_id=1,
        start_kv_index=start,
        end_kv_index=start + 4,
        time_out_secs=180,
        pd_master_node_id=0,
        prefill_dp_index=0,
        decode_dp_index=0,
        src_device_id=0,
        dst_device_id=0,
        mem_indexes=[-1] * 4,
        prefill_agent_name=None,
        prefill_agent_metadata=None,
        prefill_num_pages=None,
        prefill_page_reg_desc=None,
        decode_agent_name=None,
        decode_agent_metadata=None,
        decode_num_pages=None,
        decode_page_reg_desc=None,
        first_gen_token_id=None,
        first_gen_token_logprob=None,
        defer_allocation=True,
    )


def module_with_tasks(*tasks):
    module = _DecodeTransModule.__new__(_DecodeTransModule)
    module.waiting_dict_lock = threading.Lock()
    module.waiting_dict = {t.get_key(): t for t in tasks}
    module.task_out_queue = queue.Queue()
    module.request_page_task_queue = queue.Queue()
    module.failed_queue = queue.Queue()
    return module


def allocated_group(*tasks):
    updates = [copy.copy(t) for t in tasks]
    for t in updates:
        t.mem_indexes = list(range(t.start_kv_index, t.end_kv_index))
        t.defer_allocation = False
    return PDChunckedTransTaskGroup(updates, allocation_update=True)


def test_write_waits_for_real_indexes_and_duplicate_update_does_not_requeue():
    first, second = task(), task(4)
    module = module_with_tasks(first, second)
    remote = copy.copy(first)
    remote.prefill_agent_name = "peer"
    module._handle_write_request(remote)
    module._handle_write_request(remote)
    ret = module.task_out_queue.get_nowait()
    assert ret.allocation_requested and not ret.has_error
    assert module.task_out_queue.empty()
    assert module.request_page_task_queue.empty()
    assert first.mem_indexes == [-1] * 4
    group = allocated_group(first, second)
    module._apply_kv_allocation(group)
    module._apply_kv_allocation(group)
    assert module.request_page_task_queue.get_nowait() is first
    assert first.mem_indexes == [0, 1, 2, 3]
    assert first.prefill_agent_name == "peer"
    assert module.request_page_task_queue.empty()
    module._handle_write_request(copy.copy(second))
    assert module.request_page_task_queue.get_nowait() is second
    assert second.mem_indexes == [4, 5, 6, 7]
    assert module.task_out_queue.empty()


@pytest.mark.parametrize("failure", ["abort", "timeout"])
def test_failure_before_allocation_does_not_resurrect_tasks(failure):
    first, second = task(), task(4)
    module = module_with_tasks(first, second)
    group = allocated_group(first, second)
    module._handle_write_request(copy.copy(first))
    if failure == "abort":
        module._abort(1)
    else:
        first.create_time = second.create_time = 0
        module._check_tasks_time_out()
    assert module.failed_queue.qsize() == 2
    module._apply_kv_allocation(group)
    assert module.waiting_dict == {}
    assert module.request_page_task_queue.empty()
    assert module._handle_write_request(copy.copy(first)) is None


@pytest.mark.parametrize("backend_class", [PDDecodeNode, PDDPForDecodeNode])
@pytest.mark.parametrize("blocked", ["capacity", "growth", "abort", "failed", "not_ready"])
def test_infer_waits_without_allocation_when_blocked(monkeypatch, backend_class, blocked):
    backend = backend_class.__new__(backend_class)
    backend.args = SimpleNamespace(page_size=4, mtp_step=0)
    backend.dp_world_size = 1
    backend._alloc_req_kv_mem = MagicMock()
    req = SimpleNamespace(
        pd_pending_kv_group=PDChunckedTransTaskGroup([task()]),
        pd_first_chunk_ready=blocked != "not_ready",
        infer_aborted=blocked == "abort",
        pd_task_failed_num=int(blocked == "failed"),
        _kv_cache_alloc_need=lambda n: 8,
        cur_output_len=0,
        cur_kv_len=0,
        hold_kv_len=0,
        shm_req=SimpleNamespace(input_len=8),
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(max_new_tokens=16, ignore_eos=True)),
    )
    monkeypatch.setattr(decode_impl.g_infer_context, "requests_mapping", {1: req})
    monkeypatch.setattr(
        decode_impl.g_infer_context, "get_can_alloc_token_num", lambda: 0 if blocked == "capacity" else 8
    )
    backend._allocate_ready_kv([1])
    backend._alloc_req_kv_mem.assert_not_called()
    assert req.pd_pending_kv_group is not None


def test_tp_uses_minimum_capacity_before_allocating(monkeypatch):
    backend = PDDecodeNode.__new__(PDDecodeNode)
    backend.args = SimpleNamespace(page_size=4, mtp_step=0)
    backend.dp_world_size = 2
    backend._alloc_req_kv_mem = MagicMock()
    req = SimpleNamespace(
        pd_pending_kv_group=PDChunckedTransTaskGroup([task()]),
        pd_first_chunk_ready=True,
        infer_aborted=False,
        pd_task_failed_num=0,
        _kv_cache_alloc_need=lambda n: 8,
        cur_output_len=0,
        cur_kv_len=0,
        hold_kv_len=0,
        shm_req=SimpleNamespace(input_len=8),
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(max_new_tokens=1, ignore_eos=True)),
    )
    monkeypatch.setattr(decode_impl.g_infer_context, "requests_mapping", {1: req})
    monkeypatch.setattr(decode_impl.g_infer_context, "get_can_alloc_token_num", lambda: 1000)
    tensor = torch.tensor
    monkeypatch.setattr(decode_impl.torch, "tensor", lambda values, **kw: tensor(values, dtype=kw["dtype"]))
    from lightllm.distributed import dist_group_manager

    monkeypatch.setattr(dist_group_manager, "get_default_group", lambda: SimpleNamespace(device_group=object()))

    def reduce_min(budget, op, group):
        assert op == torch.distributed.ReduceOp.MIN
        budget.fill_(0)

    monkeypatch.setattr(decode_impl.dist, "all_reduce", reduce_min)
    backend._allocate_ready_kv([1])
    backend._alloc_req_kv_mem.assert_not_called()


def test_router_bounds_pending_requests_but_does_not_reserve_their_prompts():
    router_queue = PDDecodeQueue.__new__(PDDecodeQueue)
    router_queue.max_total_tokens = 100
    router_queue.running_max_req_size = 2
    router_queue.dp_index = 0
    router_queue.is_busy = lambda: False
    router_queue.router = SimpleNamespace(
        router_statics=SimpleNamespace(ema_req_out_len=1), shared_token_load=MagicMock()
    )
    req = SimpleNamespace(get_tuple_tokens=lambda *a: (60, 20))
    assert router_queue._can_add_new_req(req, 80, 1) == (True, 80, 2)
    assert req.pd_waiting_for_kv
    assert router_queue._can_add_new_req(req, 80, 2) == (False, None, None)
    req.get_tuple_tokens = lambda *a: (100, 20)
    assert router_queue._can_add_new_req(req, 0, 0) == (False, None, None)


@pytest.mark.parametrize("hybrid", [False, True])
def test_partial_prefix_is_released_before_waiting_and_full_hit_is_reused(monkeypatch, hybrid):
    backend = PDDecodeNode.__new__(PDDecodeNode)
    backend.args = SimpleNamespace(pd_kv_page_size=4, page_size=4)
    backend.is_master_in_dp = True
    backend.node_world_size = 1
    backend.dp_rank_in_node = 0
    backend.info_queue = queue.Queue()
    backend.model = SimpleNamespace(mem_manager=MagicMock())
    backend._alloc_req_kv_mem = MagicMock()
    monkeypatch.setattr(decode_impl.g_infer_context, "is_hybrid_att_model", hybrid)
    synchronize = MagicMock()
    monkeypatch.setattr(decode_impl.torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=synchronize))
    released = []

    def release(free_indexes, req):
        released.append(req.req_id)
        req.shared_kv_node = None

    monkeypatch.setattr(decode_impl.g_infer_context, "_free_req_mem_without_radix_insert", release)

    def request(req_id, prefix):
        return SimpleNamespace(
            req_id=req_id,
            req_idx=0,
            cur_kv_len=prefix,
            hold_kv_len=prefix,
            pd_trans_device_id=-1,
            pd_task_num=0,
            shared_kv_node=object(),
            shm_req=SimpleNamespace(input_len=8, shm_cur_kv_len=prefix, prompt_cache_len=prefix),
            sampling_param=SimpleNamespace(pd_master_node_id=0),
        )

    partial = request(1, 4)
    backend._decode_node_gen_trans_tasks(partial)
    assert released == [1]
    assert synchronize.call_count == int(hybrid)
    assert partial.shared_kv_node is None
    assert partial.cur_kv_len == partial.hold_kv_len == 0
    group = backend.info_queue.get_nowait()
    assert group.task_list[0].start_kv_index == 0
    full = request(2, 7)
    backend._decode_node_gen_trans_tasks(full)
    assert released == [1]
    assert full.cur_kv_len == 7
    assert full.pd_pending_kv_group is None
    assert not full.shm_req.pd_waiting_for_kv
    group = backend.info_queue.get_nowait()
    assert len(group.task_list) == 1
    assert not group.task_list[0].need_transfer_page()
    backend._alloc_req_kv_mem.assert_not_called()


def test_allocation_control_does_not_complete_a_transfer(monkeypatch):
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend

    req = SimpleNamespace(pd_task_success_num=0, pd_task_failed_num=0)
    ret = task().createRetObj()
    ret.allocation_requested = True
    monkeypatch.setattr(decode_impl.g_infer_context, "requests_mapping", {1: req})
    backend = SimpleNamespace(shm_pd_trans_io_buffer=MagicMock())
    backend.shm_pd_trans_io_buffer.read_obj.return_value = [ret]
    ModeBackend._read_pd_trans_io_buffer_and_update_req_status(backend)
    assert req.pd_first_chunk_ready
    assert req.pd_task_success_num == req.pd_task_failed_num == 0


def test_prefill_notifies_all_ready_chunks_even_with_an_exhausted_source_ring():
    from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_trans_process import (
        _PrefillTransModule,
    )

    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.waiting_dict_lock = threading.Lock()
    module.waiting_dict = {}
    module.page_index_queue = queue.Queue()
    module.failed_queue = queue.Queue()
    module.transporter = MagicMock()
    tasks = [task(i * 4) for i in range(8)]
    for t in tasks:
        module._request_write(t)
    assert module.transporter.send_write_request_task_to_decode_node.call_count == 8
    assert all(t.src_page_index is None for t in tasks)
    assert len(module.waiting_dict) == 8
    module._abort(1)
    assert module.failed_queue.qsize() == 8
    assert module.page_index_queue.empty()


def test_prefill_control_send_error_does_not_allocate_a_source_page():
    from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_trans_process import (
        _PrefillTransModule,
    )

    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.waiting_dict_lock = threading.Lock()
    module.waiting_dict = {}
    module.page_index_queue = queue.Queue()
    module.failed_queue = queue.Queue()
    module.transporter = MagicMock()
    module.transporter.send_write_request_task_to_decode_node.side_effect = RuntimeError("peer unavailable")
    t = task()
    module._request_write(t)
    assert module.failed_queue.get_nowait() is t
    assert t.src_page_index is None
    assert module.waiting_dict == {}
    assert "peer unavailable" in t.error_info


def test_admitted_unaligned_prompt_allocates_on_idle_decode(monkeypatch):
    from lightllm.server.core.objs import req as req_module
    from lightllm.server.router.model_infer.infer_batch import InferReq

    args = SimpleNamespace(page_size=16, mtp_step=0)
    params = SimpleNamespace(max_new_tokens=1, ignore_eos=True)
    req = SimpleNamespace(
        args=args,
        req_idx=0,
        input_len=65,
        shm_cur_output_len=0,
        shm_cur_kv_len=0,
        sample_params=params,
        shm_req=SimpleNamespace(input_len=65, pd_waiting_for_kv=True),
        sampling_param=SimpleNamespace(shm_param=params),
        hold_kv_len=0,
        cur_kv_len=0,
        cur_output_len=0,
        pd_pending_kv_group=PDChunckedTransTaskGroup([task()]),
        pd_first_chunk_ready=True,
        infer_aborted=False,
        pd_task_failed_num=0,
    )
    monkeypatch.setattr(req_module, "get_env_start_args", lambda: args)
    req.get_tuple_tokens = lambda *a: req_module.ChunkedPrefillReq.get_tuple_tokens(req, *a)
    req._kv_cache_alloc_need = lambda n: InferReq._kv_cache_alloc_need(req, n)
    router = PDDecodeQueue.__new__(PDDecodeQueue)
    router.max_total_tokens, router.running_max_req_size, router.dp_index = 112, 1, 0
    router.is_busy = lambda: False
    router.router = SimpleNamespace(router_statics=SimpleNamespace(ema_req_out_len=64), shared_token_load=MagicMock())
    assert router._can_add_new_req(req, 0, 0)[0]

    backend = PDDecodeNode.__new__(PDDecodeNode)
    backend.args, backend.dp_world_size, backend.is_master_in_dp = args, 1, True
    backend.info_queue = queue.Queue()
    backend.model = SimpleNamespace(req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(80).reshape(1, 80)))
    backend._alloc_req_kv_mem = MagicMock(side_effect=lambda r, n: setattr(r, "hold_kv_len", r.hold_kv_len + n))
    monkeypatch.setattr(decode_impl.g_infer_context, "requests_mapping", {1: req})
    monkeypatch.setattr(decode_impl.g_infer_context, "get_can_alloc_token_num", lambda: 112)
    backend._allocate_ready_kv([1])
    backend._alloc_req_kv_mem.assert_called_once_with(req, 80)
    assert req.pd_pending_kv_group is None
    assert not req.shm_req.pd_waiting_for_kv
    assert backend.info_queue.get_nowait().allocation_update


@pytest.mark.parametrize("completion", ["done", "error"])
def test_decode_timeout_retains_granted_page_until_sender_finishes(monkeypatch, completion):
    import pickle
    from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_trans_process

    t = task()
    t.dst_page_index, t.start_trans_time = 0, 0
    remote = copy.copy(t)
    remote.write_stage = completion
    remote.error_info = "sender cancelled" if completion == "error" else None
    module = module_with_tasks(t)
    module.args = SimpleNamespace(pd_trans_mode="nixl")
    module.device_id = 0
    module.transporter = MagicMock()
    module.ready_page_task_queue = queue.Queue()
    monkeypatch.setattr(decode_trans_process.time, "time", lambda: 1000)
    monkeypatch.setattr(decode_trans_process.torch.cuda, "set_device", lambda _: None)
    module._check_tasks_time_out()
    module._check_tasks_time_out()
    assert module.waiting_dict[t.get_key()] is t
    assert module.failed_queue.empty()
    module.transporter.send_error_info_to_prefill_node.assert_called_once_with(t)

    module.transporter.get_new_notifs.return_value = {"peer": [pickle.dumps(remote)]}
    module._check_tasks_time_out = MagicMock(side_effect=RuntimeError("stop loop"))
    with pytest.raises(RuntimeError, match="stop loop"):
        module.accept_peer_task_loop()
    assert module.failed_queue.get_nowait() is t
    assert module.failed_queue.empty()
    assert not module.waiting_dict
    assert module.ready_page_task_queue.empty()


def test_decode_stuck_write_fails_worker_without_recycling_page(monkeypatch):
    from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_trans_process

    t = task()
    t.dst_page_index, t.start_trans_time, t._cancel_time = 0, 0, 1
    module = module_with_tasks(t)
    monkeypatch.setattr(decode_trans_process.time, "time", lambda: 1000)
    with pytest.raises(RuntimeError, match="did not drain"):
        module._check_tasks_time_out()
    assert module.waiting_dict[t.get_key()] is t
    assert module.failed_queue.empty()


def test_ready_send_failure_cannot_recycle_a_possibly_granted_page(monkeypatch):
    from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_trans_process

    t = task()
    module = module_with_tasks()
    module.device_id = 0
    module.page_index_queue = queue.Queue()
    module.page_index_queue.put(0)
    module.request_page_task_queue.put(t)
    module.transporter = MagicMock()
    module.transporter.send_write_ready_task_to_prefill_node.side_effect = ConnectionError("ambiguous send")
    monkeypatch.setattr(decode_trans_process.torch.cuda, "set_device", lambda _: None)
    with pytest.raises(RuntimeError, match="Cannot safely reclaim"):
        module.request_page_loop()
    assert module.waiting_dict[t.get_key()] is t
    assert module.page_index_queue.empty()
    assert module.failed_queue.empty()


def test_nccl_sender_error_cannot_recycle_posted_receive(monkeypatch):
    import pickle
    from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_trans_process

    t = task()
    t.dst_page_index = 0
    remote = copy.copy(t)
    remote.error_info = "sender cancelled"
    module = module_with_tasks(t)
    module.device_id = 0
    module.args = SimpleNamespace(pd_trans_mode="nccl")
    module.transporter = MagicMock()
    module.transporter.get_new_notifs.return_value = {"peer": [pickle.dumps(remote)]}
    monkeypatch.setattr(decode_trans_process.torch.cuda, "set_device", lambda _: None)
    with pytest.raises(RuntimeError, match="NCCL receive failed"):
        module.accept_peer_task_loop()
    assert module.failed_queue.empty()


def test_request_send_failure_after_ready_cannot_return_source_ownership():
    from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_trans_process import (
        _PrefillTransModule,
    )

    t = task()
    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.waiting_dict_lock = threading.Lock()
    module.waiting_dict = {}
    module.failed_queue = queue.Queue()
    module.transporter = MagicMock()

    def receive_ready_then_fail(_):
        module.waiting_dict.pop(t.get_key())
        t.dst_page_index = 0
        raise ConnectionError("send completed before exception")

    module.transporter.send_write_request_task_to_decode_node.side_effect = receive_ready_then_fail
    with pytest.raises(RuntimeError, match="source copy may be active"):
        module._request_write(t)
    assert module.failed_queue.empty()
