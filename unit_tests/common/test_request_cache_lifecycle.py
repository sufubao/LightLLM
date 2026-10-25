from types import SimpleNamespace

import pytest
import torch

from lightllm.common import request_cache_manager as cache_module
from lightllm.common.req_manager.base import ReqManager, _ReqLinkedList
from lightllm.common.request_cache_manager import RequestCacheManager
from lightllm.server.router.model_infer.infer_batch import InferenceContext


def test_request_slots_do_not_release_kv():
    requests = ReqManager.__new__(ReqManager)
    requests.req_list = _ReqLinkedList(2)
    slots = [requests.alloc(), requests.alloc()]
    assert not hasattr(requests, "free_token")
    assert not hasattr(requests, "mem_manager")
    requests.free(slots)
    assert requests.req_list.is_all_free()


@pytest.mark.parametrize("pause", [False, True])
def test_kv_is_released_before_slot_reuse(monkeypatch, pause):
    events = []
    requests = ReqManager.__new__(ReqManager)
    requests.req_list = _ReqLinkedList(2)
    requests.req_to_token_indexs = torch.arange(16).reshape(2, 8)
    slots = [requests.alloc(), requests.alloc()]
    manager = RequestCacheManager(SimpleNamespace(page_size=1, diverse_mode=False))
    manager.bind_mem_manager(SimpleNamespace(free=lambda indexes: events.append(("kv", indexes.tolist()))))
    manager.bind_context(requests.req_to_token_indexs, None)
    monkeypatch.setattr(cache_module, "custom_cat", torch.cat)
    release_slots = requests.free

    def free_slots(indexes):
        events.append(("slots", indexes))
        requests.req_to_token_indexs.fill_(999)
        release_slots(indexes)

    monkeypatch.setattr(requests, "free", free_slots)
    reqs = [
        SimpleNamespace(
            req_id=i,
            req_idx=slot,
            cur_kv_len=3,
            hold_kv_len=4,
            shared_kv_node=None,
            cache_tiers=(),
            wait_pause=True,
            paused=False,
            shm_req=SimpleNamespace(shm_cur_kv_len=3),
        )
        for i, slot in enumerate(slots)
    ]
    context = InferenceContext(
        req_manager=requests,
        request_cache_manager=manager,
        requests_mapping={req.req_id: req for req in reqs},
        shm_req_manager=SimpleNamespace(put_back_req_obj=lambda req: events.append(("shm", None))),
    )
    context.args = manager.args
    context.backend = SimpleNamespace(is_master_in_dp=False)
    context.infer_req_ids = [0, 1]
    if pause:
        context.pause_reqs(reqs, is_master_in_dp=False)
        assert [kind for kind, _ in events] == ["kv"]
        assert requests.req_list.can_alloc_size == 0
        assert all(req.paused and not req.wait_pause for req in reqs)
    else:
        context._filter([0, 1], modify_shm_finish_state=False)
        assert [kind for kind, _ in events] == ["kv", "slots", "shm", "shm"]
        assert requests.req_list.is_all_free()
        assert not context.requests_mapping and not context.infer_req_ids
    assert events[0][1] == [0, 1, 2, 3, 8, 9, 10, 11]
    assert all(req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == 0 for req in reqs)


def test_cache_dependencies_are_bound_once():
    manager = RequestCacheManager(SimpleNamespace())
    memory = SimpleNamespace(operator=SimpleNamespace(load_cpu_cache_to_gpu=lambda *args: calls.append(args)))
    manager.bind_mem_manager(memory)
    table = torch.empty((1, 4))
    manager.bind_context(table, None)
    manager.bind_context(table, None)
    with pytest.raises(ValueError, match="different request context"):
        manager.bind_context(table.clone(), None)
    calls = []
    client = object()
    manager.bind_cpu_cache_client(client)
    manager.bind_cpu_cache_client(client)
    with pytest.raises(ValueError, match="different CPU cache client"):
        manager.bind_cpu_cache_client(object())
    req = object()
    indexes = torch.arange(2)
    pages = torch.tensor([3])
    manager.load_cpu_cache_to_gpu(indexes, pages, req)
    assert calls[0][2] is client and calls[0][3] is req
