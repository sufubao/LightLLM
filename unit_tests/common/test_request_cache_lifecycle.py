from types import SimpleNamespace

import pytest
import torch

from lightllm.common import cache_manager as cache_module
from lightllm.common.req_manager.base import ReqManager, _ReqLinkedList
from lightllm.common.cache_manager import CacheManager
from lightllm.server.router.model_infer.infer_batch import InferenceContext


def test_cache_manager_owns_the_index_table():
    cache = CacheManager(SimpleNamespace(page_size=4), 2, 9, device="cpu")
    requests = ReqManager.__new__(ReqManager)
    requests.req_list = _ReqLinkedList(2)
    requests.cache_manager = cache
    requests.HOLD_REQUEST_ID = 2
    requests.init_hold_request_indexs((100, 101, 102, 103))
    assert cache.req_to_token_indexs.shape == (3, 12)
    assert cache.req_to_token_indexs[2].tolist() == [100, 101, 102, 103] * 3
    assert not hasattr(requests, "req_to_token_indexs")
    assert not hasattr(requests, "mem_manager")
    assert not hasattr(requests, "free_token")


@pytest.mark.parametrize("pause", [False, True])
def test_kv_is_released_before_slot_reuse(monkeypatch, pause):
    events = []
    requests = ReqManager.__new__(ReqManager)
    requests.req_list = _ReqLinkedList(2)
    slots = [requests.alloc_slot(), requests.alloc_slot()]
    manager = CacheManager(SimpleNamespace(page_size=1, diverse_mode=False), 2, 8, device="cpu")
    manager.bind_mem_manager(SimpleNamespace(free=lambda indexes: events.append(("kv", indexes.tolist()))))
    manager.req_to_token_indexs[:2].copy_(torch.arange(16).reshape(2, 8))
    requests.cache_manager = manager
    monkeypatch.setattr(cache_module, "custom_cat", torch.cat)
    release_slots = requests._release_slots

    def free_slots(indexes):
        events.append(("slots", indexes))
        requests.cache_manager.req_to_token_indexs.fill_(999)
        release_slots(indexes)

    monkeypatch.setattr(requests, "_release_slots", free_slots)
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
    manager = CacheManager(SimpleNamespace(page_size=1), 1, 4, device="cpu")
    calls = []
    memory = SimpleNamespace(operator=SimpleNamespace(load_cpu_cache_to_gpu=lambda *args: calls.append(args)))
    manager.bind_mem_manager(memory)
    table = manager.req_to_token_indexs
    radix = SimpleNamespace(mem_manager=memory)
    manager.bind_radix_cache(radix)
    manager.bind_radix_cache(radix)
    with pytest.raises(ValueError, match="different radix cache"):
        manager.bind_radix_cache(SimpleNamespace(mem_manager=memory))
    assert manager.req_to_token_indexs is table
    client = object()
    manager.bind_cpu_cache_client(client)
    manager.bind_cpu_cache_client(client)
    with pytest.raises(ValueError, match="different CPU cache client"):
        manager.bind_cpu_cache_client(object())
    req = object()
    manager.load_cpu_cache_to_gpu(torch.arange(2), torch.tensor([3]), req)
    assert calls[0][2] is client and calls[0][3] is req


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("deepseek_mtp", "Deepseek3MTPModel"),
        ("glm5_next_mtp", "Glm5NextMTPModel"),
        ("qwen3_5_mtp", "Qwen3_5MTPModel"),
        ("mistral_mtp", "MistralMTPModel"),
        ("glm4_moe_lite_mtp", "Glm4MoeLiteMTPModel"),
        ("qwen3_moe_mtp", "Qwen3MOEMTPModel"),
        ("qwen3_eagle", "Qwen3EagleModel"),
        ("qwen3_dflash", "Qwen3DFlashModel"),
    ],
)
def test_draft_shares_request_cache_and_mtp_cursor(monkeypatch, module_name, class_name):
    import importlib
    from lightllm.common.cache_manager import HybridCacheManager
    from lightllm.common.linear_state import LinearStateManager
    from lightllm.common.req_manager import req_sampling_params

    monkeypatch.setattr(req_sampling_params, "ReqSamplingParamsManager", lambda size: object())
    state = LinearStateManager.__new__(LinearStateManager)
    state.mtp_step = 2
    state.mtp_state_index = torch.zeros(3, dtype=torch.int32)
    state.conv_state = SimpleNamespace(buffer=torch.ones(1, 3, 1, 4))
    state.ssm_state = SimpleNamespace(buffer=torch.ones(1, 9, 1, 1))
    cache = HybridCacheManager(SimpleNamespace(page_size=1, mtp_step=2), 2, 8, state, device="cpu")
    requests = ReqManager(2, cache)
    cursor = state.mtp_state_index
    assert cursor.shape == (3,) and not cursor.any()
    assert not hasattr(cache, "req_to_mtp_state_index")
    assert not hasattr(requests, "req_to_mtp_state_index")
    cursor.fill_(2)

    model_class = getattr(importlib.import_module(f"lightllm.models.{module_name}.model"), class_name)
    draft = model_class.__new__(model_class)
    draft.main_model = SimpleNamespace(req_manager=requests, linear_config=object())
    draft._init_req_manager()
    assert draft.req_manager is requests
    assert draft.req_manager.cache_manager.req_to_token_indexs is cache.req_to_token_indexs
    assert draft.req_manager.cache_manager.linear_state_manager.mtp_state_index is cursor
    state.restore(0, torch.zeros(1, 1, 2), torch.zeros(1, 1, 1))
    assert cursor.tolist() == [0, 2, 2]
    cache.reset(1)
    assert cursor.tolist() == [0, 0, 2]
    requests.reset_slots()
    assert state.mtp_state_index is cursor
    assert requests.cache_manager.req_to_token_indexs is cache.req_to_token_indexs
