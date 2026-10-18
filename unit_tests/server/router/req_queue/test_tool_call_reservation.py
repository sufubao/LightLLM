from types import SimpleNamespace

import pytest

from lightllm.server.router.req_queue.chunked_prefill.impl import ChunkedPrefillQueue


def queue(reserved, current=(), enabled=True):
    result = ChunkedPrefillQueue.__new__(ChunkedPrefillQueue)
    result.dp_index = 0
    result.max_total_tokens = 100
    result.running_max_req_size = 10
    result.cache_len_list = list(current)
    result.tool_call_admission_enabled = enabled
    result.router = SimpleNamespace(
        router_statics=SimpleNamespace(ema_req_out_len=0),
        get_tool_call_reserved_tokens=lambda _: reserved,
        shared_token_load=SimpleNamespace(
            set_estimated_peak_token_count=lambda *args: None,
            set_dynamic_max_load=lambda *args: None,
        ),
    )
    return result


def req(tokens):
    return SimpleNamespace(get_tuple_tokens=lambda *args: (tokens, 0))


def test_waiting_cache_capacity_is_not_spent_on_unrelated_active_requests():
    assert queue(0, [(30, 0)])._can_add_new_req(req(40), False)
    assert not queue(40, [(30, 0)])._can_add_new_req(req(40), False)


@pytest.mark.parametrize("reserved", [0, 40, 100])
def test_first_feasible_request_can_borrow_waiting_space(reserved):
    assert queue(reserved)._can_add_new_req(req(80), False)
    assert not queue(reserved)._can_add_new_req(req(100), False)


def test_consumption_or_expiry_immediately_reopens_admission():
    result = queue(40, [(30, 0)])
    assert not result._can_add_new_req(req(40), False)
    result.cache_len_list = [(30, 0)]
    result.router.get_tool_call_reserved_tokens = lambda _: 0
    assert result._can_add_new_req(req(40), False)


def test_disabled_policy_does_not_read_reservation_or_change_capacity():
    result = queue(100, [(30, 0)], enabled=False)
    result.router.get_tool_call_reserved_tokens = lambda _: pytest.fail("disabled reservation read")
    assert result._can_add_new_req(req(40), False)


def test_explicit_high_priority_can_borrow_without_waiting_for_cache_expiry():
    result = queue(40, [(30, 0)])
    urgent = req(40)
    urgent.sample_params = SimpleNamespace(high_priority_request=True)
    assert result._can_add_new_req(urgent, False)


@pytest.mark.parametrize("mode,enabled", [("normal", True), ("prefill", False), ("decode", False)])
def test_reservation_admission_is_disabled_for_pd_roles(monkeypatch, mode, enabled):
    from lightllm.server.router.req_queue import base_queue

    monkeypatch.setattr(base_queue, "get_fixed_kv_len", lambda: 0)
    monkeypatch.setattr(base_queue, "get_running_max_req_size_per_dp", lambda _: 10)
    args = SimpleNamespace(
        max_total_token_num=100, router_token_ratio=0.8, tool_call_cache_retention_seconds=3, run_mode=mode
    )
    result = base_queue.BaseQueue(args, SimpleNamespace(), 0, 1)
    assert result.tool_call_admission_enabled is enabled


def test_router_accounts_for_idle_protected_tokens_once_and_keeps_disabled_accounting():
    from lightllm.server.router.manager import RouterManager

    router = RouterManager.__new__(RouterManager)
    router.args = SimpleNamespace(disable_dynamic_prompt_cache=False)
    router.max_total_token_num = 100
    router.read_only_statics_mem_manager = SimpleNamespace(get_unrefed_token_num=lambda _: 50)
    router.radix_cache_client = SimpleNamespace(
        get_unrefed_tokens_num=lambda _: 30, get_tool_call_reserved_tokens=lambda _: 20
    )
    router.tool_call_admission_enabled = True
    assert router.get_used_tokens(0) == 40
    router.tool_call_admission_enabled = False
    assert router.get_used_tokens(0) == 20
    assert router.get_tool_call_reserved_tokens(0) == 0
