import asyncio
from types import SimpleNamespace

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector.cache_aware import (
    BalanceRelThresholdController,
    CacheAwareConfig,
    CacheAwarePolicy,
)


def _worker(address: str, dispatched_prompt_chars: int = 0, dispatched_req_num: int = 0):
    return SimpleNamespace(
        client_ip_port=address,
        dispatched_prompt_chars=dispatched_prompt_chars,
        dispatched_req_num=dispatched_req_num,
    )


def test_balance_threshold_controller_adjusts_threshold_each_window():
    config = CacheAwareConfig(cache_hit_rate_window_size=3, balance_rel_threshold_step=0.1)
    controller = BalanceRelThresholdController()

    for cache_hit_rate, expected_threshold in (
        (0.5, 1.5),
        (0.6, 1.6),
        (0.4, 1.5),
        (0.4, 1.5),
    ):
        for _ in range(config.cache_hit_rate_window_size):
            controller.append(cache_hit_rate)
        controller.update_config(config)
        assert config.balance_rel_threshold == pytest.approx(expected_threshold)


@pytest.mark.parametrize(
    ("initial_threshold", "first_hit_rate", "second_hit_rate", "expected_threshold"),
    (
        (1.95, 0.5, 0.6, 2.0),
        (1.05, 0.5, 0.4, 1.0),
    ),
)
def test_balance_threshold_controller_limits_threshold_range(
    initial_threshold,
    first_hit_rate,
    second_hit_rate,
    expected_threshold,
):
    config = CacheAwareConfig(
        balance_rel_threshold=initial_threshold,
        cache_hit_rate_window_size=1,
        balance_rel_threshold_step=0.1,
    )
    controller = BalanceRelThresholdController()

    controller.append(first_hit_rate)
    controller.update_config(config)
    controller.append(second_hit_rate)
    controller.update_config(config)

    assert config.balance_rel_threshold == pytest.approx(expected_threshold)


def test_cache_aware_updates_threshold_from_inference_cache_hit_rate():
    policy = CacheAwarePolicy(CacheAwareConfig(cache_hit_rate_window_size=2))

    for _ in range(2):
        policy.record_prompt_cache_hit_rate(0.25)
    assert policy.config.balance_rel_threshold == 1.5
    for _ in range(2):
        policy.record_prompt_cache_hit_rate(0.75)

    assert policy.config.balance_rel_threshold == pytest.approx(1.55)


def test_cache_aware_estimates_hit_rate_only_for_connected_worker():
    policy = CacheAwarePolicy(CacheAwareConfig(sample_stride=1))
    cached_worker = _worker("10.0.0.1:8000")
    prompt = "shared conversation history and a new user turn"
    policy.prompt_cache_tree.insert(prompt[:-10], cached_worker.client_ip_port)

    expected_hit_rate = len(prompt[:-10]) / len(prompt)
    assert policy.estimate_cache_hit_rate([cached_worker], prompt) == pytest.approx(expected_hit_rate)
    assert policy.estimate_cache_hit_rate([_worker("10.0.0.2:8000")], prompt) == 0.0


def test_cache_aware_reuses_admission_match_during_worker_selection(monkeypatch):
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=110, dispatched_req_num=2)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100, dispatched_req_num=2)
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    prefix_match = policy.prompt_cache_tree.prefix_match
    match_call_count = 0

    def counting_prefix_match(text):
        nonlocal match_call_count
        match_call_count += 1
        return prefix_match(text)

    monkeypatch.setattr(policy.prompt_cache_tree, "prefix_match", counting_prefix_match)

    async def select_worker():
        return policy.select_worker([cache_worker, least_loaded_worker], prompt)

    async def estimate_then_select_in_child_task():
        policy.estimate_cache_hit_rate([cache_worker, least_loaded_worker], prompt)
        return await asyncio.gather(*(asyncio.create_task(select_worker()) for _ in range(2)))

    selected_workers = asyncio.run(estimate_then_select_in_child_task())
    assert selected_workers == [cache_worker, cache_worker]
    assert match_call_count == 1

    policy.select_worker([cache_worker, least_loaded_worker], prompt + " new turn")
    assert match_call_count == 2


def test_cache_aware_keeps_reused_matches_isolated_between_requests():
    policy = CacheAwarePolicy()
    workers = [
        _worker("10.0.0.1:8000", dispatched_req_num=1),
        _worker("10.0.0.2:8000", dispatched_req_num=1),
    ]
    prompts = ["a" * 1024, "b" * 1024]
    for worker, prompt in zip(workers, prompts):
        policy.prompt_cache_tree.insert(prompt, worker.client_ip_port)

    async def estimate_then_select(prompt):
        policy.estimate_cache_hit_rate(workers, prompt)
        await asyncio.sleep(0)
        return policy.select_worker(workers, prompt)

    async def run_concurrent_requests():
        return await asyncio.gather(*(estimate_then_select(prompt) for prompt in prompts))

    assert asyncio.run(run_concurrent_requests()) == workers


def test_cache_aware_keeps_cache_worker_when_inflight_load_is_balanced():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=110, dispatched_req_num=2)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100, dispatched_req_num=2)
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([cache_worker, least_loaded_worker], prompt)

    assert selected_worker is cache_worker


def test_cache_aware_uses_least_loaded_worker_when_cache_worker_is_overloaded():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=2000, dispatched_req_num=2)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100, dispatched_req_num=2)
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([cache_worker, least_loaded_worker], prompt)

    assert selected_worker is least_loaded_worker


def test_cache_aware_keeps_cache_worker_when_both_workers_are_idle():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000")
    other_worker = _worker("10.0.0.2:8000")
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([other_worker, cache_worker], prompt)

    assert selected_worker is cache_worker


def test_cache_aware_forces_idle_worker_over_busy_cache_worker():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=100, dispatched_req_num=1)
    idle_worker = _worker("10.0.0.2:8000")
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([cache_worker, idle_worker], prompt)

    assert selected_worker is idle_worker


def test_cache_aware_matches_cache_only_within_idle_workers():
    policy = CacheAwarePolicy()
    busy_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=100, dispatched_req_num=1)
    cache_idle_worker = _worker("10.0.0.2:8000")
    other_idle_worker = _worker("10.0.0.3:8000")
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_idle_worker.client_ip_port)

    selected_worker = policy.select_worker([other_idle_worker, busy_worker, cache_idle_worker], prompt)

    assert selected_worker is cache_idle_worker
