import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector.cache_aware import (
    BalanceRelThresholdController,
    CacheAwareConfig,
    CacheAwarePolicy,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_selector import (
    LoadBalancedCacheAwareSelector,
    PDSelectionExtraInfo,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree import PromptCacheTree


def test_prompt_cache_tree_long_prompt_insert_match_and_reinsert():
    tree = PromptCacheTree()
    # The sampled key must exceed the actual interpreter recursion limit.
    prompt = "x" * ((sys.getrecursionlimit() + 512) * tree.sample_stride + 1)
    tree.insert(prompt, "P0")
    result = tree.prefix_match(prompt)
    assert result.prefill_node == "P0"
    assert result.matched_char_count == len(prompt)

    tree.insert(prompt, "P1")
    result = tree.prefix_match(prompt + "different suffix")
    assert result.prefill_node == "P1"
    assert result.matched_char_count == len(prompt)
    assert len(tree._leaf_lru) == 1


def test_prompt_cache_tree_long_prompt_eviction_and_branch_recency():
    tree = PromptCacheTree(sample_stride=1, max_node_count=8, evict_node_batch=1)
    tree.insert("abc-old", "P0")
    tree.insert("abc-new", "P1")
    assert tree.prefix_match("abc-new").prefill_node == "P1"
    # Eviction must still run after a key longer than the recursion limit.
    tree.insert("z" * (sys.getrecursionlimit() + 512), "P2")
    assert tree._node_count <= tree.max_node_count
    assert tree.prefix_match("abc-old").matched_char_count == 0
    assert tree.prefix_match("abc-new").matched_char_count == 0
    assert len(tree._leaf_lru) <= 1


def _worker(address: str, dispatched_prompt_chars: int = 0, dispatched_req_num: int = 0):
    return SimpleNamespace(
        client_ip_port=address,
        connection_key=address,
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


def test_cache_aware_selector_returns_selected_nodes_and_estimated_cache_hit_rate():
    selector = LoadBalancedCacheAwareSelector(pd_manager=None)
    p_node = _worker("10.0.0.1:8000")
    d_node = SimpleNamespace(
        client_ip_port="10.0.0.2:8000",
        run_status=SimpleNamespace(total_token_usage_rate=0.0),
    )
    selector.update_nodes([p_node], [d_node])
    prompt = "x" * 1025
    with patch(
        "lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree.time.monotonic",
        return_value=123.0,
    ):
        selector.policy.prompt_cache_tree.insert(prompt, p_node.client_ip_port)

    selected_p_node, selected_d_node, selection_extra_info = selector.select_p_d_node(prompt, None, None)

    assert selected_p_node is p_node
    assert selected_d_node is d_node
    assert selection_extra_info == PDSelectionExtraInfo(
        estimated_cache_hit_rate=pytest.approx(1.0),
        cache_last_insert_time=123.0,
    )


def test_cache_aware_reinsert_refreshes_matched_node_time():
    policy = CacheAwarePolicy()
    worker = _worker("10.0.0.1:8000")
    prompt = "shared prefix " * 100

    with patch(
        "lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree.time.monotonic",
        return_value=100.0,
    ):
        policy.insert_prompt_cache(prompt, worker)
        first_result = policy.prompt_cache_tree.prefix_match(prompt)
    with patch(
        "lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree.time.monotonic",
        return_value=160.0,
    ):
        policy.insert_prompt_cache(prompt, worker)
        second_result = policy.prompt_cache_tree.prefix_match(prompt)

    assert first_result.last_insert_time == 100.0
    assert second_result.last_insert_time == 160.0


def test_cache_aware_inserts_prompt_only_when_explicitly_recorded():
    policy = CacheAwarePolicy()
    worker = _worker("10.0.0.1:8000")
    prompt = "new prompt"

    assert policy.select_worker([worker], prompt) is worker
    assert policy.prompt_cache_tree.prefix_match(prompt).prefill_node is None

    policy.insert_prompt_cache(prompt, worker)

    assert policy.prompt_cache_tree.prefix_match(prompt).prefill_node == worker.client_ip_port


def test_cache_aware_keeps_cache_worker_when_inflight_load_is_balanced():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=110, dispatched_req_num=2)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100, dispatched_req_num=2)
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([cache_worker, least_loaded_worker], prompt)
    cache_info = policy.get_estimated_cache_info(selected_worker, prompt)

    assert selected_worker is cache_worker
    # 前缀树每 512 个字符抽样一次，命中率保持原有的保守估算方式。
    assert cache_info.estimated_cache_hit_rate == pytest.approx(1025 / len(prompt))


def test_cache_aware_uses_least_loaded_worker_when_cache_worker_is_overloaded():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=2000, dispatched_req_num=2)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100, dispatched_req_num=2)
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([cache_worker, least_loaded_worker], prompt)
    cache_info = policy.get_estimated_cache_info(selected_worker, prompt)

    assert selected_worker is least_loaded_worker
    assert cache_info.estimated_cache_hit_rate == 0.0


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
