import uuid
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sortedcontainers import SortedDict

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache
from lightllm.server.router.dynamic_prompt.tool_call_cache import contains_tool_call
from lightllm.utils import shm_utils


@pytest.fixture(autouse=True)
def service_name(monkeypatch):
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: f"tool_cache_{uuid.uuid4().hex}")


def insert(tree, tokens):
    return tree.insert(torch.tensor(tokens, dtype=torch.int64))[1]


def cache():
    tree = RadixCache(40, 0)
    tree.enable_tool_call_retention(10, 0.5)
    tree.tool_call_retention.advance_clock(100)
    return tree


def test_prefers_tool_prefix_then_restores_original_recency():
    tree = cache()
    tool = insert(tree, [1, 2, 3])
    ordinary = insert(tree, [4, 5, 6])
    tree.tool_call_retention.protect(tool)
    assert tree.evict_tree_set[0] is ordinary
    tree.tool_call_retention.advance_clock(110)
    assert tree.evict_tree_set[0] is tool
    assert tree.tool_call_retention.protected_tokens == 0
    tree.evict_tree_set._check()


def test_pressure_can_evict_every_protected_token():
    tree = cache()
    for tokens in ([1, 2, 3], [4, 5, 6]):
        tree.tool_call_retention.protect(insert(tree, tokens))
    evicted = []
    tree.evict(6, lambda values: evicted.extend(values.tolist()))
    assert len(evicted) == 6
    assert tree.get_tree_total_tokens_num() == 0
    assert tree.tool_call_retention.protected_tokens == 0
    assert not tree.tool_call_retention._leases


def test_references_are_stronger_than_retention():
    tree = cache()
    insert(tree, [1, 2, 3])
    active, _, _ = tree.match_prefix(torch.tensor([1, 2, 3]), update_refs=True)
    tool = insert(tree, [4, 5, 6])
    tree.tool_call_retention.protect(tool)
    tree.evict(3, lambda _: None)
    assert tree.get_tree_total_tokens_num() == 3
    assert active.ref_counter == 1
    tree.dec_node_ref_counter(active)


def test_budget_and_oversized_prefix():
    tree = cache()
    first = insert(tree, list(range(10)))
    second = insert(tree, list(range(30, 45)))
    tree.tool_call_retention.protect(first)
    tree.tool_call_retention.protect(second)
    assert not first.cache_protected
    assert second.cache_protected
    assert tree.tool_call_retention.protected_tokens == 15
    oversized = insert(tree, list(range(100, 121)))
    tree.tool_call_retention.protect(oversized)
    assert not oversized.cache_protected
    assert tree.tool_call_retention.protected_tokens <= 20


@pytest.mark.parametrize("split_by_insert", [False, True])
def test_split_preserves_lease_deadline_and_token_accounting(split_by_insert):
    tree = cache()
    tail = insert(tree, [1, 2, 3, 4, 5, 6])
    retention = tree.tool_call_retention
    retention.protect(tail)
    if split_by_insert:
        insert(tree, [1, 2, 3, 8, 9])
    else:
        tree.match_prefix(torch.tensor([1, 2, 3]), update_refs=False)
    assert tail.cache_protected and tail.parent.cache_protected
    assert retention.protected_tokens == 6
    assert len(retention._leases) == 2
    tree.merge_unreferenced_nodes()
    assert retention.protected_tokens == 6
    tree.evict_tree_set._check()
    retention.advance_clock(110)
    tree.merge_unreferenced_nodes()
    assert retention.protected_tokens == 0
    tree.evict_tree_set._check()


def test_shared_prefix_counted_once_and_renewal_is_bounded():
    tree = cache()
    left = insert(tree, [1, 2, 3, 4])
    right = insert(tree, [1, 2, 5, 6])
    retention = tree.tool_call_retention
    retention.protect(left)
    retention.protect(right)
    assert retention.protected_tokens == 6
    for _ in range(100):
        retention.protect(right)
    assert len(retention._expiry) == len(retention._leases) == 3
    retention.advance_clock(109)
    retention.protect(right)
    retention.advance_clock(110)
    assert not left.cache_protected
    assert right.cache_protected and right.parent.cache_protected
    assert retention.protected_tokens == 4


def test_budget_counts_structurally_retained_ancestors():
    tree = cache()
    insert(tree, list(range(10)))
    old_tail = insert(tree, list(range(15)))
    tree.tool_call_retention.protect(old_tail)
    newer = insert(tree, list(range(30, 40)))
    tree.tool_call_retention.protect(newer)
    assert newer.cache_protected
    assert not old_tail.cache_protected
    assert not old_tail.parent.cache_protected
    assert tree.tool_call_retention.protected_tokens == 10


def test_final_answer_revokes_stale_tool_prefix_but_preserves_other_branches():
    tree = cache()
    left = insert(tree, [1, 2, 3, 4])
    right = insert(tree, [1, 2, 5, 6])
    retention = tree.tool_call_retention
    retention.protect(left)
    retention.protect(right)
    final = insert(tree, [1, 2, 3, 4, 7, 8])
    retention.complete(final)
    assert not left.cache_protected
    assert right.cache_protected
    assert right.parent.cache_protected
    assert retention.protected_tokens == 4
    tree.evict_tree_set._check()
    retention.complete(right)
    assert retention.protected_tokens == 0
    tree.evict_tree_set._check()


def test_hybrid_retains_both_kv_and_state_without_pinning():
    freed = []
    buffers = SimpleNamespace(
        free_state_cache=lambda indexes=None, free_indexes=None: freed.extend(indexes or free_indexes),
        get_free_cache_num=lambda: 0,
    )
    tree = HybridAttPagedRadixCache(40, 0, 2, 2, SimpleNamespace(big_page_buffers=buffers), buffers)
    tree.enable_tool_call_retention(10, 0.5)
    _, tool = tree.insert(torch.tensor([1, 2]), block_hashs=[11], block_state_idxs=[10])
    _, ordinary = tree.insert(torch.tensor([3, 4]), block_hashs=[12], block_state_idxs=[20])
    tree.tool_call_retention.protect(tool)
    assert tree._evict_tree_set[0] is ordinary
    tree.free_one_small_page_buffer()
    assert freed == [20]
    # A node with no reusable recurrent state must still be reclaimed first.
    assert tree._evict_tree_set[0] is ordinary
    tree._evict(4, lambda values, state: None)
    assert tree.tool_call_retention.protected_tokens == 0


@pytest.mark.parametrize(
    "tokens, markers, expected",
    [
        ([4, 5, 6], [[5, 6]], True),
        ([4, 5], [[5, 6]], False),
        ([], [[5]], False),
        ([4], [[]], False),
        (np.array([1, 2, 3]), [[2]], True),
    ],
)
def test_generated_marker_detection(tokens, markers, expected):
    assert contains_tool_call(tokens, markers) is expected


def test_compact_markers_round_trip_and_do_not_become_stop_sequences():
    from lightllm.server.core.objs.sampling_params import SamplingParams, ToolCallCacheMarkers
    import ctypes

    assert ctypes.sizeof(ToolCallCacheMarkers) < 1024
    params = SamplingParams()
    params.init(None, tool_call_cache_markers=[[8, 9], [10]], stop_sequences=[[11]])
    copy = SamplingParams()
    copy.init(None, **params.to_origin_dict())
    assert copy.tool_call_cache_markers.to_list() == [[8, 9], [10]]
    assert copy.stop_sequences.to_list() == [[11]]
    copy.init(None)
    assert copy.tool_call_cache_markers.to_list() == []


@pytest.mark.parametrize(
    "parser", ["qwen25", "qwen3_coder", "kimi_k2", "deepseekv3", "deepseekv32", "glm47", "llama3", "mistral"]
)
def test_supported_parser_markers(parser):
    from lightllm.server.function_call_parser import FunctionCallParser

    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text])
    markers = FunctionCallParser.cache_marker_token_ids(parser, tokenizer)
    assert markers
    if parser == "qwen3_coder":
        assert list(map(ord, "<function=")) in markers
    if parser == "kimi_k2":
        assert list(map(ord, "<|tool_call_begin|>")) in markers


@pytest.mark.parametrize(
    "status, enabled, generated, expected",
    [
        (1, True, [8, 9], True),
        (2, True, [8, 9], False),
        (3, True, [8, 9], False),
        (4, True, [8, 9], False),
        (0, True, [8, 9], False),
        (1, False, [8, 9], False),
        (1, True, [7, 9], False),
    ],
)
def test_only_normal_stop_with_generated_tool_signal_receives_lease(status, enabled, generated, expected):
    from lightllm.server.core.objs import FinishStatus
    from lightllm.server.core.objs.sampling_params import ToolCallCacheMarkers
    from lightllm.server.router.model_infer.infer_batch import InferenceContext

    tree = cache()
    node = insert(tree, [1, 2, 3])
    marker = ToolCallCacheMarkers()
    marker.initialize([[8, 9]])
    req = SimpleNamespace(
        finish_status=FinishStatus(status),
        sampling_param=SimpleNamespace(
            shm_param=SimpleNamespace(tool_call_cache_markers=marker), disable_prompt_cache=not enabled
        ),
        shm_req=SimpleNamespace(input_len=2),
        # The history contains a tool marker even when generated output does not.
        get_input_token_ids=lambda: np.array([8, 9] + generated),
    )
    InferenceContext(radix_cache=tree)._retain_tool_call_prefix(req, node)
    assert node.cache_protected is expected


@pytest.mark.parametrize("seed", range(8))
def test_randomized_split_expiry_eviction_and_rank_agreement(seed):
    rng = random.Random(seed)
    ranks = [cache(), cache()]
    for step in range(250):
        tokens = [rng.randrange(4) for _ in range(rng.randrange(1, 15))]
        protect = rng.random() < 0.5
        evict = rng.randrange(8)
        outputs = []
        for tree in ranks:
            retention = tree.tool_call_retention
            retention.advance_clock(100 + step / 4)
            node = insert(tree, tokens)
            if protect:
                retention.protect(node)
            else:
                retention.complete(node)
            shared, _, _ = tree.match_prefix(torch.tensor(tokens[:4]), update_refs=True)
            if shared is not None:
                tree.dec_node_ref_counter(shared)
            tree.merge_unreferenced_nodes()
            values = []
            if tree.get_tree_total_tokens_num() >= evict:
                tree.evict(evict, lambda tensor: values.extend(tensor.tolist()))
            outputs.append(values)
            assert retention.protected_tokens == sum(len(n.token_id_key) for n in retention._leases)
            assert 0 <= retention.protected_tokens <= retention.token_budget
            assert len(retention._expiry) == len(retention._leases)
            for node in retention._leases:
                while node.parent is not None:
                    assert node.cache_protected
                    node = node.parent
            tree.evict_tree_set._check()
        assert outputs[0] == outputs[1]
