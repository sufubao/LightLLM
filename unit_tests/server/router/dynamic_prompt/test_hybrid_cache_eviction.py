from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache


pytestmark = pytest.mark.usefixtures("kv_allocator_environment")


class _Allocator:
    def __init__(self, *, free_count=0):
        self.free_count = free_count
        self.freed = []

    def free(self, slots):
        self.freed.extend(slots)


def _resources(freed_small):
    return SimpleNamespace(
        bounded_att_big_page_allocator=_Allocator(),
        bounded_att_small_page_allocator=SimpleNamespace(
            free_count=0,
            free=lambda slots: freed_small.extend(slots),
        ),
    )


def test_matching_without_state_restore_does_not_refresh_small_state_lru():
    freed_small = []
    tree = HybridAttPagedRadixCache(
        total_token_num=64,
        rank_in_node=108,
        hash_page_size=4,
        big_page_num=2,
        att_state_manager=_resources(freed_small),
    )

    tree.insert(
        torch.arange(12),
        torch.arange(100, 112),
        block_hashs=[1, 2, 3],
        block_state_idxs=[None, None, 7],
        len_to_big_page_id={8: 11},
    )
    tree.insert(
        torch.arange(12),
        torch.arange(200, 212),
        block_hashs=[4, 5, 6],
        block_state_idxs=[None, None, 8],
        len_to_big_page_id={8: 12},
    )

    node, matched_len, _ = tree.match_prefix(
        torch.arange(12),
        block_hashs=[1, 2, 3],
        update_refs=True,
        mm_input_spans=[(8, 13)],
    )
    assert matched_len == 8
    tree.dec_node_ref_counter(node)

    tree.free_one_small_page_buffer()

    assert freed_small == [7]
