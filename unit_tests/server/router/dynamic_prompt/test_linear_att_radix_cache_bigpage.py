"""Big-page-regime coverage + invariant fuzz for LinearAttPagedRadixCache.

Active in production only when --linear_att_page_block_num is set (e.g. the GSM8K
launch scripts use 8). Here big_page_num is small so inserts create big-page nodes
plus an optional small tail, mirroring _linear_att_free_req's two insert calls and
copy_linear_att_state_to_cache_buffer's len_to_big_page_id construction.
"""
import uuid

import numpy as np
import pytest
import torch
from sortedcontainers import SortedDict

from lightllm.server.router.dynamic_prompt.linear_att_radix_cache import LinearAttPagedRadixCache
from lightllm.utils.kv_cache_utils import compute_token_list_hash

PAGE = 4
BIGN = 2
BIG_TOKENS = PAGE * BIGN


class FakePool:
    def __init__(self, size):
        self.size = size
        self.free_set = set(range(size))
        self.order = list(range(size))

    def alloc_one_state_cache(self):
        if not self.order:
            return None
        i = self.order.pop(0)
        self.free_set.discard(i)
        return i

    def free_state_cache(self, free_indexes):
        for i in free_indexes:
            assert i is not None and i not in self.free_set, f"double free {i}"
            self.free_set.add(i)
            self.order.append(i)

    def get_free_cache_num(self):
        return len(self.order)


class FakeAllocator:
    def __init__(self, size):
        self.size = size
        self.can_use_mem_size = size


class FakeMem:
    def __init__(self, size, big_pool):
        self.allocator = FakeAllocator(size)
        self.linear_att_big_page_buffers = big_pool

    def free(self, mem_index):
        self.allocator.can_use_mem_size += len(mem_index)


def build(small_size=32, big_size=64, mem=400_000):
    small = FakePool(small_size)
    big = FakePool(big_size)
    mm = FakeMem(mem, big)
    cache = LinearAttPagedRadixCache(
        unique_name=f"bp_{uuid.uuid4().hex[:8]}",
        total_token_num=mem,
        rank_in_node=0,
        hash_page_size=PAGE,
        big_page_num=BIGN,
        kv_cache_mem_manager=mm,
        linear_att_small_page_buffers=small,
    )
    return cache, small, big, mm


def walk(cache):
    out = []
    st = list(cache.root_node.children.values())
    while st:
        n = st.pop()
        out.append(n)
        st.extend(n.children.values())
    return out


def page_tokens(pid):
    return list(range(pid * PAGE, pid * PAGE + PAGE))


def hashes_for(pids):
    toks = []
    for p in pids:
        toks += page_tokens(p)
    toks.append(-1)
    return compute_token_list_hash(toks, PAGE)


def check(cache, small, big):
    nodes = walk(cache)
    # structural + accounting
    total = 0
    refed = 0
    for n in nodes:
        assert n.parent is not None
        assert n.node_prefix_total_len == n.parent.node_prefix_total_len + n.node_value_len
        assert n.ref_counter >= 0
        assert n.node_value_len == len(n.token_mem_index_value)
        if n.is_big_page_node():
            assert n.page_num == BIGN and n.node_value_len == BIG_TOKENS
            assert n.big_page_buffer_idx is not None
            assert n.small_page_buffer_idx is None
        else:
            assert n.page_num == 1 and n.node_value_len == PAGE
            assert n.big_page_buffer_idx is None
        total += n.node_value_len
        if n.ref_counter > 0:
            refed += n.node_value_len
        for k, c in n.children.items():
            assert c.page_hash == k and c.parent is n
    assert cache.get_tree_total_tokens_num() == total
    assert cache.get_refed_tokens_num() == refed
    # evict set == non-root leaves
    leaves = {id(n) for n in nodes if n.is_leaf()}
    assert {id(n) for n in cache._evict_tree_set} == leaves
    assert id(cache.root_node) not in {id(n) for n in cache._evict_tree_set}
    # buffer-evict set == small-buffer holders
    assert {id(n) for n in cache._evict_tree_set_for_linear_att} == {
        id(n) for n in nodes if n.small_page_buffer_idx is not None
    }
    # big-page id conservation
    big_in_tree = [n.big_page_buffer_idx for n in nodes if n.is_big_page_node()]
    assert len(big_in_tree) == len(set(big_in_tree)), "big-page id reused by two nodes"
    assert set(big_in_tree).isdisjoint(big.free_set)
    assert set(big_in_tree) | big.free_set == set(range(big.size)), "big-page id leaked"
    # small-page id conservation
    small_in_tree = [n.small_page_buffer_idx for n in nodes if n.small_page_buffer_idx is not None]
    assert len(small_in_tree) == len(set(small_in_tree))
    assert set(small_in_tree).isdisjoint(small.free_set)
    assert set(small_in_tree) | small.free_set == set(range(small.size)), "small-page id leaked"


def make_insert(cache, small, big):
    """Mirror _linear_att_free_req: big-page-aligned prefix (+ optional small tail)."""

    def insert(pids, mem_base, with_small_tail):
        L = len(pids)
        num_big = L // BIGN
        # len_to_big_page_id: one fresh big id per big-page boundary along the path
        l2b = SortedDict()
        big_ids_alloced = []
        for j in range(1, num_big + 1):
            bid = big.alloc_one_state_cache()
            if bid is None:
                # big pool exhausted: the real caller would not start this insert; roll back.
                for got in big_ids_alloced:
                    big.free_state_cache([got])
                return
            big_ids_alloced.append(bid)
            l2b[j * BIG_TOKENS] = bid
        hashs = hashes_for(pids)
        key = torch.tensor([t for p in pids for t in page_tokens(p)], dtype=torch.int64)
        value = torch.arange(mem_base, mem_base + L * PAGE, dtype=torch.int64)
        linear_idxs = [None] * L
        tail_buf = None
        if with_small_tail and (L % BIGN != 0):
            tail_buf = small.alloc_one_state_cache()
            if tail_buf is None:
                # contract: cannot insert a None-tailed non-aligned path; drop the tail page
                pids = pids[:-1]
                L = len(pids)
                if L == 0:
                    # nothing to insert; release any big ids we grabbed (none, since num_big recomputed)
                    for bid in big_ids_alloced:
                        big.free_state_cache([bid])
                    return
                hashs = hashes_for(pids)
                key = torch.tensor([t for p in pids for t in page_tokens(p)], dtype=torch.int64)
                value = torch.arange(mem_base, mem_base + L * PAGE, dtype=torch.int64)
                linear_idxs = [None] * L
            else:
                linear_idxs[-1] = tail_buf
        elif L % BIGN != 0:
            # no small tail wanted but path is not big-aligned -> trim to aligned length
            pids = pids[: num_big * BIGN]
            L = len(pids)
            if L == 0:
                for bid in big_ids_alloced:
                    big.free_state_cache([bid])
                return
            hashs = hashes_for(pids)
            key = torch.tensor([t for p in pids for t in page_tokens(p)], dtype=torch.int64)
            value = torch.arange(mem_base, mem_base + L * PAGE, dtype=torch.int64)
            linear_idxs = [None] * L

        before_small = set(small.free_set)
        cache.insert(key, value, block_hashs=hashs, block_linear_idxs=linear_idxs, len_to_big_page_id=l2b)
        # any tail buffer that was a duplicate got freed by the cache; nothing to track
        _ = before_small

    return insert


def test_pure_bigpage_insert_and_match():
    cache, small, big = build()[:3]
    ins = make_insert(cache, small, big)
    # 4 pages -> 2 big pages, no small tail
    ins([1, 2, 3, 4], 1000, with_small_tail=False)
    check(cache, small, big)
    assert cache.get_tree_total_tokens_num() == 4 * PAGE

    hashs = hashes_for([1, 2, 3, 4])
    key = torch.tensor([t for p in [1, 2, 3, 4] for t in page_tokens(p)], dtype=torch.int64)
    node, kv, mem = cache.match_prefix(key, block_hashs=hashs, update_refs=True)
    assert node is not None and node.is_big_page_node()
    assert kv == 16 and len(mem) == 16
    assert torch.equal(mem, torch.arange(1000, 1016, dtype=torch.int64))
    cache.dec_node_ref_counter(node)
    check(cache, small, big)


def test_mixed_insert_match_trims_to_bigpage_when_tail_unusable():
    cache, small, big = build(small_size=1)[:3]
    ins = make_insert(cache, small, big)
    # 5 pages -> 2 big pages (8 tokens *2 =16) + 1 small tail page (4) = 20 tokens
    ins([1, 2, 3, 4, 5], 2000, with_small_tail=True)
    check(cache, small, big)
    assert cache.get_tree_total_tokens_num() == 20

    # exhaust small pool and steal the tail buffer -> tail page unusable
    while small.alloc_one_state_cache() is not None:
        pass
    cache.free_one_small_page_linear_att_buffer()
    check(cache, small, big)

    hashs = hashes_for([1, 2, 3, 4, 5])
    key = torch.tensor([t for p in [1, 2, 3, 4, 5] for t in page_tokens(p)], dtype=torch.int64)
    node, kv, mem = cache.match_prefix(key, block_hashs=hashs, update_refs=True)
    # tail small page has no buffer -> trim back to the last big-page boundary (16)
    assert node is not None and node.is_big_page_node()
    assert kv == 16
    cache.dec_node_ref_counter(node)
    check(cache, small, big)


@pytest.mark.parametrize("seed", list(range(10)))
def test_bigpage_fuzz(seed):
    rng = np.random.default_rng(seed)
    cache, small, big, mm = build(small_size=10, big_size=48, mem=400_000)
    ins = make_insert(cache, small, big)
    live = []
    mem_base = [10_000]

    def do_ins():
        L = int(rng.integers(1, 7))
        pids = [int(rng.integers(0, 25)) for _ in range(L)]
        ins(pids, mem_base[0], with_small_tail=bool(rng.integers(0, 2)))
        mem_base[0] += 100

    def do_match():
        L = int(rng.integers(1, 7))
        pids = [int(rng.integers(0, 25)) for _ in range(L)]
        hashs = hashes_for(pids)
        key = torch.tensor([t for p in pids for t in page_tokens(p)], dtype=torch.int64)
        node, kv, mem = cache.match_prefix(key, block_hashs=hashs, update_refs=True)
        if node is None:
            assert kv == 0 and mem is None
            return
        assert kv == node.node_prefix_total_len == len(mem)
        assert node.is_big_page_node() or node.small_page_buffer_idx is not None
        live.append(node)

    def do_dec():
        if live:
            cache.dec_node_ref_counter(live.pop(int(rng.integers(0, len(live)))))

    def do_steal():
        cache.free_one_small_page_linear_att_buffer()

    def do_evict():
        unref = cache.get_tree_total_tokens_num() - cache.get_refed_tokens_num()
        if unref < PAGE:
            return
        need = int(rng.integers(1, unref // PAGE + 1)) * PAGE
        cache._evict(need, lambda m, b: small.free_state_cache([b]) if b is not None else None)

    ops = [do_ins, do_ins, do_match, do_match, do_dec, do_steal, do_evict]
    for _ in range(400):
        ops[int(rng.integers(0, len(ops)))]()
        check(cache, small, big)
        assert cache.root_node.ref_counter == 1 + len(live), "root ref drifted (big-page regime)"

    while live:
        cache.dec_node_ref_counter(live.pop())
    assert cache.get_refed_tokens_num() == 0
    t = cache.get_tree_total_tokens_num()
    if t:
        cache._evict(t, lambda m, b: small.free_state_cache([b]) if b is not None else None)
    assert cache.get_tree_total_tokens_num() == 0
    check(cache, small, big)
