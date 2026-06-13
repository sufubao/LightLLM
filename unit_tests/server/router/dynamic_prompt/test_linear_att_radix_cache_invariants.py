"""Property-based invariant fuzzer for LinearAttPagedRadixCache (small-page regime).

This drives the cache the way infer_batch.py does in the default serving regime
(big-page matching disabled, i.e. big_page_num huge so every request inserts only
small pages with a single tail state buffer). After every random operation it
asserts the full set of internal invariants and verifies value integrity against
an independent first-write-wins oracle.

Run: pytest unit_tests/server/router/dynamic_prompt/test_linear_att_radix_cache_invariants.py -q
"""
import uuid

import numpy as np
import pytest
import torch

from lightllm.server.router.dynamic_prompt.linear_att_radix_cache import LinearAttPagedRadixCache

PAGE = 4
BIG = 10_000_000  # big pages effectively disabled (serving default regime)


class FakeSmallPageBuffers:
    """Models LinearAttCacheManager's id pool with strict double-free detection."""

    def __init__(self, size):
        self.size = size
        self.free_set = set(range(size))
        self.free_order = list(range(size))

    def alloc_one_state_cache(self):
        if not self.free_order:
            return None
        idx = self.free_order.pop(0)
        self.free_set.discard(idx)
        return idx

    def free_state_cache(self, free_indexes):
        for idx in free_indexes:
            assert idx is not None
            assert idx not in self.free_set, f"double free of small-page buffer {idx}"
            self.free_set.add(idx)
            self.free_order.append(idx)

    def get_free_cache_num(self):
        return len(self.free_order)


class FakeBigPageBuffers:
    def __init__(self):
        self.freed = []

    def free_state_cache(self, free_indexes):
        self.freed.extend(free_indexes)


class FakeAllocator:
    def __init__(self, size):
        self.size = size
        self.can_use_mem_size = size


class FakeMemManager:
    def __init__(self, size):
        self.allocator = FakeAllocator(size)
        self.linear_att_big_page_buffers = FakeBigPageBuffers()
        self.freed_mem = []

    def free(self, mem_index):
        self.freed_mem.append(mem_index)
        self.allocator.can_use_mem_size += len(mem_index)


def build(small_pool_size=32, mem_size=100_000):
    small = FakeSmallPageBuffers(small_pool_size)
    mm = FakeMemManager(mem_size)
    cache = LinearAttPagedRadixCache(
        unique_name=f"fuzz_{uuid.uuid4().hex[:8]}",
        total_token_num=mem_size,
        rank_in_node=0,
        hash_page_size=PAGE,
        big_page_num=BIG,
        kv_cache_mem_manager=mm,
        linear_att_small_page_buffers=small,
    )
    return cache, small, mm


# ------------------------- tree walking helpers -------------------------


def walk(cache):
    """Return all non-root nodes via BFS."""
    out = []
    stack = list(cache.root_node.children.values())
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children.values())
    return out


def check_invariants(cache, small: FakeSmallPageBuffers, allocated_ids: set):
    nodes = walk(cache)

    # 1. structural: prefix len, child-key consistency, page bookkeeping
    for n in nodes:
        assert n.parent is not None
        assert n.node_value_len == len(n.token_mem_index_value) == len(n.token_id_key)
        assert n.node_prefix_total_len == n.parent.node_prefix_total_len + n.node_value_len
        assert n.ref_counter >= 0, f"negative ref_counter {n.ref_counter}"
        for k, c in n.children.items():
            assert c.page_hash == k
            assert c.parent is n
        # small-page regime: every node is exactly one page
        assert n.node_value_len == PAGE
        assert n.page_num == 1
        assert not n.is_big_page_node() or n.node_prefix_total_len == 0

    # 2. accounting: tree_total and refed token counts match the live tree
    total = sum(n.node_value_len for n in nodes)
    refed = sum(n.node_value_len for n in nodes if n.ref_counter > 0)
    assert (
        cache.get_tree_total_tokens_num() == total
    ), f"tree_total {cache.get_tree_total_tokens_num()} != actual {total}"
    assert cache.get_refed_tokens_num() == refed, f"refed {cache.get_refed_tokens_num()} != actual {refed}"

    # 3. evict-set membership: exactly the leaves, root excluded
    leaves = {id(n) for n in nodes if n.is_leaf()}
    evict_ids = {id(n) for n in cache._evict_tree_set}
    assert evict_ids == leaves, "evict set must equal the set of non-root leaves"
    assert id(cache.root_node) not in evict_ids

    # 4. buffer-eviction set membership: exactly nodes holding a small-page buffer
    with_buf = {id(n) for n in nodes if n.small_page_buffer_idx is not None}
    buf_evict_ids = {id(n) for n in cache._evict_tree_set_for_linear_att}
    assert buf_evict_ids == with_buf, "linear-att evict set must equal nodes with a buffer"

    # 5. buffer-id conservation: ids in tree, ids free in pool, partition the universe
    in_tree = [n.small_page_buffer_idx for n in nodes if n.small_page_buffer_idx is not None]
    assert len(in_tree) == len(set(in_tree)), "a small-page buffer id is used by two nodes"
    in_tree_set = set(in_tree)
    # every allocated id is either in the tree or free in the pool, never both, never lost
    assert in_tree_set.isdisjoint(small.free_set), "buffer id is both in tree and free"
    assert in_tree_set | small.free_set == allocated_ids | set(
        range(small.size)
    ), "buffer id leaked (neither in tree nor free)"


# ------------------------- oracle for value integrity -------------------------


def page_tokens(page_id):
    # distinct, deterministic token block per logical page id
    return list(range(page_id * PAGE, page_id * PAGE + PAGE))


def hashes_for(page_ids):
    # chained hash so prefixes are prefix-closed (same as real block hashing)
    from lightllm.utils.kv_cache_utils import compute_token_list_hash

    toks = []
    for p in page_ids:
        toks.extend(page_tokens(p))
    toks.append(-1)  # one extra so (len-1)//PAGE == len(page_ids)
    return compute_token_list_hash(toks, PAGE)


class Oracle:
    """First-write-wins record of the value stored for each hashed page path."""

    def __init__(self):
        self.value_by_hash = {}  # block_hash -> mem value tensor (PAGE long)

    def record(self, hashs, values):
        for i, h in enumerate(hashs):
            if h not in self.value_by_hash:
                self.value_by_hash[h] = values[i * PAGE : (i + 1) * PAGE].clone()

    def forget(self, freed_hashs):
        for h in freed_hashs:
            self.value_by_hash.pop(h, None)

    def expected_mem(self, hashs):
        return torch.cat([self.value_by_hash[h] for h in hashs])


# ------------------------- the fuzzer -------------------------


@pytest.mark.parametrize("seed", list(range(16)))
@pytest.mark.parametrize("pool", [6, 24])  # 6 => constant state-buffer pressure & real steals
def test_invariant_fuzz(seed, pool):
    rng = np.random.default_rng(seed * 100 + pool)
    cache, small, mm = build(small_pool_size=pool, mem_size=200_000)
    oracle = Oracle()

    next_mem = [1000]

    def alloc_mem(n):
        base = next_mem[0]
        next_mem[0] += n
        return torch.arange(base, base + n, dtype=torch.int64)

    allocated_ids = set()
    live = []  # list of (ans_node, matched_hashs) holding a ref to dec later

    def do_insert():
        npages = int(rng.integers(1, 7))
        page_ids = [int(rng.integers(0, 40)) for _ in range(npages)]
        hashs = hashes_for(page_ids)
        key = torch.tensor([t for p in page_ids for t in page_tokens(p)], dtype=torch.int64)
        value = alloc_mem(npages * PAGE)
        buf = small.alloc_one_state_cache()
        if buf is None:
            # Contract (see _linear_att_free_req): in the small-page regime the radix
            # insert is only performed when a tail state buffer exists. With the pool
            # exhausted the request simply caches nothing. Skip — do not insert a
            # None-tailed path (the cache rightly asserts against that).
            return
        allocated_ids.add(buf)
        linear_idxs = [None] * npages
        linear_idxs[-1] = buf
        oracle.record(hashs, value)
        before_free = set(small.free_set)
        cache.insert(key, value, block_hashs=hashs, block_linear_idxs=linear_idxs)
        # if our buffer was immediately freed (duplicate tail), drop it from allocated set
        newly_free = small.free_set - before_free
        for idx in newly_free:
            allocated_ids.discard(idx)

    def do_match():
        npages = int(rng.integers(1, 7))
        page_ids = [int(rng.integers(0, 40)) for _ in range(npages)]
        hashs = hashes_for(page_ids)
        key = torch.tensor([t for p in page_ids for t in page_tokens(p)], dtype=torch.int64)
        node, kv_len, mem = cache.match_prefix(key, block_hashs=hashs, update_refs=True)
        if node is None:
            assert kv_len == 0 and mem is None
            return
        assert kv_len == node.node_prefix_total_len == len(mem)
        assert kv_len % PAGE == 0
        matched_pages = kv_len // PAGE
        matched_hashs = hashs[:matched_pages]
        # value integrity: returned mem must equal first-written values for this path
        expected = oracle.expected_mem(matched_hashs)
        assert torch.equal(mem, expected), "match returned wrong mem values"
        # the matched tail must be reusable: big-page or has a state buffer
        assert node.is_big_page_node() or node.small_page_buffer_idx is not None
        live.append((node, matched_hashs))

    def do_dec():
        if not live:
            return
        i = int(rng.integers(0, len(live)))
        node, _ = live.pop(i)
        cache.dec_node_ref_counter(node)

    def do_steal():
        before = small.get_free_cache_num()
        cache.free_one_small_page_linear_att_buffer()
        after = small.get_free_cache_num()
        if after > before:
            # a stolen buffer returned to the pool; drop any of our tracked ids that
            # are no longer in the tree (conservation invariant rechecks the rest)
            in_tree = {n.small_page_buffer_idx for n in walk(cache) if n.small_page_buffer_idx is not None}
            for idx in list(allocated_ids):
                if idx not in in_tree and idx in small.free_set:
                    allocated_ids.discard(idx)

    def do_evict():
        unref = cache.get_tree_total_tokens_num() - cache.get_refed_tokens_num()
        if unref <= 0:
            return
        want_pages = int(rng.integers(1, unref // PAGE + 1)) if unref >= PAGE else 0
        if want_pages == 0:
            return
        need = want_pages * PAGE

        def cb(mem_index, small_buf_id):
            # mirror free_radix_cache_to_get_enough_token: evicted node's state buffer
            # is returned to the pool by the caller's callback.
            if small_buf_id is not None:
                small.free_state_cache([small_buf_id])

        # capture which page hashes leave the tree
        before_nodes = {n.page_hash for n in walk(cache)}
        cache._evict(need, cb)
        after_nodes = {n.page_hash for n in walk(cache)}
        freed_hashs = before_nodes - after_nodes
        oracle.forget(freed_hashs)
        # drop freed buffer ids
        in_tree = {n.small_page_buffer_idx for n in walk(cache) if n.small_page_buffer_idx is not None}
        for idx in list(allocated_ids):
            if idx not in in_tree and idx in small.free_set:
                allocated_ids.discard(idx)

    ops = [do_insert, do_insert, do_match, do_match, do_dec, do_steal, do_steal, do_evict]
    for step in range(600):
        op = ops[int(rng.integers(0, len(ops)))]
        op()
        check_invariants(cache, small, allocated_ids)
        # root must hold exactly baseline(1) + one ref per still-held match; it must NOT
        # drift on misses / trim-to-empty (regression guard for the root-ref leak).
        assert cache.root_node.ref_counter == 1 + len(
            live
        ), f"root ref drifted: {cache.root_node.ref_counter} != 1 + {len(live)}"

    # drain all references, then evict everything; tree must end empty and balanced
    while live:
        node, _ = live.pop()
        cache.dec_node_ref_counter(node)
    check_invariants(cache, small, allocated_ids)
    assert cache.get_refed_tokens_num() == 0
    total = cache.get_tree_total_tokens_num()
    if total > 0:
        cache._evict(total, lambda m, b: small.free_state_cache([b]) if b is not None else None)
    assert cache.get_tree_total_tokens_num() == 0
    assert len(cache._evict_tree_set) == 0
    assert len(cache._evict_tree_set_for_linear_att) == 0


def test_root_ref_balanced_across_many_matches():
    """Root ref must not drift: many match+dec cycles leave it at its initial value."""
    cache, small, mm = build()
    root_ref0 = cache.root_node.ref_counter
    page_ids = [1, 2, 3]
    hashs = hashes_for(page_ids)
    key = torch.tensor([t for p in page_ids for t in page_tokens(p)], dtype=torch.int64)
    value = torch.arange(1000, 1000 + len(page_ids) * PAGE, dtype=torch.int64)
    buf = small.alloc_one_state_cache()
    linear_idxs = [None, None, buf]
    cache.insert(key, value, block_hashs=hashs, block_linear_idxs=linear_idxs)

    for _ in range(50):
        node, kv_len, mem = cache.match_prefix(key, block_hashs=hashs, update_refs=True)
        assert node is not None
        cache.dec_node_ref_counter(node)
    assert cache.root_node.ref_counter == root_ref0, "root ref_counter drifted"
    assert cache.get_refed_tokens_num() == 0


def test_match_then_trim_to_empty_balances_refs():
    """A match that trims all the way back to nothing must restore refs and refed tokens."""
    cache, small, mm = build(small_pool_size=2)
    # insert a 2-page path whose tail has NO buffer and is not a big page:
    # build it via a longer insert then steal the tail buffer so the tail is unusable.
    page_ids = [5, 6]
    hashs = hashes_for(page_ids)
    key = torch.tensor([t for p in page_ids for t in page_tokens(p)], dtype=torch.int64)
    value = torch.arange(2000, 2000 + 2 * PAGE, dtype=torch.int64)
    buf = small.alloc_one_state_cache()
    cache.insert(key, value, block_hashs=hashs, block_linear_idxs=[None, buf])

    # exhaust the pool so the steal actually fires (it is a no-op while slots are free),
    # then steal the only in-tree buffer -> both pages unusable (no buffer, not big page)
    while small.alloc_one_state_cache() is not None:
        pass
    cache.free_one_small_page_linear_att_buffer()
    refed0 = cache.get_refed_tokens_num()
    root_ref0 = cache.root_node.ref_counter
    node, kv_len, mem = cache.match_prefix(key, block_hashs=hashs, update_refs=True)
    assert node is None and kv_len == 0 and mem is None
    assert cache.get_refed_tokens_num() == refed0, "trim-to-empty leaked refed tokens"
    assert cache.root_node.ref_counter == root_ref0, "trim-to-empty leaked a root ref"
    # all nodes back to ref 0
    for n in walk(cache):
        assert n.ref_counter == 0


def test_deref_to_root_balances_root_ref():
    """deref_to_first_big_page_node returning None (reached root) must release the
    match-time root ref — otherwise the big-page-enabled match path leaks it."""
    cache, small, mm = build()
    page_ids = [1, 2]
    hashs = hashes_for(page_ids)
    key = torch.tensor([t for p in page_ids for t in page_tokens(p)], dtype=torch.int64)
    value = torch.arange(3000, 3000 + 2 * PAGE, dtype=torch.int64)
    buf = small.alloc_one_state_cache()
    cache.insert(key, value, block_hashs=hashs, block_linear_idxs=[None, buf])

    r0 = cache.root_node.ref_counter
    share_node, kv, mem = cache.match_prefix(key, block_hashs=hashs, update_refs=True)
    assert share_node is not None and not share_node.is_big_page_node()
    assert cache.root_node.ref_counter == r0 + 1  # match took one root ref
    # small-page regime: root is the only big-page node, so deref walks to root -> None
    node = cache.deref_to_first_big_page_node(share_node)
    assert node is None
    assert cache.root_node.ref_counter == r0, "deref-to-root leaked a root ref"
    assert cache.get_refed_tokens_num() == 0
    for n in walk(cache):
        assert n.ref_counter == 0


def test_root_ref_not_leaked_on_miss():
    """Repeated complete misses must not drift root.ref_counter (regression)."""
    cache, small, mm = build()
    r0 = cache.root_node.ref_counter
    hashs = hashes_for([7, 8])
    key = torch.tensor([t for p in [7, 8] for t in page_tokens(p)], dtype=torch.int64)
    for _ in range(25):
        node, kv, mem = cache.match_prefix(key, block_hashs=hashs, update_refs=True)
        assert node is None and kv == 0 and mem is None
    assert cache.root_node.ref_counter == r0, "root ref leaked on cache miss"
