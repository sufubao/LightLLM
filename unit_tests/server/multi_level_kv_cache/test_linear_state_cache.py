from types import SimpleNamespace
from uuid import uuid4

import pytest

from lightllm.common.linear_att_cache_manager.checkpoints import prefix_hashes
from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient, _CpuPageStatus
from lightllm.server.multi_level_kv_cache.linear_state_cache import CpuLinearStateCacheClient, _LinearStateStatus


@pytest.fixture
def clients():
    clients = []
    for cls, status, capacity in [
        (CpuKvCacheClient, _CpuPageStatus, 8),
        (CpuLinearStateCacheClient, _LinearStateStatus, 2),
    ]:
        client = cls.__new__(cls)
        client.cache_name = f"checkpoint_test_{uuid4().hex}"
        client.item_class = status
        client.page_num = capacity
        client.args = SimpleNamespace(cpu_cache_token_page_size=8)
        client._create_cpu_status_list(init_shm_data=True)
        clients.append(client)
    kv, state = clients
    state.kv_client = kv
    yield kv, state
    for client in clients:
        for linked_list in [client.page_items, client.page_hash_dict.link_items]:
            linked_list.shm.unlink()
            for item in linked_list.linked_items:
                item.linked_items = []
            del item
            linked_list.linked_items = []
            linked_list.head = linked_list.tail = None
            linked_list.shm.close()
        client.offload_page_indexes.shm.unlink()
        client.offload_page_indexes.arr = None
        client.offload_page_indexes.shm.close()


def store_kv(kv, tokens, endpoints):
    hashes = prefix_hashes(tokens, endpoints)
    pages, _ = kv.allocate_pages([hashes[n] for n in endpoints], False)
    kv.update_pages_status_to_ready(pages)
    return pages


def test_partial_checkpoint_alias_survives_a_different_physical_page_suffix(clients):
    kv, state = clients
    tokens = list(range(24))
    pages = store_kv(kv, tokens, [8, 16, 24])
    key = prefix_hashes(tokens, [11])[11]
    checkpoint = state.allocate_checkpoint(key, 11, pages[1])
    # Pending state is invisible even though its KV page is already ready.
    assert state.match(tokens, 23) == (-1, [], [])
    assert state.allocate_checkpoint(key, 11, pages[1]) is None
    state.update_pages_status_to_ready([checkpoint])
    other = tokens[:11] + [100] * 20
    match, matched_pages, lengths = state.match(other, 30)
    assert match == checkpoint
    assert matched_pages == pages[:2]
    assert lengths == [8, 11]
    state.deref_one_page(match)
    kv.deref_pages(matched_pages)
    assert kv.page_items.get_item_by_index(pages[1]).ref_count == 1
    other[0] = 200
    assert state.match(other, 30) == (-1, [], [])


def test_state_eviction_releases_only_its_tail_reference(clients):
    kv, state = clients
    tokens = list(range(32))
    pages = store_kv(kv, tokens, [8, 16, 24, 32])
    for length in [9, 17, 25]:
        key = prefix_hashes(tokens, [length])[length]
        slot = state.allocate_checkpoint(key, length, pages[(length - 1) // 8])
        state.update_pages_status_to_ready([slot])
    assert kv.page_items.get_item_by_index(pages[1]).ref_count == 0
    assert kv.page_items.get_item_by_index(pages[2]).ref_count == 1
    assert kv.page_items.get_item_by_index(pages[3]).ref_count == 1
    assert state.match(tokens, 16) == (-1, [], [])
    match, matched_pages, lengths = state.match(tokens, 31)
    assert lengths[-1] == 25
    # A borrowed checkpoint cannot be reallocated.
    state.deref_one_page(match)
    kv.deref_pages(matched_pages)


def test_missing_earlier_kv_page_prevents_state_only_hit(clients):
    kv, state = clients
    tokens = list(range(24))
    pages = store_kv(kv, tokens, [8, 16, 24])
    key = prefix_hashes(tokens, [17])[17]
    slot = state.allocate_checkpoint(key, 17, pages[2])
    state.update_pages_status_to_ready([slot])
    # Remove the index entry to model independent eviction of earlier KV.
    kv.page_hash_dict.remove(prefix_hashes(tokens, [8])[8])
    assert state.match(tokens, 23) == (-1, [], [])
    assert state.page_items.get_item_by_index(slot).ref_count == 0


def test_exact_page_boundary_uses_endpoint_page_once(clients):
    kv, state = clients
    tokens = list(range(24))
    pages = store_kv(kv, tokens, [8, 16, 24])
    slot = state.allocate_checkpoint(prefix_hashes(tokens, [16])[16], 16, pages[1])
    state.update_pages_status_to_ready([slot])
    found, matches, lengths = state.match(tokens, 23)
    assert found == slot
    assert matches == pages[:2]
    assert lengths == [8, 16]
    kv.deref_pages(matches)
    state.deref_one_page(found)


def test_kv_pressure_reclaims_idle_checkpoint_pins(clients):
    kv, state = clients
    tokens = list(range(64))
    pages = store_kv(kv, tokens, list(range(8, 65, 8)))
    # All other KV slots are borrowed; the checkpoint is the only possible victim.
    for page in pages[:-1]:
        item = kv.page_items.get_item_by_index(page)
        kv.query_one_page(item.hash_key)
    key = prefix_hashes(tokens, [61])[61]
    slot = state.allocate_checkpoint(key, 61, pages[-1])
    state.update_pages_status_to_ready([slot])
    assert kv.allocate_pages([123456], False) == ([-1], [False])
    allocated, ready = state.allocate_kv_pages([123456], False)
    assert allocated == [pages[-1]]
    assert ready == [False]
    assert state.page_hash_dict.get(key) is None
    assert state.page_items.get_item_by_index(slot).tail_page == -1
    kv.recycle_pages(allocated)
    kv.deref_pages(pages[:-1])


def test_pressure_does_not_evict_borrowed_or_pending_states(clients):
    kv, state = clients
    tokens = list(range(24))
    pages = store_kv(kv, tokens, [8, 16, 24])
    slot = state.allocate_checkpoint(prefix_hashes(tokens, [11])[11], 11, pages[1])
    assert not state.evict_one_checkpoint()
    state.update_pages_status_to_ready([slot])
    found, matches, _ = state.match(tokens, 23)
    assert not state.evict_one_checkpoint()
    state.deref_one_page(found)
    kv.deref_pages(matches)
    assert state.evict_one_checkpoint()
