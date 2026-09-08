import pytest

from lightllm.common.linear_att_cache_manager.checkpoints import CheckpointIndex, CheckpointPolicy, prefix_hashes


def test_output_checkpoint_inside_kv_page_and_fine_hash_bucket():
    tokens = list(range(12000))
    cache = CheckpointIndex(4)
    writes = []
    cache.insert(tokens, 8192, writes.append)
    output = cache.insert(tokens, 10243, writes.append)
    assert cache.match(tokens, 11000) == output
    assert cache.match(tokens, 10242).token_count == 8192
    # Matching the same tail with a different preceding prefix is insufficient.
    different_prefix = tokens.copy()
    different_prefix[0] = 12345
    assert cache.match(different_prefix, 11000) is None


def test_state_capacity_and_eviction_are_independent_of_token_count():
    cache = CheckpointIndex(2)
    tokens = list(range(100000))
    first = cache.insert(tokens, 2, lambda slot: None)
    cache.insert(tokens, 32000, lambda slot: None)
    assert cache.match(tokens, 2) == first
    cache.insert(tokens, 99999, lambda slot: None)
    assert cache.match(tokens, 32000) == first
    assert len(cache.entries) == 2
    cache.clear()
    assert cache.match(tokens, len(tokens)) is None
    assert len(cache.free_slots) == 2


def test_snapshot_is_not_published_before_copy_and_failed_copy_releases_slot():
    cache = CheckpointIndex(1)
    tokens = [1, 2, 3]

    def copy(slot):
        assert cache.match(tokens, 3) is None
        raise RuntimeError("copy failed")

    with pytest.raises(RuntimeError, match="copy failed"):
        cache.insert(tokens, 2, copy)
    assert not cache.entries
    assert cache.free_slots == [0]
    checkpoint = cache.insert(tokens, 2, lambda slot: None)
    assert cache.insert(tokens, 2, copy) == checkpoint  # immutable duplicate


def test_hash_is_independent_of_intermediate_boundaries():
    tokens = list(range(20))
    assert prefix_hashes(tokens, [3, 7, 19])[19] == prefix_hashes(tokens, [19])[19]
    with pytest.raises(ValueError):
        prefix_hashes(tokens, [21])


def test_checkpoint_schedule_and_demand_retention():
    policy = CheckpointPolicy(interval=32768, hash_page_size=512)
    assert policy.prefill_end(8192, 20000, 50000) == 20000
    assert policy.prefill_end(30000, 40000, 50000) == 32768
    assert policy.prefill_end(8192, 20000, 50000, demand=10240) == 10240
    assert policy.retain_prefill(10240, 50000, demand=10240)
    assert not policy.retain_prefill(8192, 50000)
    ends_only = CheckpointPolicy(interval=0, hash_page_size=512)
    assert ends_only.prefill_end(0, 60000, 50000) == 49664
    assert ends_only.prefill_end(49664, 60000, 50000) == 50000
    assert not ends_only.retain_prefill(32768, 50000)
