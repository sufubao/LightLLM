import pytest

from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.utils.kv_cache_utils import _normalize_registration_ranges


def test_normalize_registration_ranges_aligns_and_merges():
    assert _normalize_registration_ranges(
        size=32 * 1024,
        ranges=((4097, 4095), (8192, 4096), (20 * 1024, 1)),
        page_size=4096,
    ) == ((4096, 8192), (20 * 1024, 4096))
    assert _normalize_registration_ranges(size=5000, ranges=((4999, 1),), page_size=4096) == ((4096, 904),)

    with pytest.raises(ValueError, match="invalid registration range"):
        _normalize_registration_ranges(size=4096, ranges=((4096, 1),), page_size=4096)


def test_linear_cache_registers_only_current_tp_rank_ranges():
    config = object.__new__(LinearAttCacheConfig)
    config.tp_world_size = 4
    config.full_att_all_num_kv_heads = 4
    config.get_cpu_cache_full_att_bytes = lambda: 16 * 4096
    config.get_cpu_cache_conv_bytes = lambda: 8 * 4096
    config.get_cpu_cache_ssm_bytes = lambda: 4 * 4096
    config.get_cpu_cache_big_page_bytes = lambda: 28 * 4096

    ranges = config.get_cpu_cache_rank_registration_ranges(page_num=2, tp_rank=2)

    assert ranges[:3] == ((0, 1), (16 * 4096, 1), (24 * 4096, 1))
    assert ranges[3:] == (
        (8 * 4096, 4 * 4096),
        (20 * 4096, 2 * 4096),
        (26 * 4096, 4096),
        (36 * 4096, 4 * 4096),
        (48 * 4096, 2 * 4096),
        (54 * 4096, 4096),
    )
    normalized = _normalize_registration_ranges(size=2 * 28 * 4096, ranges=ranges, page_size=4096)
    assert sum(length for _, length in normalized) == 2 * 28 * 4096 // 4 + 3 * 4096
