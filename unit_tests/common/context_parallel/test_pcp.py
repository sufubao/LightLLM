import pytest

from lightllm.common.context_parallel.pcp import PrefillContextParallelConfig, PrefillTokenRange, rank_layout


def test_rank_layout_is_dp_pcp_tp_ordered():
    assert rank_layout(0, tp_size=2, pcp_size=2, dp_size=2) == (0, 0, 0)
    assert rank_layout(3, tp_size=2, pcp_size=2, dp_size=2) == (0, 1, 1)
    assert rank_layout(4, tp_size=2, pcp_size=2, dp_size=2) == (1, 0, 0)
    assert rank_layout(7, tp_size=2, pcp_size=2, dp_size=2) == (1, 1, 1)


def test_pcp_group_keeps_tp_shard_constant():
    config = PrefillContextParallelConfig(global_rank=3, tp_size=2, pcp_size=2, dp_size=2)
    assert (config.dp_rank, config.pcp_rank, config.tp_rank) == (0, 1, 1)
    assert config.group_ranks == (1, 3)
    assert config.world_size == 8


@pytest.mark.parametrize(
    ("token_count", "expected"),
    [
        (0, ((0, 0), (0, 0), (0, 0))),
        (1, ((0, 1), (1, 1), (1, 1))),
        (8, ((0, 3), (3, 6), (6, 8))),
    ],
)
def test_token_ranges_are_balanced_and_cover_prompt(token_count, expected):
    config = PrefillContextParallelConfig(pcp_size=3)
    ranges = config.token_ranges(token_count)
    assert tuple((item.start, item.end) for item in ranges) == expected
    assert ranges[0].start == 0
    assert ranges[-1].end == token_count
    assert all(left.end == right.start for left, right in zip(ranges, ranges[1:]))
    assert config.token_range(token_count) == ranges[0]


def test_empty_ranges_are_valid_when_prompt_is_shorter_than_pcp_group():
    config = PrefillContextParallelConfig(global_rank=2, pcp_size=3)
    assert config.token_range(1) == PrefillTokenRange(1, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"global_rank": 2, "tp_size": 2, "pcp_size": 1},
        {"global_rank": 0, "tp_size": 0, "pcp_size": 1},
        {"global_rank": 0, "tp_size": 1, "pcp_size": 0},
    ],
)
def test_invalid_topology_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PrefillContextParallelConfig(**kwargs)
