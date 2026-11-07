import pytest
import torch

from lightllm.common.context_parallel.pcp import (
    PrefillContextParallelConfig,
    PrefillTokenMetadata,
    PrefillTokenRange,
    assemble_prefill_slices,
    rank_layout,
)


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


def test_slice_tokens_returns_local_buffer_and_global_metadata():
    config = PrefillContextParallelConfig(global_rank=1, pcp_size=3)

    local_tokens, metadata = config.slice_tokens(list(range(8)))

    assert local_tokens == [3, 4, 5]
    assert metadata == PrefillTokenMetadata(8, PrefillTokenRange(3, 6), pcp_rank=1, pcp_size=3)
    assert (metadata.global_start, metadata.global_end, metadata.local_token_count) == (3, 6, 3)


def test_slice_tokens_preserves_empty_local_ranges():
    config = PrefillContextParallelConfig(global_rank=2, pcp_size=3)

    local_tokens, metadata = config.slice_tokens([10])

    assert local_tokens == []
    assert metadata.global_start == metadata.global_end == 1
    assert metadata.token_count == 1


def test_assemble_prefill_slices_restores_global_token_order():
    config = PrefillContextParallelConfig(pcp_size=3)
    ranges = config.token_ranges(5)
    slices = (torch.tensor([[0], [1]]), torch.tensor([[2], [3]]), torch.tensor([[4]]))

    assembled = assemble_prefill_slices(slices, ranges, token_count=5)

    torch.testing.assert_close(assembled, torch.arange(5).reshape(5, 1))


def test_assemble_prefill_slices_accepts_empty_local_shards():
    config = PrefillContextParallelConfig(pcp_size=3)
    ranges = config.token_ranges(1)
    slices = (torch.tensor([[10.0]]), torch.empty((0, 1)), torch.empty((0, 1)))

    assembled = assemble_prefill_slices(slices, ranges, token_count=1)

    torch.testing.assert_close(assembled, torch.tensor([[10.0]]))


@pytest.mark.parametrize(
    ("slices", "ranges", "token_count", "message"),
    [
        (
            (torch.ones(1, 2), torch.ones(1, 2)),
            (PrefillTokenRange(1, 2), PrefillTokenRange(0, 1)),
            2,
            "ordered and contiguous",
        ),
        (
            (torch.ones(1, 2),),
            (PrefillTokenRange(0, 1),),
            2,
            "cover",
        ),
        (
            (torch.ones(1, 2), torch.ones(1, 3)),
            (PrefillTokenRange(0, 1), PrefillTokenRange(1, 2)),
            2,
            "trailing shape",
        ),
    ],
)
def test_assemble_prefill_slices_rejects_invalid_ranges(slices, ranges, token_count, message):
    with pytest.raises(ValueError, match=message):
        assemble_prefill_slices(slices, ranges, token_count)


def test_runtime_layout_maps_pcp_to_context_dimension():
    config = PrefillContextParallelConfig(global_rank=3, tp_size=2, pcp_size=2, dp_size=2)

    layout = config.runtime_layout()

    assert (layout.dp_size, layout.context_size, layout.tp_size, layout.global_rank) == (2, 2, 2, 3)


def test_create_runtime_groups_delegates_to_shared_owner(monkeypatch):
    config = PrefillContextParallelConfig(pcp_size=2)
    captured = []
    monkeypatch.setattr(
        "lightllm.common.context_parallel_runtime.create_context_parallel_groups",
        lambda layout, backend: captured.append((layout, backend)) or "groups",
    )

    assert config.create_runtime_groups(backend="gloo") == "groups"
    assert captured[0][0] == config.runtime_layout()
    assert captured[0][1] == "gloo"


def test_prefill_token_metadata_rejects_range_outside_prompt():
    with pytest.raises(ValueError, match="exceeds token_count"):
        PrefillTokenMetadata(2, PrefillTokenRange(1, 3), pcp_rank=0, pcp_size=1)


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


@pytest.mark.parametrize("token_count", [-1, 1.5, True])
def test_invalid_token_counts_are_rejected(token_count):
    with pytest.raises(ValueError):
        PrefillContextParallelConfig(pcp_size=2).token_ranges(token_count)
