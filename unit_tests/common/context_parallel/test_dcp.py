import pytest
import torch

from lightllm.common.context_parallel.dcp import PAD_SLOT, DecodeContextParallelConfig, merge_attention_partials


def test_dcp_reuses_contiguous_tp_subgroups():
    rank = DecodeContextParallelConfig(global_rank=3, tp_size=8, dcp_size=2, dp_size=2)
    assert (rank.dp_rank, rank.tp_rank, rank.dcp_rank) == (0, 3, 1)
    assert rank.dcp_group_ranks == (2, 3)

    next_replica = DecodeContextParallelConfig(global_rank=9, tp_size=8, dcp_size=2, dp_size=2)
    assert next_replica.dcp_group_ranks == (8, 9)


def test_interleaved_slot_mapping_is_compact_per_dcp_rank():
    rank0 = DecodeContextParallelConfig(tp_size=2, block_size=4, dcp_size=2, interleave_size=2)
    rank1 = DecodeContextParallelConfig(global_rank=1, tp_size=2, block_size=4, dcp_size=2, interleave_size=2)
    positions = tuple(range(8))

    assert rank0.slot_mapping(positions) == (0, 1, PAD_SLOT, PAD_SLOT, 2, 3, PAD_SLOT, PAD_SLOT)
    assert rank1.slot_mapping(positions) == (PAD_SLOT, PAD_SLOT, 0, 1, PAD_SLOT, PAD_SLOT, 2, 3)
    assert rank0.local_token_count(8) == rank1.local_token_count(8) == 4


def test_owner_wraps_at_virtual_block_boundary():
    config = DecodeContextParallelConfig(tp_size=2, block_size=4, dcp_size=2, interleave_size=1)
    assert [config.owner(position) for position in range(8)] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert config.local_slot(0) == 0
    assert config.local_slot(1) == PAD_SLOT
    assert config.local_slot(4) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tp_size": 3, "dcp_size": 2},
        {"tp_size": 2, "dcp_size": 0},
        {"block_size": 3, "interleave_size": 2},
        {"global_rank": 2, "tp_size": 2},
    ],
)
def test_invalid_dcp_layout_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DecodeContextParallelConfig(**kwargs)


@pytest.mark.parametrize("field", ["global_rank", "tp_size", "dcp_size", "dp_size", "block_size", "interleave_size"])
@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_non_integer_dcp_layout_values_are_rejected(field, value):
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        DecodeContextParallelConfig(**{field: value})


@pytest.mark.parametrize(
    ("method", "value"),
    [
        ("owner", True),
        ("owner", 1.5),
        ("local_slot", "1"),
        ("local_token_count", True),
        ("local_token_count", 1.5),
    ],
)
def test_non_integer_dcp_position_values_are_rejected(method, value):
    config = DecodeContextParallelConfig(tp_size=2, dcp_size=2)
    with pytest.raises(ValueError, match="must be an integer"):
        getattr(config, method)(value)


def test_slot_mapping_rejects_non_integer_positions():
    config = DecodeContextParallelConfig(tp_size=2, dcp_size=2)
    with pytest.raises(ValueError, match="must be an integer"):
        config.slot_mapping((0, "1"))


def test_runtime_layout_maps_dcp_to_context_dimension():
    config = DecodeContextParallelConfig(global_rank=3, tp_size=4, dcp_size=2, dp_size=2)

    layout = config.runtime_layout()

    assert (layout.dp_size, layout.context_size, layout.tp_size, layout.global_rank) == (2, 2, 2, 3)
    assert layout.order == "tp_major"
    assert layout.group_ranks("context") == (2, 3)


def test_create_runtime_groups_delegates_to_shared_owner(monkeypatch):
    config = DecodeContextParallelConfig(tp_size=2, dcp_size=2)
    captured = []
    monkeypatch.setattr(
        "lightllm.common.context_parallel_runtime.create_context_parallel_groups",
        lambda layout, backend: captured.append((layout, backend)) or "groups",
    )

    assert config.create_runtime_groups(backend="gloo") == "groups"
    assert captured[0][0] == config.runtime_layout()
    assert captured[0][1] == "gloo"


def test_merge_attention_partials_is_stable_and_matches_lse_weights():
    partial_output = torch.tensor(
        [
            [[[1.0, 3.0], [2.0, 4.0]]],
            [[[5.0, 7.0], [6.0, 8.0]]],
        ]
    )
    partial_lse = torch.tensor([[[0.0, 1.0]], [[2.0, -1.0]]])

    merged_output, merged_lse = merge_attention_partials(partial_output, partial_lse)

    weights = torch.softmax(partial_lse, dim=0)
    torch.testing.assert_close(merged_output, (partial_output * weights.unsqueeze(-1)).sum(dim=0))
    torch.testing.assert_close(merged_lse, torch.logsumexp(partial_lse, dim=0))

    low_precision_output, _ = merge_attention_partials(partial_output.half(), partial_lse)
    assert low_precision_output.dtype == partial_output.half().dtype


def test_merge_attention_partials_ignores_empty_shards_and_all_empty_is_zero():
    partial_output = torch.tensor(
        [
            [[[float("nan"), float("nan")]]],
            [[[2.0, 4.0]]],
        ]
    )
    partial_lse = torch.tensor([[[-float("inf")]], [[1.5]]])

    merged_output, merged_lse = merge_attention_partials(partial_output, partial_lse)
    torch.testing.assert_close(merged_output, torch.tensor([[[2.0, 4.0]]]))
    torch.testing.assert_close(merged_lse, torch.tensor([[1.5]]))

    empty_output, empty_lse = merge_attention_partials(
        torch.full((2, 1, 1, 2), float("nan")), torch.full((2, 1, 1), -float("inf"))
    )
    torch.testing.assert_close(empty_output, torch.zeros_like(empty_output))
    assert torch.isneginf(empty_lse).all()


@pytest.mark.parametrize(
    ("partial_output", "partial_lse", "message"),
    [
        (torch.ones(1), torch.ones(1), r"shape \[shard, \.\.\., hidden\]"),
        (torch.ones(1, 2), torch.ones(1, 2, 1), "one fewer dimension"),
        (torch.ones(2, 1, 3), torch.ones(1, 1), "same shard dimension"),
        (torch.ones(1, 2, 3), torch.ones(1, 4), "batch dimensions"),
    ],
)
def test_merge_attention_partials_rejects_shape_mismatches(partial_output, partial_lse, message):
    with pytest.raises(ValueError, match=message):
        merge_attention_partials(partial_output, partial_lse)


def test_merge_attention_partials_rejects_invalid_lse_values_and_dtypes():
    output = torch.ones(1, 1, 1, 2)
    with pytest.raises(ValueError, match="NaN"):
        merge_attention_partials(output, torch.tensor([[[float("nan")]]]))
    with pytest.raises(ValueError, match="positive infinity"):
        merge_attention_partials(output, torch.tensor([[[float("inf")]]]))
    with pytest.raises(ValueError, match="floating-point dtype"):
        merge_attention_partials(output.to(torch.int64), torch.ones(1, 1, 1))
    with pytest.raises(ValueError, match="floating-point dtype"):
        merge_attention_partials(output, torch.ones(1, 1, 1, dtype=torch.int64))
