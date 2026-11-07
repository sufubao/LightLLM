import pytest

from lightllm.common.context_parallel.dcp import PAD_SLOT, DecodeContextParallelConfig


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
