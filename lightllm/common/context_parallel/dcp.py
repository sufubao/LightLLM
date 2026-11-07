"""Decode context-parallel KV ownership and slot mapping.

DCP reuses the TP workers inside each DP replica.  This module contains the
device-independent ownership rule used by paged KV allocators; collective
attention and process-group construction remain outside this pure contract.
"""

import operator
from dataclasses import dataclass
from typing import Any


def _validated_int(value: Any, name: str, *, minimum: int) -> int:
    """Return an integer argument or raise a useful topology error."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


PAD_SLOT = -1


@dataclass(frozen=True)
class DecodeContextParallelConfig:
    """Validated DCP layout for one worker and one paged KV cache."""

    global_rank: int = 0
    tp_size: int = 1
    dcp_size: int = 1
    dp_size: int = 1
    block_size: int = 1
    interleave_size: int = 1

    def __post_init__(self) -> None:
        for name, minimum in (
            ("global_rank", 0),
            ("tp_size", 1),
            ("dcp_size", 1),
            ("dp_size", 1),
            ("block_size", 1),
            ("interleave_size", 1),
        ):
            object.__setattr__(self, name, _validated_int(getattr(self, name), name, minimum=minimum))
        if self.tp_size % self.dcp_size:
            raise ValueError("dcp_size must divide tp_size")
        if self.block_size % self.interleave_size:
            raise ValueError("block_size must be divisible by interleave_size")
        world_size = self.dp_size * self.tp_size
        if not 0 <= self.global_rank < world_size:
            raise ValueError(f"global_rank must be in [0, {world_size}), got {self.global_rank}")

    @property
    def dp_rank(self) -> int:
        return self.global_rank // self.tp_size

    @property
    def tp_rank(self) -> int:
        return self.global_rank % self.tp_size

    @property
    def dcp_rank(self) -> int:
        """Rank within the DCP group that reuses a contiguous TP subgroup."""

        return self.tp_rank % self.dcp_size

    @property
    def dcp_group_ranks(self) -> tuple[int, ...]:
        group_start = self.dp_rank * self.tp_size + (self.tp_rank // self.dcp_size) * self.dcp_size
        return tuple(group_start + rank for rank in range(self.dcp_size))

    @property
    def virtual_block_size(self) -> int:
        return self.block_size * self.dcp_size

    def owner(self, token_position: int) -> int:
        """Return the DCP rank that stores one sequence position."""

        token_position = _validated_int(token_position, "token_position", minimum=0)
        offset = token_position % self.virtual_block_size
        return (offset // self.interleave_size) % self.dcp_size

    def local_slot(self, token_position: int) -> int:
        """Return the compact local slot, or ``PAD_SLOT`` when not owned."""

        token_position = _validated_int(token_position, "token_position", minimum=0)
        offset = token_position % self.virtual_block_size
        if (offset // self.interleave_size) % self.dcp_size != self.dcp_rank:
            return PAD_SLOT
        virtual_block, offset = divmod(token_position, self.virtual_block_size)
        interleave_index, intra_interleave = divmod(offset, self.interleave_size)
        local_offset = (interleave_index // self.dcp_size) * self.interleave_size + intra_interleave
        return virtual_block * self.block_size + local_offset

    def slot_mapping(self, token_positions: tuple[int, ...] | list[int]) -> tuple[int, ...]:
        """Map global token positions to local KV slots and ``PAD_SLOT``."""

        return tuple(self.local_slot(position) for position in token_positions)

    def local_token_count(self, sequence_length: int) -> int:
        """Return the number of KV entries owned below ``sequence_length``."""

        sequence_length = _validated_int(sequence_length, "sequence_length", minimum=0)
        return sum(self.owner(position) == self.dcp_rank for position in range(sequence_length))
