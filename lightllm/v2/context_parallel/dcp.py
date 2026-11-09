"""Decode context-parallel KV ownership and slot mapping.

DCP reuses the TP workers inside each DP replica.  This module contains the
device-independent ownership rule used by paged KV allocators; collective
attention and process-group construction remain outside this pure contract.
"""

import operator
from dataclasses import dataclass
from typing import Any

import torch


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


def merge_attention_partials(
    partial_output: torch.Tensor, partial_lse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized attention outputs from DCP KV shards.

    ``partial_output`` has shape ``[shard, ..., hidden]`` and ``partial_lse``
    has shape ``[shard, ...]``.  Each shard output is assumed to be normalized
    by its own attention denominator; the log-sum-exp values provide the stable
    weights needed to reconstruct the result over the complete KV history.
    A shard with ``-inf`` LSE is empty and contributes no output.  If every
    shard is empty, the result is zero with an ``-inf`` LSE.

    This is a device-independent math contract.  It does not allocate KV
    pages, perform collectives, or select an attention backend.
    """

    if partial_output.ndim < 2:
        raise ValueError("partial_output must have shape [shard, ..., hidden]")
    if partial_lse.ndim != partial_output.ndim - 1:
        raise ValueError("partial_lse must have one fewer dimension than partial_output")
    if partial_output.shape[0] == 0:
        raise ValueError("partial_output must contain at least one shard")
    if partial_output.shape[0] != partial_lse.shape[0]:
        raise ValueError("partial_output and partial_lse must have the same shard dimension")
    if partial_output.shape[1:-1] != partial_lse.shape[1:]:
        raise ValueError("partial_output and partial_lse batch dimensions must match")
    if partial_output.device != partial_lse.device:
        raise ValueError("partial_output and partial_lse must be on the same device")
    if not partial_output.is_floating_point():
        raise ValueError("partial_output must use a floating-point dtype")
    if not partial_lse.is_floating_point():
        raise ValueError("partial_lse must use a floating-point dtype")
    if torch.isnan(partial_lse).any():
        raise ValueError("partial_lse must not contain NaN")
    if torch.isposinf(partial_lse).any():
        raise ValueError("partial_lse must not contain positive infinity")

    merged_lse = torch.logsumexp(partial_lse, dim=0)
    finite = torch.isfinite(merged_lse)
    normalized_lse = partial_lse - merged_lse.unsqueeze(0)
    weights = torch.where(finite.unsqueeze(0), normalized_lse.exp(), torch.zeros_like(normalized_lse))
    safe_output = torch.where(weights.unsqueeze(-1) == 0, torch.zeros_like(partial_output), partial_output)
    accumulation_dtype = torch.promote_types(partial_output.dtype, partial_lse.dtype)
    merged_output = (safe_output.to(accumulation_dtype) * weights.unsqueeze(-1).to(accumulation_dtype)).sum(dim=0)
    merged_output = torch.where(finite.unsqueeze(-1), merged_output, torch.zeros_like(merged_output))
    return merged_output.to(partial_output.dtype), merged_lse


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

    def runtime_layout(self):
        """Return the explicit process-group layout for a DCP worker."""

        from lightllm.v2.context_parallel_runtime import ContextParallelLayout

        return ContextParallelLayout(
            global_rank=self.global_rank,
            dp_size=self.dp_size,
            context_size=self.dcp_size,
            tp_size=self.tp_size // self.dcp_size,
            order="tp_major",
        )

    def create_runtime_groups(self, *, backend: str | None = None):
        """Create DCP/TP/DP groups after distributed bootstrap.

        The DCP contract treats ``dcp_size`` ranks as the context dimension;
        the remaining TP degree is the number of DCP subgroups per replica.
        Launcher code must call this consistently on every rank.
        """

        from lightllm.v2.context_parallel_runtime import create_context_parallel_groups

        return create_context_parallel_groups(self.runtime_layout(), backend=backend)
