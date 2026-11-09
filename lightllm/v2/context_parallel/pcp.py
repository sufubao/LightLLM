"""Prefill context-parallel topology and token partitioning.

This module owns the deterministic, device-independent part of PCP.  Attention
backends can use the returned ranges to build their local query/key/value
inputs without duplicating rank arithmetic.  Communication and attention
execution remain backend responsibilities.
"""

import operator
from dataclasses import dataclass, field
from typing import Any, Sequence

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


def assemble_prefill_slices(
    slices: Sequence[torch.Tensor], ranges: Sequence["PrefillTokenRange"], token_count: int
) -> torch.Tensor:
    """Assemble rank-local prefill outputs in global token order.

    The caller owns the collective that produced ``slices``.  This function
    only validates the global ranges and performs the deterministic local
    assembly; it therefore remains useful in CPU tests and does not imply
    distributed runtime support.
    """

    token_count = _validated_int(token_count, "token_count", minimum=0)
    if len(slices) != len(ranges) or not slices:
        raise ValueError("slices and ranges must contain the same nonzero number of entries")

    cursor = 0
    trailing_shape = None
    reference_dtype = None
    reference_device = None
    for index, (local, token_range) in enumerate(zip(slices, ranges)):
        if not isinstance(token_range, PrefillTokenRange):
            raise ValueError(f"ranges[{index}] must be a PrefillTokenRange")
        if token_range.start != cursor:
            raise ValueError(
                f"ranges must be ordered and contiguous; ranges[{index}] starts at {token_range.start}, "
                f"expected {cursor}"
            )
        if not isinstance(local, torch.Tensor) or local.ndim == 0:
            raise ValueError(f"slices[{index}] must be a tensor with a token dimension")
        if local.shape[0] != token_range.length:
            raise ValueError(
                f"slices[{index}] has {local.shape[0]} tokens, expected {token_range.length} for its range"
            )
        if trailing_shape is None:
            trailing_shape = local.shape[1:]
            reference_dtype = local.dtype
            reference_device = local.device
        elif local.shape[1:] != trailing_shape:
            raise ValueError(
                f"slices[{index}] trailing shape {tuple(local.shape[1:])} does not match {tuple(trailing_shape)}"
            )
        if local.dtype != reference_dtype or local.device != reference_device:
            raise ValueError("all slices must use the same dtype and device")
        cursor = token_range.end

    if cursor != token_count:
        raise ValueError(f"ranges cover {cursor} tokens, expected token_count {token_count}")
    return torch.cat(tuple(slices), dim=0)


@dataclass(frozen=True)
class PrefillTokenRange:
    """The half-open token interval assigned to one PCP rank."""

    start: int
    end: int

    def __post_init__(self) -> None:
        start = _validated_int(self.start, "start", minimum=0)
        end = _validated_int(self.end, "end", minimum=0)
        if end < start:
            raise ValueError(f"token range must satisfy 0 <= start <= end, got ({start}, {end})")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PrefillTokenMetadata:
    """Global coordinates for the local slice of one prefill PCP rank.

    ``token_range`` is expressed in the unsharded prompt coordinate space.  A
    backend can therefore slice token-aligned tensors and retain enough
    metadata to map local outputs back to the original prompt.  This class
    does not describe attention state or communication; those remain backend
    responsibilities.
    """

    token_count: int
    token_range: PrefillTokenRange
    pcp_rank: int
    pcp_size: int

    def __post_init__(self) -> None:
        token_count = _validated_int(self.token_count, "token_count", minimum=0)
        pcp_rank = _validated_int(self.pcp_rank, "pcp_rank", minimum=0)
        pcp_size = _validated_int(self.pcp_size, "pcp_size", minimum=1)
        if pcp_rank >= pcp_size:
            raise ValueError(f"pcp_rank must be in [0, {pcp_size}), got {pcp_rank}")
        if self.token_range.end > token_count:
            raise ValueError(
                f"token range ({self.token_range.start}, {self.token_range.end}) exceeds token_count {token_count}"
            )
        object.__setattr__(self, "token_count", token_count)
        object.__setattr__(self, "pcp_rank", pcp_rank)
        object.__setattr__(self, "pcp_size", pcp_size)

    @property
    def global_start(self) -> int:
        return self.token_range.start

    @property
    def global_end(self) -> int:
        return self.token_range.end

    @property
    def local_token_count(self) -> int:
        return self.token_range.length


def rank_layout(global_rank: int, *, tp_size: int, pcp_size: int, dp_size: int = 1) -> tuple[int, int, int]:
    """Return ``(dp_rank, pcp_rank, tp_rank)`` for the process rank.

    Ranks are laid out as ``DP -> PCP -> TP``.  This makes every PCP group
    contain the same TP ranks while keeping the existing DP/TP rank arithmetic
    stable when ``pcp_size == 1``.
    """

    global_rank = _validated_int(global_rank, "global_rank", minimum=0)
    tp_size = _validated_int(tp_size, "tp_size", minimum=1)
    pcp_size = _validated_int(pcp_size, "pcp_size", minimum=1)
    dp_size = _validated_int(dp_size, "dp_size", minimum=1)
    world_size = dp_size * pcp_size * tp_size
    if not 0 <= global_rank < world_size:
        raise ValueError(f"global_rank must be in [0, {world_size}), got {global_rank}")

    rank_in_dp, tp_rank = divmod(global_rank, tp_size)
    dp_rank, pcp_rank = divmod(rank_in_dp, pcp_size)
    return dp_rank, pcp_rank, tp_rank


@dataclass(frozen=True)
class PrefillContextParallelConfig:
    """Validated PCP topology for one prefill worker.

    ``global_rank`` is the process rank in the DP×PCP×TP layout.  The config
    intentionally does not create a process group: group creation belongs to
    the distributed bootstrap, while token partitioning is safe to test and
    reuse before a backend is initialized.
    """

    global_rank: int = 0
    tp_size: int = 1
    pcp_size: int = 1
    dp_size: int = 1
    dp_rank: int = field(init=False)
    pcp_rank: int = field(init=False)
    tp_rank: int = field(init=False)

    def __post_init__(self) -> None:
        dp_rank, pcp_rank, tp_rank = rank_layout(
            self.global_rank, tp_size=self.tp_size, pcp_size=self.pcp_size, dp_size=self.dp_size
        )
        object.__setattr__(self, "dp_rank", dp_rank)
        object.__setattr__(self, "pcp_rank", pcp_rank)
        object.__setattr__(self, "tp_rank", tp_rank)

    @property
    def world_size(self) -> int:
        return self.dp_size * self.pcp_size * self.tp_size

    @property
    def group_ranks(self) -> tuple[int, ...]:
        """Global ranks in this DP replica's PCP group for this TP shard."""

        base = self.dp_rank * self.pcp_size * self.tp_size
        return tuple(base + pcp_rank * self.tp_size + self.tp_rank for pcp_rank in range(self.pcp_size))

    def token_range(self, token_count: int) -> PrefillTokenRange:
        """Return this rank's balanced contiguous range for a prefill prompt."""

        token_count = _validated_int(token_count, "token_count", minimum=0)
        base, remainder = divmod(token_count, self.pcp_size)
        start = self.pcp_rank * base + min(self.pcp_rank, remainder)
        return PrefillTokenRange(start, start + base + (self.pcp_rank < remainder))

    def token_ranges(self, token_count: int) -> tuple[PrefillTokenRange, ...]:
        """Return all PCP ranges in rank order and prove they cover the prompt."""

        token_count = _validated_int(token_count, "token_count", minimum=0)
        base, remainder = divmod(token_count, self.pcp_size)
        ranges = []
        start = 0
        for pcp_rank in range(self.pcp_size):
            length = base + (pcp_rank < remainder)
            ranges.append(PrefillTokenRange(start, start + length))
            start += length
        return tuple(ranges)

    def token_metadata(self, token_count: int) -> PrefillTokenMetadata:
        """Return this rank's range together with its global prompt metadata."""

        token_count = _validated_int(token_count, "token_count", minimum=0)
        return PrefillTokenMetadata(
            token_count=token_count,
            token_range=self.token_range(token_count),
            pcp_rank=self.pcp_rank,
            pcp_size=self.pcp_size,
        )

    def slice_tokens(self, tokens: Any) -> tuple[Any, PrefillTokenMetadata]:
        """Slice a token-aligned buffer along its first dimension.

        The input may be a Python sequence, NumPy array, or tensor-like object;
        the object's normal slice semantics (view or copy) are preserved.  The
        returned metadata remains in global prompt coordinates so callers can
        perform the required attention and output reconciliation explicitly.
        """

        metadata = self.token_metadata(len(tokens))
        return tokens[metadata.global_start : metadata.global_end], metadata

    def runtime_layout(self):
        """Return the explicit process-group layout for a PCP worker."""

        from lightllm.v2.context_parallel_runtime import ContextParallelLayout

        return ContextParallelLayout(
            global_rank=self.global_rank,
            dp_size=self.dp_size,
            context_size=self.pcp_size,
            tp_size=self.tp_size,
        )

    def create_runtime_groups(self, *, backend: str | None = None):
        """Create PCP/TP/DP groups after distributed bootstrap.

        This method owns no global state and does not initialize the default
        process group; launcher code must call it consistently on every rank.
        """

        from lightllm.v2.context_parallel_runtime import create_context_parallel_groups

        return create_context_parallel_groups(self.runtime_layout(), backend=backend)
