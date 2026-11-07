"""Prefill context-parallel topology and token partitioning.

This module owns the deterministic, device-independent part of PCP.  Attention
backends can use the returned ranges to build their local query/key/value
inputs without duplicating rank arithmetic.  Communication and attention
execution remain backend responsibilities.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrefillTokenRange:
    """The half-open token interval assigned to one PCP rank."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("token range must satisfy 0 <= start <= end")

    @property
    def length(self) -> int:
        return self.end - self.start


def rank_layout(global_rank: int, *, tp_size: int, pcp_size: int, dp_size: int = 1) -> tuple[int, int, int]:
    """Return ``(dp_rank, pcp_rank, tp_rank)`` for the process rank.

    Ranks are laid out as ``DP -> PCP -> TP``.  This makes every PCP group
    contain the same TP ranks while keeping the existing DP/TP rank arithmetic
    stable when ``pcp_size == 1``.
    """

    if tp_size <= 0 or pcp_size <= 0 or dp_size <= 0:
        raise ValueError("parallel sizes must be positive")
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

        if token_count < 0:
            raise ValueError("token_count must be nonnegative")
        base, remainder = divmod(token_count, self.pcp_size)
        start = self.pcp_rank * base + min(self.pcp_rank, remainder)
        return PrefillTokenRange(start, start + base + (self.pcp_rank < remainder))

    def token_ranges(self, token_count: int) -> tuple[PrefillTokenRange, ...]:
        """Return all PCP ranges in rank order and prove they cover the prompt."""

        if token_count < 0:
            raise ValueError("token_count must be nonnegative")
        base, remainder = divmod(token_count, self.pcp_size)
        ranges = []
        start = 0
        for pcp_rank in range(self.pcp_size):
            length = base + (pcp_rank < remainder)
            ranges.append(PrefillTokenRange(start, start + length))
            start += length
        return tuple(ranges)
