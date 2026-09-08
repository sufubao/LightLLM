"""Recurrent checkpoints are identified by prefixes, independently of KV pages.

A checkpoint represents the state *after* its ``token_count`` input tokens.
In particular, an emitted token that has not run through the model is not part
of that state. Hashing at arbitrary retained lengths also supports output ends
that are not multiples of the prefill hash or CPU transfer page size.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import xxhash


def prefix_hashes(tokens: Sequence[int], lengths: Iterable[int]) -> dict[int, int]:
    """Hash whole prefixes in one pass, using the existing uint64 token format."""
    lengths = sorted(set(lengths))
    if not lengths:
        return {}
    if lengths[0] <= 0 or lengths[-1] > len(tokens):
        raise ValueError("checkpoint lengths must be within the computed token sequence")
    data = memoryview(np.asarray(tokens[: lengths[-1]], dtype=np.uint64)).cast("B")
    hasher = xxhash.xxh3_128()
    previous = 0
    result = {}
    for length in lengths:
        hasher.update(data[previous * 8 : length * 8])
        result[length] = hasher.intdigest()
        previous = length
    return result


@dataclass(frozen=True)
class StateCheckpoint:
    token_count: int
    prefix_hash: int
    slot: int


class CheckpointIndex:
    """Bounded LRU ownership of immutable, ready state slots.

    ``write`` must complete before returning. Until then the slot is private
    and cannot be matched. ``read`` runs while the entry is owned by this
    index; callers must finish copying before another mutation of the index.
    The inference thread serializes these operations, as it does radix edits.
    KV eviction has no effect on this index and state eviction does not free KV.
    """

    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("checkpoint capacity must be nonnegative")
        self.capacity = capacity
        self.entries: OrderedDict[tuple[int, int], StateCheckpoint] = OrderedDict()
        self.free_slots = list(reversed(range(capacity)))

    def insert(self, tokens: Sequence[int], token_count: int, write: Callable[[int], None]):
        prefix_hash = prefix_hashes(tokens, [token_count])[token_count]
        key = (token_count, prefix_hash)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        if self.capacity == 0:
            return None
        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            _, evicted = self.entries.popitem(last=False)
            slot = evicted.slot
        try:
            write(slot)
        except BaseException:
            self.free_slots.append(slot)
            raise
        checkpoint = StateCheckpoint(token_count, prefix_hash, slot)
        self.entries[key] = checkpoint
        return checkpoint

    def match(self, tokens: Sequence[int], max_tokens: int) -> Optional[StateCheckpoint]:
        """Find the last saved state within the caller's complete KV coverage."""
        lengths = {length for length, _ in self.entries if length <= min(max_tokens, len(tokens))}
        hashes = prefix_hashes(tokens, lengths)
        for length in sorted(lengths, reverse=True):
            key = (length, hashes[length])
            checkpoint = self.entries.get(key)
            if checkpoint is not None:
                self.entries.move_to_end(key)
                return checkpoint
        return None

    def matching_lengths(self, tokens: Sequence[int], max_tokens: int) -> set[int]:
        lengths = {n for n, _ in self.entries if n <= min(max_tokens, len(tokens))}
        hashes = prefix_hashes(tokens, lengths)
        return {n for n, h in hashes.items() if (n, h) in self.entries}

    def clear(self):
        self.entries.clear()
        self.free_slots = list(reversed(range(self.capacity)))


@dataclass(frozen=True)
class CheckpointPolicy:
    interval: int = 32768
    hash_page_size: int = 512

    def __post_init__(self):
        if self.interval < 0 or self.hash_page_size <= 0:
            raise ValueError("invalid checkpoint interval or hash page size")
        if self.interval and self.interval % self.hash_page_size:
            raise ValueError("checkpoint interval must be a multiple of the hash page size")

    def prefill_end(self, start: int, end: int, prompt_length: int, demand: int = 0) -> int:
        """Stop at a selected logical checkpoint, regardless of KV page size."""
        boundaries = [end, prompt_length]
        # Keep an endpoint usable by an identical prompt, which still needs a
        # token evaluated to produce logits, as well as the actual prompt end.
        boundaries.append((prompt_length - 1) // self.hash_page_size * self.hash_page_size)
        if self.interval:
            boundaries.append((start // self.interval + 1) * self.interval)
        if demand:
            boundaries.append(demand)
        return min(boundary for boundary in boundaries if boundary > start)

    def retain_prefill(self, length: int, prompt_length: int, demand: int = 0) -> bool:
        return length > 0 and (
            length == prompt_length
            or length == (prompt_length - 1) // self.hash_page_size * self.hash_page_size
            or (self.interval > 0 and length % self.interval == 0)
            or length == demand
        )
