from dataclasses import dataclass
from operator import index
from typing import Any

from lightllm.common.attention_state.full_att import GQAFullAttentionState
from lightllm.common.execution.contracts.state import FormatSpec, StateSpec, StateView


@dataclass(frozen=True)
class GQAStateParams:
    """Logical GQA dimensions copied from an existing native state."""

    global_head_num: int
    head_dim: int
    dtype: Any
    tp_rank: int
    tp_world_size: int
    replicated: bool

    def validate(self) -> None:
        if self.global_head_num <= 0 or self.head_dim <= 0:
            raise ValueError("GQA state dimensions must be positive")
        if self.tp_world_size <= 0 or not 0 <= self.tp_rank < self.tp_world_size:
            raise ValueError("GQA TP rank must be inside its world")


class GQAStateAdapter:
    """Expose one existing GQA state through the execution state contract."""

    def __init__(self, state: GQAFullAttentionState, *, role: str = "main", retention: str = "history") -> None:
        if not isinstance(state, GQAFullAttentionState):
            raise TypeError("GQAStateAdapter requires a GQAFullAttentionState")
        if not role or not retention:
            raise ValueError("GQA state role and retention must be non-empty")
        self.state = state
        self.role = role
        self.retention = retention

    def spec_for(self, logical_layer_id: int) -> StateSpec:
        if isinstance(logical_layer_id, bool):
            raise ValueError("logical layer ID must be an integer")
        try:
            logical_layer_id = index(logical_layer_id)
        except TypeError as exc:
            raise ValueError("logical layer ID must be an integer") from exc
        for group_index, group in enumerate(self.state.config.logical_layer_groups):
            if logical_layer_id in group:
                params = GQAStateParams(
                    global_head_num=self.state.config.global_head_num,
                    head_dim=self.state.config.head_dim,
                    dtype=self.state.config.dtype,
                    tp_rank=self.state.config.tp_rank,
                    tp_world_size=self.state.config.tp_world_size,
                    replicated=self.state.config.replicated,
                )
                return StateSpec(
                    self.role,
                    f"gqa.{group_index}",
                    group[0],
                    tuple(group),
                    params,
                    self.retention,
                )
        raise ValueError(f"logical layer {logical_layer_id} is not registered in the GQA state")

    def view_for(self, logical_layer_id: int) -> StateView:
        if self.state.k_storage is None or self.state.v_storage is None:
            raise RuntimeError("GQA state storage must be allocated before creating a StateView")
        spec = self.spec_for(logical_layer_id)
        config = self.state.config
        format_spec = FormatSpec(
            "plain",
            self.state.storage_dtype,
            "token_head_dim",
            ("k", "v"),
            self.state.page_size,
            shard=(config.tp_rank, config.tp_world_size),
        )
        return StateView(spec, format_spec, self.state.get_layer_view(logical_layer_id))


class GQAStateStore:
    """Own the lifecycle of one native GQA state behind the execution contract.

    ``GQAFullAttentionState`` remains the owner of layout-specific operations.
    This class owns only the transition between unallocated, allocated, and
    closed storage, so callers no longer need to reach into ``k_storage`` or
    ``v_storage`` to determine whether a view is usable.
    """

    def __init__(self, adapter: GQAStateAdapter) -> None:
        if not isinstance(adapter, GQAStateAdapter):
            raise TypeError("GQAStateStore requires a GQAStateAdapter")
        self.adapter = adapter
        state = adapter.state
        if (state.k_storage is None) != (state.v_storage is None):
            raise RuntimeError("GQA native storage must contain both K and V fields")
        self._allocated = state.k_storage is not None
        self._closed = False

    @property
    def is_allocated(self) -> bool:
        """Whether this store currently owns allocated native storage."""
        return self._allocated and not self._closed

    def allocate_storage(self, token_capacity: int) -> None:
        """Allocate native storage exactly once for this store."""
        if self._closed:
            raise RuntimeError("cannot allocate a closed GQA state store")
        if self._allocated:
            raise RuntimeError("GQA state storage is already allocated")
        if isinstance(token_capacity, bool) or not isinstance(token_capacity, int) or token_capacity <= 0:
            raise ValueError("GQA token capacity must be a positive integer")
        state = self.adapter.state
        if state.k_storage is not None or state.v_storage is not None:
            raise RuntimeError("GQA native storage was allocated outside its execution store")
        state.allocate_storage(token_capacity)
        if state.k_storage is None or state.v_storage is None:
            raise RuntimeError("GQA native state did not allocate both K and V storage")
        self._allocated = True

    def view_for(self, logical_layer_id: int) -> StateView:
        """Return a contract view from allocated, live storage."""
        if not self.is_allocated:
            raise RuntimeError("GQA state storage must be allocated before creating a StateView")
        return self.adapter.view_for(logical_layer_id)

    def close(self) -> None:
        """Release native tensors and make future views fail deterministically."""
        if self._closed:
            return
        state = self.adapter.state
        state.k_storage = None
        state.v_storage = None
        state.capacity = 0
        self._allocated = False
        self._closed = True
