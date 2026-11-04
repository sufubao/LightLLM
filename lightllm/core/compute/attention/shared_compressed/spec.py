from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SharedCompressedAttentionSpec:
    heads: int
    head_dim: int
    rope_dim: int
    window_size: int
    ratio: int
    kv_source: int | None
    index_source: int | None
    layer_id: int
    index_heads: int
    total_index_heads: int
    index_dim: int
    index_topk: int
    candidate_source: int
    candidate_blocks: int
    candidate_block_size: int
    dtype: Any

    def validate_state(self, request):
        if request.state.shape != (self.window_size, self.head_dim) or request.state.retention != "window":
            raise ValueError("shared compressed attention requires a private sliding window")
        expected = {"compressed": (self.head_dim,), "index": (self.index_dim,)} if self.ratio else {}
        actual = {name: state.shape for name, state, _ in request.auxiliary_states}
        if actual != expected or request.params.spec != self:
            raise ValueError("shared compressed attention state or static parameters disagree")
        for name, state, _ in request.auxiliary_states:
            if state.producer != self.kv_source or state.retention != "history":
                raise ValueError(f"{name} must name the declared compressed KV producer")
