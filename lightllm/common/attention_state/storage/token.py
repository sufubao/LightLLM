import torch
from lightllm.common.attention_state.layouts import TokenStorageSpec


class TokenStorage:
    def __init__(self, spec: TokenStorageSpec):
        self.spec = spec
        self.capacity = 0
        self.buffers: dict[str, torch.Tensor] = {}
        self._layer_indices = {
            logical: physical for physical, group in enumerate(spec.logical_layer_groups) for logical in group
        }

    def layer_index(self, logical_layer_id: int) -> int:
        return self._layer_indices[logical_layer_id]

    def contains_layer(self, logical_layer_id: int) -> bool:
        return logical_layer_id in self._layer_indices

    def allocate(self, token_capacity: int, hold_tokens: int, device: torch.device) -> None:
        assert token_capacity >= 0 and hold_tokens > 0
        assert (
            hold_tokens % self.spec.block_tokens == 0
        ), "allocator page_size must be a positive multiple of block_token_size"
        assert token_capacity % self.spec.block_tokens == 0, "token capacity must contain complete compressed blocks"
        allocate = torch.zeros if self.spec.zero_initialize else torch.empty
        buffers = {
            field.name: allocate(
                self.spec.shape(field, token_capacity + hold_tokens),
                dtype=field.dtype,
                device=device,
            )
            for field in self.spec.fields
        }
        self.buffers = buffers
        self.capacity = token_capacity
