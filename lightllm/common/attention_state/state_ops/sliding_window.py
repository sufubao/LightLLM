from __future__ import annotations
import torch
from lightllm.common.attention_state.types import TensorField, PageKind
from lightllm.common.attention_state.state_ops.base import BoundedRuntime
from lightllm.common.attention_state.kernels.swa_snapshot import copy_swa_window


class SlidingWindowRuntime(BoundedRuntime):
    def layer_view(self, storage, logical_layer_id: int) -> torch.Tensor:
        """返回该层的 [request, ring_token, packed_byte] 视图。"""
        return storage.buffers["swa_buffer"][storage.layer_indices[logical_layer_id]]

    def reset(self, storage, req_indices: list[int]) -> None:
        for req_idx in req_indices:
            storage.buffers["swa_buffer"][:, req_idx].zero_()

    def save(
        self,
        storage,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        pool = storage.pool(page_kind=page_kind)
        for req_idx, length, slot in zip(req_indices, state_token_lens, page_slot_ids):
            self.copy_window(storage, req_idx=req_idx, external=pool.buffer[slot], state_token_len=length, write=True)

    def restore(
        self,
        storage,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        pool = storage.pool(page_kind=page_kind)
        for req_idx, length, slot in zip(req_indices, state_token_lens, page_slot_ids):
            self.copy_window(storage, req_idx=req_idx, external=pool.buffer[slot], state_token_len=length, write=False)

    def commit(self, storage, req_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
        """ring 按绝对 token 位置寻址，无需提交额外游标。"""
        pass

    def copy_window(self, storage, req_idx: int, external: torch.Tensor, state_token_len: int, write: bool) -> None:
        copy_swa_window(
            local=storage.buffers["swa_buffer"][:, req_idx],
            external=external,
            state_token_len=state_token_len,
            config=self.config,
            write=write,
        )

    def dp_fields(self, storage, req):
        return {"window": TensorField(storage.buffers["swa_buffer"][:, req], ("layer", "window", "channel"))}
