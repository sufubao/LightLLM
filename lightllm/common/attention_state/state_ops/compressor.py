from __future__ import annotations
import torch
from lightllm.common.attention_state.types import TensorField, PageKind
from lightllm.common.attention_state.state_ops.base import BoundedRuntime
from lightllm.common.attention_state.kernels.compressor_snapshot import copy_compressor_window


class CompressorRuntime(BoundedRuntime):
    def layer_view(self, storage, logical_layer_id: int) -> torch.Tensor:
        """返回该层 [request, window+MTP, candidate|gate] 的计算视图。"""
        return storage.buffers["buffer"][storage.layer_indices[logical_layer_id]]

    def reset(self, storage, req_indices: list[int]) -> None:
        for req_idx in req_indices:
            storage.buffers["buffer"][:, req_idx].zero_()
            storage.buffers["buffer"][:, req_idx, :, self.config.candidate_dim :].fill_(float("-inf"))

    def save(
        self,
        storage,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        """边界必须完成一个产出步长；仅保留下一步仍需读取的窗口重叠部分。"""
        assert all(
            (length % self.config.compress_stride == 0 for length in state_token_lens)
        ), "prefix must close a compressor stride"
        if not self.config.overlap_tokens:
            return
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
        assert all(
            (length % self.config.compress_stride == 0 for length in state_token_lens)
        ), "prefix must close a compressor stride"
        if not self.config.overlap_tokens:
            return
        pool = storage.pool(page_kind=page_kind)
        for req_idx, length, slot in zip(req_indices, state_token_lens, page_slot_ids):
            self.copy_window(storage, req_idx=req_idx, external=pool.buffer[slot], state_token_len=length, write=False)

    def commit(self, storage, req_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
        """按绝对位置寻址；接受后继续覆盖 ring，不维护额外 MTP 游标。"""
        pass

    def copy_window(self, storage, req_idx: int, external: torch.Tensor, state_token_len: int, write: bool) -> None:
        copy_compressor_window(
            local=storage.buffers["buffer"][:, req_idx], external=external, state_token_len=state_token_len, write=write
        )

    def dp_fields(self, storage, req):
        return {"window": TensorField(storage.buffers["buffer"][:, req], ("layer", "window", "channel"))}

    def prepare(self, storage):
        storage.buffers["buffer"][:, storage.request_capacity - 1, :, self.config.candidate_dim :].fill_(float("-inf"))
