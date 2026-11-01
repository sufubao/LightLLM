from __future__ import annotations
from typing import List
import torch
from lightllm.common.attention_state.types import TensorField, PageKind
from lightllm.common.attention_state.state_ops.base import BoundedRuntime
from lightllm.common.attention_state.kernels.copy import copy_bounded_att_tensors


class ConvSSMRuntime(BoundedRuntime):
    def layer_view(self, storage, logical_layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑层号映射到物理层，返回 Conv/SSM Tensor 引用；MTP 游标和步数直接从 State 读取。"""
        storage_layer_index = storage.layer_indices[logical_layer_id]
        return (storage.buffers["conv_state"][storage_layer_index], storage.buffers["ssm_state"][storage_layer_index])

    def reset(self, storage, req_indices: List[int]) -> None:
        """清零指定请求的 Conv 历史、全部 SSM 候选及 MTP 游标，保留页池快照。"""
        assert isinstance(req_indices, list), "request indices must be a list"
        for req_idx in req_indices:
            storage.buffers["conv_state"][:, req_idx].zero_()
            start = req_idx * (storage.mtp_step + 1)
            storage.buffers["ssm_state"][:, start : start + storage.mtp_step + 1].zero_()
            if storage.buffers.get("mtp_state_index") is not None:
                storage.buffers.get("mtp_state_index")[req_idx].fill_(0)

    def save(
        self,
        storage,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        assert isinstance(req_indices, list) and isinstance(
            page_slot_ids, list
        ), "request and slot indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = storage.pool(page_kind)
        conv_copies, ssm_copies = ([], [])
        for req_idx, slot_id in zip(req_indices, page_slot_ids):
            conv_copies.append(
                (storage.buffers["conv_state"][:, req_idx, :, : self.config.conv_width], pool.conv_state[slot_id])
            )
            ssm_copies.append(
                (storage.buffers["ssm_state"][:, req_idx * (storage.mtp_step + 1)], pool.ssm_state[slot_id])
            )
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)

    def restore(
        self,
        storage,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        """恢复各请求指定绝对 token 长度处的基准状态并归零游标；Conv/SSM 不依赖长度定位。"""
        assert isinstance(req_indices, list) and isinstance(
            page_slot_ids, list
        ), "request and slot indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = storage.pool(page_kind)
        conv_copies, ssm_copies = ([], [])
        for req_idx, slot_id in zip(req_indices, page_slot_ids):
            conv_copies.append(
                (pool.conv_state[slot_id], storage.buffers["conv_state"][:, req_idx, :, : self.config.conv_width])
            )
            ssm_copies.append(
                (pool.ssm_state[slot_id], storage.buffers["ssm_state"][:, req_idx * (storage.mtp_step + 1)])
            )
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)
        if storage.buffers.get("mtp_state_index") is not None:
            for req_idx in req_indices:
                storage.buffers.get("mtp_state_index")[req_idx].fill_(0)

    def commit(self, storage, req_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
        """用 CUDA 请求索引和 int32 接受步数选择 SSM 候选位置；步数包含基准输入步。"""
        assert req_indices.is_cuda and accepted_steps.is_cuda, "commit inputs must be CUDA tensors"
        if storage.buffers.get("mtp_state_index") is not None:
            storage.buffers.get("mtp_state_index")[req_indices] = accepted_steps - 1

    def dp_fields(self, storage, req: int):
        return self._fields(
            storage.buffers["conv_state"][:, req, :, : self.config.conv_width],
            storage.buffers["ssm_state"][:, req * (storage.mtp_step + 1)],
        )

    @staticmethod
    def _fields(conv: torch.Tensor, ssm: torch.Tensor):
        return {
            "conv": TensorField(tensor=conv, axes=("layer", "channel", "window")),
            "ssm": TensorField(tensor=ssm, axes=("layer", "head", "key_dim", "value_dim")),
        }

    def after_restore(self, storage, req_indices):
        cursor = storage.buffers.get("mtp_state_index")
        if cursor is not None:
            for req in req_indices:
                cursor[req].fill_(0)

    def check_dp_compatible(self, source, destination):
        pass
