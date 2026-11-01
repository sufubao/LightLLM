from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from lightllm.common.attention_state.state_ops.base import KVWriter

if TYPE_CHECKING:
    from lightllm.common.attention_state.types import PackedMLALayerView
    from lightllm.common.attention_state.types import CompressedAttentionLayerView


class MLAKVWriter(KVWriter):
    def layer_view(self, storage, logical_layer_id: int) -> torch.Tensor:
        """按逻辑层号返回压缩 latent 视图，不拆分为 GQA 的 K/V head。"""
        return storage.buffers["kv"][storage.layer_index(logical_layer_id=logical_layer_id)]

    def write(self, storage, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """通过 CUDA 算子将形状匹配的 MLA latent 数据写入指定物理 token 位置。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        storage_layer_index = storage.layer_index(logical_layer_id=logical_layer_id)
        if not len(token_indices):
            return
        assert storage.buffers["kv"].is_cuda, "KV writes require CUDA storage"
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        destindex_copy_kv(K=kv, DestLoc=token_indices, Out=storage.buffers["kv"][storage_layer_index])


class PackedMLAKVWriter(KVWriter):
    def layer_view(self, storage, logical_layer_id: int) -> PackedMLALayerView:
        """分别返回 KV 与 packed indexer key/scale 的视图，保持原行步长。"""
        from lightllm.common.attention_state.types import PackedMLALayerView

        row = storage.buffers["kv"][storage.layer_index(logical_layer_id=logical_layer_id)]
        raw = row.view(torch.uint8)
        kv = (
            raw[..., : self.config.kv_bytes]
            if self.config.kv_encoding == "fp8_group128"
            else row[..., : self.config.latent_dim]
        )
        indexer = raw[..., self.config.indexer_offset : self.config.indexer_offset + self.config.indexer_dim + 4]
        return PackedMLALayerView(kv=kv, indexer_k=indexer)

    def write(self, storage, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """写入 MLA KV 及其量化参数，保留同一 token 行中的 indexer 区域。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        storage_layer_index = storage.layer_index(logical_layer_id=logical_layer_id)
        if not len(token_indices):
            return
        assert storage.buffers["kv"].is_cuda, "KV writes require CUDA storage"
        if self.config.kv_encoding == "fp8_group128":
            from lightllm.common.basemodel.triton_kernel.kv_copy.destindex_copy_kv_flashmla_fp8 import (
                destindex_copy_kv_flashmla_fp8,
            )

            raw = storage.buffers["kv"][storage_layer_index]
            nope_dim = self.config.latent_dim - self.config.rope_dim
            scale_bytes = nope_dim // self.config.quant_group_size * torch.float32.itemsize
            rope_start = nope_dim + scale_bytes
            destindex_copy_kv_flashmla_fp8(
                KV_nope=kv[..., :nope_dim],
                KV_rope=kv[..., nope_dim:],
                DestLoc=token_indices,
                O_nope=raw[..., :nope_dim].view(torch.float8_e4m3fn),
                O_scale=raw[..., nope_dim:rope_start].view(torch.float32),
                O_rope=raw[..., rope_start : rope_start + self.config.rope_dim * torch.bfloat16.itemsize].view(
                    torch.bfloat16
                ),
            )
        else:
            from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

            destindex_copy_kv(
                K=kv,
                DestLoc=token_indices,
                Out=storage.buffers["kv"][storage_layer_index, ..., : self.config.latent_dim],
            )


class FP8E8M0KVWriter(KVWriter):
    def layer_view(self, storage, logical_layer_id: int) -> PackedMLALayerView:
        """返回固定 584 字节 KV 和 indexer key/scale 视图，保留物理行 stride。"""
        from lightllm.common.attention_state.types import PackedMLALayerView

        row = storage.buffers["kv"][storage.layer_index(logical_layer_id=logical_layer_id)]
        return PackedMLALayerView(
            kv=row[..., : self.config.KV_BYTES], indexer_k=row[..., self.config.KV_BYTES : self.config.payload_bytes]
        )

    def write(self, storage, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """使用固定 group64/E8M0 算子量化并写入 KV，保留 indexer 和行尾 padding。"""
        from lightllm.common.attention_state.kernels.fp8_e8m0_write import write_fp8_e8m0_packed_kv

        assert token_indices.is_cuda, "token indices must reside on CUDA"
        if token_indices.numel() == 0:
            return
        write_fp8_e8m0_packed_kv(
            kv=kv, token_indices=token_indices, packed_kv=self.layer_view(storage, logical_layer_id=logical_layer_id).kv
        )


class CompressedKVWriter(KVWriter):
    def layer_view(self, storage, logical_layer_id: int) -> CompressedAttentionLayerView:
        """从同一物理压缩块取 KV 和 indexer 视图，不重排数据或创建副本。"""
        from lightllm.common.attention_state.types import CompressedAttentionLayerView

        config = self.config
        layer = storage.buffers["kv"][storage.layer_index(logical_layer_id=logical_layer_id)]
        blocks = layer.view(-1, config.block_token_size * config.token_slot_bytes)
        entries = config.block_token_size // config.compress_ratio
        kv_bytes = config.kv_block_bytes
        indexer = (
            blocks[:, kv_bytes : kv_bytes + entries * (config.indexer_dim + torch.float32.itemsize)]
            if config.indexer_dim
            else None
        )
        return CompressedAttentionLayerView(kv=blocks[:, :kv_bytes], indexer_k=indexer)

    def write(self, storage, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """压缩器通过计算视图融合写入；普通 KV 写入路径不应调用此方法。"""
        raise NotImplementedError("compressed KV is written by the fused compressor")
