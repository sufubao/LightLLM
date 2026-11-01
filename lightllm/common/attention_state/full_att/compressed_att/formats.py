from __future__ import annotations
from dataclasses import dataclass
from math import lcm
import torch
from lightllm.common.attention_state.layouts import TokenField, TokenStorageSpec


@dataclass(frozen=True)
class CompressedAttentionConfig:
    # 外层顺序是物理 layer 轴；内层列出共享整份压缩 KV/indexer 存储的逻辑层。
    # 例如 ((0, 4), (8,)) 分配两份物理层存储，逻辑层 0/4 访问同一份数据。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # 以原始 token 为单位的压缩比例，不改变外部请求映射的索引域。
    compress_ratio: int
    # 每个压缩 KV 条目编码前的向量维度，包含末尾的 RoPE 部分；不是存储字节数。
    kv_dim: int
    # 保留 BF16 的 RoPE 维度；其余维度按每 64 个元素一组编码为 FP8，scale 使用 E8M0。
    rope_dim: int
    # 每个物理块覆盖的原始 token 数，必须能被 compress_ratio 整除。
    # 块内先保存全部 KV 数据，再保存 scale 和可选 indexer；不能按单个压缩条目拆开搬运。
    block_token_size: int
    # 可选 indexer key 的 FP8 维度，每个 key 另有一个 FP32 scale；0 表示不保存 indexer。
    indexer_dim: int = 0
    # 当前 State 所属 TP 组内的 rank；当前实现是单个共享 KV head，在各 rank 完整复制。
    # rank 0 写出共享 CPU/PD 页，每个 rank 均可读回，不按 TP 切分压缩条目的向量。
    tp_rank: int = 0
    # 当前模型 TP 组的 rank 数，只描述复制关系，不影响单个 rank 的存储字节数。
    tp_world_size: int = 1

    @property
    def kv_entry_bytes(self) -> int:
        """单条压缩 KV 的数据与 scale 字节数，不包含块末尾 padding 或 indexer。"""
        scale_count = (self.kv_dim - self.rope_dim) // 64
        return self.kv_dim + self.rope_dim + (scale_count + 7) // 8 * 8

    @property
    def kv_block_bytes(self) -> int:
        """块内 KV 数据及 E8M0 scale 区的字节跨度，包含数据行对齐，不包含 indexer。"""
        entries = self.block_token_size // self.compress_ratio
        # FP8 占一字节，BF16 RoPE 占两字节；每条目的 scale 区补齐到八字节。
        data_bytes = self.kv_dim + self.rope_dim
        return (entries * self.kv_entry_bytes + data_bytes - 1) // data_bytes * data_bytes

    @property
    def token_slot_bytes(self) -> int:
        """每个原始 token 索引摊销的物理字节数，包含 KV、scale、indexer 和块内 padding。"""
        block_bytes = self.kv_block_bytes
        if self.indexer_dim:
            entries = self.block_token_size // self.compress_ratio
            block_bytes += entries * (self.indexer_dim + torch.float32.itemsize)
        # 块既要能均摊到原始 token 索引域，也要保留 packed KV 数据行的对齐。
        alignment = lcm(self.block_token_size, self.kv_dim + self.rope_dim)
        return (block_bytes + alignment - 1) // alignment * alignment // self.block_token_size

    def get_attention_view(self, packed_kv: torch.Tensor) -> torch.Tensor:
        """返回算子使用的 [block, entry, head, byte] 视图，保留数据/scale 分区和块 stride。"""
        entries = self.block_token_size // self.compress_ratio
        return packed_kv[:, : entries * self.kv_entry_bytes].view(-1, entries, 1, self.kv_entry_bytes)

    def __post_init__(self) -> None:
        assert (
            self.compress_ratio > 0 and self.block_token_size > 0 and self.block_token_size % self.compress_ratio == 0
        )
        assert self.kv_dim > self.rope_dim >= 0 and (self.kv_dim - self.rope_dim) % 64 == 0
        assert self.indexer_dim >= 0 and 0 <= self.tp_rank < self.tp_world_size
        assert self.logical_layer_groups and all(self.logical_layer_groups), "storage layer groups must be nonempty"
        logical_layer_ids = [layer for group in self.logical_layer_groups for layer in group]
        assert len(logical_layer_ids) == len(
            set(logical_layer_ids)
        ), "logical layers cannot repeat across storage groups"


def compressed_spec(config):
    return TokenStorageSpec(
        encoding=f"compressed_c{config.compress_ratio}_fp8_group64_e8m0",
        logical_layer_groups=config.logical_layer_groups,
        global_head_num=1,
        head_num=1,
        tp_rank=config.tp_rank,
        tp_world_size=config.tp_world_size,
        fields=(TokenField("kv", torch.uint8, config.token_slot_bytes),),
        block_tokens=config.block_token_size,
        zero_initialize=True,
        format_key=(config.compress_ratio, config.kv_dim, config.rope_dim, config.indexer_dim),
    )


from lightllm.common.attention_state.types import CompressedAttentionLayerView as CompressedAttentionLayerView
