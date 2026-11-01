from __future__ import annotations
from dataclasses import dataclass
import torch
from lightllm.common.attention_state.layouts import BoundedStorageSpec, RuntimeBuffer, SnapshotField, window_regions


@dataclass(frozen=True)
class FP8E8M0SlidingWindowConfig:
    # 本 State 服务的逻辑层，顺序对应物理 layer 轴，各层拥有独立窗口。
    logical_layer_ids: tuple[int, ...]
    # 每个 query 可见的近期 token 数，也是大小页和 PD 快照的逻辑窗口长度。
    window_size: int
    # 未编码 KV 向量维度，包含末尾保留 BF16 的 RoPE 部分。
    kv_dim: int
    # RoPE 维度；其余维度按 group64 FP8 编码，scale 使用 E8M0。
    rope_dim: int
    # packed 物理页内的 token 数；页内 KV 数据与 scale 分区存放。
    page_token_size: int = 128
    # 各 TP rank 完整复制 KV；仅 rank 0 写出共享 CPU/PD 区域。
    tp_rank: int = 0
    tp_world_size: int = 1

    @property
    def kv_data_bytes(self) -> int:
        return self.kv_dim + self.rope_dim

    @property
    def kv_bytes(self) -> int:
        scales = (self.kv_dim - self.rope_dim) // 64
        return self.kv_data_bytes + (scales + 7) // 8 * 8

    def get_packed_kv_bytes(self, token_count: int) -> int:
        """返回 packed 页字节数，包含数据行对齐。"""
        return (token_count * self.kv_bytes + self.kv_data_bytes - 1) // self.kv_data_bytes * self.kv_data_bytes

    def get_attention_view(self, packed_kv: torch.Tensor) -> torch.Tensor:
        """将运行态或 workspace 解释为 [page, token, head, byte]，不复制或重排数据。"""
        pages = packed_kv.view(-1, self.get_packed_kv_bytes(self.page_token_size))
        return pages[:, : self.page_token_size * self.kv_bytes].view(-1, self.page_token_size, 1, self.kv_bytes)

    def __post_init__(self) -> None:
        assert self.window_size > 0 and self.page_token_size > 0
        assert self.kv_dim > self.rope_dim >= 0 and (self.kv_dim - self.rope_dim) % 64 == 0
        assert self.logical_layer_ids and len(set(self.logical_layer_ids)) == len(self.logical_layer_ids)
        assert 0 <= self.tp_rank < self.tp_world_size
        assert self.get_packed_kv_bytes(self.page_token_size) % self.page_token_size == 0


def sliding_window_spec(config, request_capacity, mtp_step):
    page_tokens = config.page_token_size
    capacity = (config.window_size + 2 * mtp_step + page_tokens - 1) // page_tokens * page_tokens
    buffers = (
        RuntimeBuffer(
            "swa_buffer",
            (
                len(config.logical_layer_ids),
                request_capacity,
                capacity,
                config.get_packed_kv_bytes(page_tokens) // page_tokens,
            ),
            torch.uint8,
        ),
    )
    regions = window_regions(config, "swa", config.window_size, config.kv_bytes, torch.uint8)
    return BoundedStorageSpec(
        f"swa_w{config.window_size}_fp8_group64_e8m0",
        config.logical_layer_ids,
        buffers,
        (SnapshotField("swa", "buffer", ("layer", "window", "channel")),),
        regions,
        regions,
    )
