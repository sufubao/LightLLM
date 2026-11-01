from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from lightllm.common.attention_state.layouts import BoundedStorageSpec, RuntimeBuffer, SnapshotField, window_regions


@dataclass(frozen=True)
class WindowCompressorStateConfig:
    """按固定窗口和产出步长保存候选向量及 gate，供模型的压缩或池化算子使用。

    这里只描述运行态的存储和历史保留方式，具体聚合、归一化及量化由模型算子完成。
    window_size / compress_stride 分别表示窗口覆盖的原始 token 数和相邻产出的间隔：
    - GLM K-pool：W4/S4，每四个 key/gate 形成一组，无跨组重叠历史。
    - V4 C4：W8/S4，每四步产出一次，下一组仍需保留四个 token 的历史。
    - V4 C128：W128/S128，每 128 步产出一次，无跨组重叠历史。
    MTP 的额外候选槽位由 State.mtp_step 扩展，不计入窗口或产出步长。
    """

    # 本实例服务的逻辑层，顺序对应运行态 Tensor 的 layer 轴。
    # attention KV 候选与索引 key 候选分别创建实例，可使用不同窗口、维度和精度。
    logical_layer_ids: tuple[int, ...]
    # 单次聚合覆盖的原始 token 窗口，不包含 MTP 额外槽位。
    # 例如 K-pool 为 4，V4 C4 为 8；压缩比例同为 4 不代表窗口长度相同。
    window_size: int
    # 相邻产出之间经过的原始 token 数；K-pool / V4 C4 为 4，V4 C128 为 128。
    # 等于 window_size 时无重叠，小于时保留 window_size - compress_stride 个历史位置。
    # 大小页快照在该步长的边界保存；PD 可在任意前缀终点传输完整逻辑窗口。
    compress_stride: int
    # 每个原始 token 保存的候选向量宽度；gate 具有相同形状和存储精度。
    # K-pool 中等于 indexer key 的 head_dim；V4 W8/S4 为前后半窗口分别保存候选，
    # 使用 2 * 输出向量维度。因此它不一定等于最终压缩条目的维度，也不包含 gate 的宽度。
    candidate_dim: int
    # 候选数据的用途，仅用于区分规范 State 成员名，不选择聚合算法或改变存储和恢复行为。
    # kv 表示 attention KV 候选；indexer 表示索引 key 候选及其 gate，不代表整个 indexer。
    target: Literal["kv", "indexer"] = "kv"
    # 候选与 gate 使用相同精度，沿最后一轴拼接为 candidate|gate，共用一个 Tensor。
    # 此精度描述压缩器输入历史，不是产出 KV / indexer key 的量化编码；
    # GLM K-pool 随模型精度配置（通常 BF16），V4 的 KV/indexer 压缩器当前使用 FP32。
    # 当前接口不支持候选与 gate 使用不同 dtype。
    dtype: torch.dtype = torch.float32
    # 所属模型 TP 组内的 rank；运行态完整复制，仅 rank 0 向共享外部页写出。
    tp_rank: int = 0
    # 保存相同运行态副本的 TP rank 数，不用于切分 candidate_dim 或窗口长度。
    tp_world_size: int = 1

    @property
    def overlap_tokens(self) -> int:
        """组完成时仍需跨步保留的重叠历史：W8/S4 为 4，W128/S128 为 0。"""
        return self.window_size - self.compress_stride

    def __post_init__(self) -> None:
        assert self.window_size >= self.compress_stride > 0 and self.candidate_dim > 0
        assert self.logical_layer_ids and len(set(self.logical_layer_ids)) == len(self.logical_layer_ids)
        assert self.target in ("kv", "indexer") and self.dtype in (torch.float32, torch.float16, torch.bfloat16)
        assert 0 <= self.tp_rank < self.tp_world_size


def compressor_spec(config, request_capacity, mtp_step):
    precision = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[config.dtype]
    buffers = (
        RuntimeBuffer(
            "buffer",
            (len(config.logical_layer_ids), request_capacity, config.window_size + mtp_step, 2 * config.candidate_dim),
            config.dtype,
        ),
    )
    fields = (SnapshotField("compressor", "buffer", ("layer", "window", "channel")),) if config.overlap_tokens else ()
    return BoundedStorageSpec(
        f"compressor_{config.target}_w{config.window_size}_s{config.compress_stride}_{precision}",
        config.logical_layer_ids,
        buffers,
        fields,
        window_regions(config, "compressor", config.overlap_tokens, 2 * config.candidate_dim, config.dtype),
        window_regions(config, "compressor", config.window_size, 2 * config.candidate_dim, config.dtype),
    )
