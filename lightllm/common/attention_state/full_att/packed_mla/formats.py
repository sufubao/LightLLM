from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, ClassVar
import torch
from lightllm.common.attention_state.layouts import TokenField, PackedField, PackedTokenStorageSpec
from ...utils import dtype_name


@dataclass(frozen=True)
class PackedMLALogicalConfig:
    """带 indexer 的 packed MLA 逻辑配置；不描述实际字节布局。"""

    logical_layer_groups: tuple[tuple[int, ...], ...]
    # 编码前的完整 latent 宽度；具体分块、scale 和 padding 由工厂转换。
    latent_dim: int
    # packed MLA 的逻辑格式；所有选项都包含 indexer。
    format: Literal["packed_fp16", "packed_bf16", "packed_group128_fp8", "packed_group64_e8m0"]
    # indexer key 的逻辑宽度。
    indexer_dim: int = 128
    # latent 中保留原始 RoPE 精度的宽度；NoPE 宽度由 latent_dim - rope_dim 得到。
    rope_dim: int = 64
    tp_rank: int = 0
    tp_world_size: int = 1

    def __post_init__(self) -> None:
        assert self.logical_layer_groups and all(self.logical_layer_groups)
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert len(layers) == len(set(layers)), "logical layers cannot repeat across storage groups"
        assert (
            self.latent_dim > 0
            and 0 <= self.rope_dim <= self.latent_dim
            and self.indexer_dim > 0
            and 0 <= self.tp_rank < self.tp_world_size
        )
        assert self.format in (
            "packed_fp16",
            "packed_bf16",
            "packed_group128_fp8",
            "packed_group64_e8m0",
        ), "unknown packed MLA logical format"


@dataclass(frozen=True)
class PackedMLAAttentionConfig:
    """由 packed 逻辑配置解析出的物理布局，不重复保存逻辑字段。"""

    logical_config: PackedMLALogicalConfig
    quant_group_size: int = 128

    @property
    def logical_layer_groups(self):
        return self.logical_config.logical_layer_groups

    @property
    def latent_dim(self):
        return self.logical_config.latent_dim

    @property
    def kv_encoding(self):
        return {
            "packed_fp16": "fp16",
            "packed_bf16": "bf16",
            "packed_group128_fp8": "fp8_group128",
        }[self.logical_config.format]

    @property
    def indexer_dim(self):
        return self.logical_config.indexer_dim

    @property
    def tp_rank(self):
        return self.logical_config.tp_rank

    @property
    def tp_world_size(self):
        return self.logical_config.tp_world_size

    @property
    def rope_dim(self):
        return self.logical_config.rope_dim

    @property
    def dtype(self) -> torch.dtype:
        """返回编码对应的逻辑存储 dtype；FP8 的物理承载仍由 State 内部使用 uint8。"""
        if self.kv_encoding == "fp16":
            return torch.float16
        if self.kv_encoding == "bf16":
            return torch.bfloat16
        return torch.float8_e4m3fn

    @property
    def kv_bytes(self) -> int:
        """按逻辑 NoPE/RoPE 宽度和 scale 数量计算单行 KV 字节数。"""
        if self.kv_encoding == "fp8_group128":
            nope_dim = self.latent_dim - self.rope_dim
            scale_count = nope_dim // self.quant_group_size
            return nope_dim + scale_count * torch.float32.itemsize + self.rope_dim * torch.bfloat16.itemsize
        return self.latent_dim * self.dtype.itemsize

    @property
    def token_bytes(self) -> int:
        """KV、indexer key/scale 及整行 16 字节对齐所需的物理字节数。"""
        return (self.kv_bytes + self.indexer_dim + 4 + 15) // 16 * 16

    @property
    def indexer_offset(self) -> int:
        """保持既有格式：FP8 紧接 KV，浮点格式将对齐间隙放在 KV 与 indexer 之间。"""
        return self.kv_bytes if self.kv_encoding == "fp8_group128" else self.token_bytes - self.indexer_dim - 4

    def __post_init__(self) -> None:
        assert self.kv_encoding in ("fp16", "bf16", "fp8_group128"), "unknown packed MLA KV encoding"
        if self.kv_encoding == "fp8_group128":
            assert (
                self.dtype == torch.float8_e4m3fn
                and self.rope_dim == 64
                and self.latent_dim == 576
                and self.quant_group_size == 128
            ), "group128 packed MLA requires latent 576, rope 64 and group size 128"
        else:
            assert self.kv_encoding in ("fp16", "bf16")


@dataclass(frozen=True)
class FP8E8M0PackedMLAConfig:
    """固定 group64/E8M0 编码，只记录调用方能够选择的原子布局信息。"""

    # 外层为物理存储层，内层为共享该层全部 KV/indexer 的逻辑层号。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # 每个 token 的 indexer key 维度；其后另保存一个 FP32 scale，不包含在该维度中。
    indexer_dim: int = 128
    # 完整 latent 在各 TP rank 复制；rank 0 写出共享外部页，各 rank 均可读回。
    tp_rank: int = 0
    # 所属 TP 组大小，不增加物理 KV head，也不改变单 token 的存储字节数。
    tp_world_size: int = 1

    # 算子输入的完整 latent 维度，不是量化后的字节数，也不包含 indexer。
    LATENT_DIM: ClassVar[int] = 512
    # 固定编码：448 个 FP8 元素 + 64 个 BF16 元素 + 7 个 E8M0 scale + 1 字节对齐。
    # 末尾 64 维保留 BF16 不代表启用了 RoPE，GLM 使用 NoPE attention。
    KV_BYTES: ClassVar[int] = 584

    @property
    def payload_bytes(self) -> int:
        """一行有效数据的字节数：KV 编码、indexer key 及其 FP32 scale，不包含行尾 padding。"""
        return self.KV_BYTES + self.indexer_dim + torch.float32.itemsize

    @property
    def token_bytes(self) -> int:
        """每个 token 的物理行字节数，包含末尾 16 字节对齐 padding。"""
        return (self.payload_bytes + 15) // 16 * 16

    def __post_init__(self) -> None:
        assert self.logical_layer_groups and all(self.logical_layer_groups)
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert len(layers) == len(set(layers)), "logical layers cannot repeat across storage groups"
        assert self.indexer_dim > 0 and 0 <= self.tp_rank < self.tp_world_size


def packed_mla_spec(config):
    fp8 = config.kv_encoding == "fp8_group128"
    dtype = torch.uint8 if fp8 else config.dtype
    fields = (
        (
            PackedField(TokenField("kv_nope", torch.uint8, 512), 0),
            PackedField(TokenField("kv_scale", torch.float32, 4, "group"), 512),
            PackedField(TokenField("kv_rope", torch.bfloat16, 64), 528),
        )
        if fp8
        else (PackedField(TokenField("kv", config.dtype, config.latent_dim), 0),)
    )
    fields += (
        PackedField(TokenField("indexer_k", torch.uint8, config.indexer_dim), config.indexer_offset),
        PackedField(TokenField("indexer_scale", torch.float32, 1, "scale"), config.indexer_offset + config.indexer_dim),
    )
    return PackedTokenStorageSpec(
        encoding="mla_packed_" + (f"fp8_group{config.quant_group_size}" if fp8 else dtype_name(config.dtype)),
        logical_layer_groups=config.logical_layer_groups,
        global_head_num=1,
        head_num=1,
        tp_rank=config.tp_rank,
        tp_world_size=config.tp_world_size,
        fields=(TokenField("kv", dtype, config.token_bytes // dtype.itemsize),),
        payload_fields=fields,
    )


def fp8_e8m0_spec(config):
    return PackedTokenStorageSpec(
        encoding="mla_packed_fp8_group64_e8m0",
        logical_layer_groups=config.logical_layer_groups,
        global_head_num=1,
        head_num=1,
        tp_rank=config.tp_rank,
        tp_world_size=config.tp_world_size,
        fields=(TokenField("kv", torch.uint8, config.token_bytes),),
        payload_fields=(PackedField(TokenField("kv", torch.uint8, config.payload_bytes), 0),),
    )


from lightllm.common.attention_state.types import PackedMLALayerView as PackedMLALayerView
