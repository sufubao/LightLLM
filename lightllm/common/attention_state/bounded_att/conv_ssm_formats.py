from __future__ import annotations
from math import prod
import torch
from lightllm.common.attention_state.layouts import BoundedStorageSpec, RuntimeBuffer, SnapshotField
from lightllm.common.attention_state.types import AxisMapping, TransferRegion
from lightllm.common.attention_state.utils import dtype_name


def conv_ssm_regions(config) -> tuple[TransferRegion, ...]:
    """依次描述全局 Conv 和 SSM 区域，并标明当前 TP rank 的读写范围。"""
    storage_layer_count = len(config.logical_layer_ids)
    conv_channels = 2 * config.global_k_heads * config.key_dim + config.global_v_heads * config.value_dim
    conv_shape = (storage_layer_count, conv_channels, config.conv_width)
    conv_byte_end = prod(conv_shape) * config.conv_dtype.itemsize
    regions = []
    local_channel_start = 0
    global_channel_start = 0
    for name, local_head_count, global_head_count, head_dim in (
        ("Q", config.num_k_heads, config.global_k_heads, config.key_dim),
        ("K", config.num_k_heads, config.global_k_heads, config.key_dim),
        ("V", config.num_v_heads, config.global_v_heads, config.value_dim),
    ):
        replica_count = config.tp_world_size * local_head_count // global_head_count
        head_start = config.tp_rank // replica_count * local_head_count
        write_enabled = config.tp_rank % replica_count == 0
        local_channel_count = local_head_count * head_dim
        target_channel_start = global_channel_start + head_start * head_dim
        conv_region = TransferRegion(
            field="conv",
            byte_start=0,
            byte_end=conv_byte_end,
            dtype=config.conv_dtype,
            axes=("layer", "channel", "window"),
            shape=conv_shape,
            axis_mappings=(
                AxisMapping(
                    axis="channel",
                    local_start=local_channel_start,
                    external_start=target_channel_start,
                    length=local_channel_count,
                ),
                AxisMapping(axis="window", local_start=0, external_start=0, length=config.conv_width),
            ),
            write_enabled=write_enabled,
        )
        regions.append(conv_region)
        local_channel_start += local_channel_count
        global_channel_start += global_head_count * head_dim
    replica_count = config.tp_world_size * config.num_v_heads // config.global_v_heads
    head_start = config.tp_rank // replica_count * config.num_v_heads
    write_enabled = config.tp_rank % replica_count == 0
    ssm_shape = (storage_layer_count, config.global_v_heads, config.key_dim, config.value_dim)
    element_bytes = config.ssm_dtype.itemsize
    ssm_byte_start = (conv_byte_end + element_bytes - 1) // element_bytes * element_bytes
    ssm_region = TransferRegion(
        field="ssm",
        byte_start=ssm_byte_start,
        byte_end=ssm_byte_start + prod(ssm_shape) * element_bytes,
        dtype=config.ssm_dtype,
        axes=("layer", "head", "key_dim", "value_dim"),
        shape=ssm_shape,
        axis_mappings=(
            AxisMapping(axis="head", local_start=0, external_start=head_start, length=config.num_v_heads),
            AxisMapping(axis="key_dim", local_start=0, external_start=0, length=config.key_dim),
            AxisMapping(axis="value_dim", local_start=0, external_start=0, length=config.value_dim),
        ),
        write_enabled=write_enabled,
    )
    regions.append(ssm_region)
    return tuple(regions)


def conv_ssm_spec(config, request_capacity, mtp_step, algorithm):
    layers = len(config.logical_layer_ids)
    buffers = (
        RuntimeBuffer(
            "conv_state",
            (layers, request_capacity, config.conv_channels, config.conv_width + mtp_step),
            config.conv_dtype,
        ),
        RuntimeBuffer(
            "ssm_state",
            (layers, request_capacity * (mtp_step + 1), config.num_v_heads, config.key_dim, config.value_dim),
            config.ssm_dtype,
        ),
    )
    if mtp_step:
        buffers += (RuntimeBuffer("mtp_state_index", (request_capacity,), torch.int32),)
    fields = (
        SnapshotField("conv", "conv_state", ("layer", "channel", "window")),
        SnapshotField("ssm", "ssm_state", ("layer", "head", "key_dim", "value_dim")),
    )
    regions = conv_ssm_regions(config)
    return BoundedStorageSpec(
        f"{algorithm}_conv_{dtype_name(config.conv_dtype)}_ssm_{dtype_name(config.ssm_dtype)}",
        config.logical_layer_ids,
        buffers,
        fields,
        regions,
        regions,
    )
