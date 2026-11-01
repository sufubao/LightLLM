from __future__ import annotations
from dataclasses import dataclass
from math import prod
import torch
from lightllm.common.attention_state.types import AxisMapping, TransferRegion


@dataclass(frozen=True)
class TokenField:
    name: str
    dtype: torch.dtype
    width: int
    width_axis: str = "dim"

    def __post_init__(self):
        assert self.name and self.width > 0
        assert self.width_axis not in ("layer", "token", "head")

    @property
    def axes(self):
        return ("layer", "token", "head", self.width_axis)


@dataclass(frozen=True)
class TokenStorageSpec:
    encoding: str
    logical_layer_groups: tuple[tuple[int, ...], ...]
    global_head_num: int
    head_num: int
    tp_rank: int
    tp_world_size: int
    fields: tuple[TokenField, ...]
    block_tokens: int = 1
    zero_initialize: bool = False
    format_key: tuple[int, ...] = ()

    def __post_init__(self):
        assert self.block_tokens > 0
        assert self.fields and len({field.name for field in self.fields}) == len(self.fields)
        assert self.logical_layer_groups and all(self.logical_layer_groups)
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert len(layers) == len(set(layers)) and all(layer >= 0 for layer in layers)
        assert self.head_num > 0 and self.global_head_num > 0
        assert self.global_head_num % self.head_num == 0, "head fields require complete shards or replicas"
        assert 0 <= self.tp_rank < self.tp_world_size
        assert (self.head_num * self.tp_world_size) % self.global_head_num == 0

    @property
    def token_bytes(self) -> int:
        return (
            len(self.logical_layer_groups)
            * self.head_num
            * sum(field.width * field.dtype.itemsize for field in self.fields)
        )

    def shape(self, field: TokenField, token_slots: int) -> tuple[int, ...]:
        return (len(self.logical_layer_groups), token_slots, self.head_num, field.width)

    @property
    def transfer_fields(self) -> tuple[TokenField, ...]:
        return self.fields

    def page_views(self, buffers: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return buffers

    def transfer_regions(self, page_tokens: int) -> tuple[TransferRegion, ...]:
        assert (
            page_tokens > 0 and page_tokens % self.block_tokens == 0
        ), "CPU/PD page token capacity must be a positive multiple of block_token_size"
        replicas = self.tp_world_size * self.head_num // self.global_head_num
        head_start = self.tp_rank // replicas * self.head_num
        regions, offset = [], 0
        for field in self.transfer_fields:
            shape = (len(self.logical_layer_groups), page_tokens, self.global_head_num, field.width)
            offset = (offset + field.dtype.itemsize - 1) // field.dtype.itemsize * field.dtype.itemsize
            end = offset + prod(shape) * field.dtype.itemsize
            regions.append(
                TransferRegion(
                    field=field.name,
                    byte_start=offset,
                    byte_end=end,
                    dtype=field.dtype,
                    axes=field.axes,
                    shape=shape,
                    axis_mappings=(
                        AxisMapping("head", 0, head_start, self.head_num),
                        AxisMapping(field.width_axis, 0, 0, field.width),
                    ),
                    write_enabled=self.tp_rank % replicas == 0,
                )
            )
            offset = end
        return tuple(regions)


@dataclass(frozen=True)
class PackedField:
    field: TokenField
    byte_offset: int


@dataclass(frozen=True)
class PackedTokenStorageSpec(TokenStorageSpec):
    payload_fields: tuple[PackedField, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        assert len(self.fields) == 1 and self.payload_fields
        row_bytes = self.fields[0].width * self.fields[0].dtype.itemsize
        assert len({part.field.name for part in self.payload_fields}) == len(self.payload_fields)
        for part in self.payload_fields:
            assert part.byte_offset >= 0 and part.byte_offset % part.field.dtype.itemsize == 0
            assert part.byte_offset + part.field.width * part.field.dtype.itemsize <= row_bytes

    @property
    def transfer_fields(self):
        return tuple(part.field for part in self.payload_fields)

    def page_views(self, buffers):
        raw = buffers[self.fields[0].name].view(torch.uint8)
        return {
            part.field.name: raw[
                ..., part.byte_offset : part.byte_offset + part.field.width * part.field.dtype.itemsize
            ].view(part.field.dtype)
            for part in self.payload_fields
        }


@dataclass(frozen=True)
class RuntimeBuffer:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class SnapshotField:
    name: str
    attribute: str
    axes: tuple[str, ...]


@dataclass(frozen=True)
class BoundedStorageSpec:
    encoding: str
    logical_layers: tuple[int, ...]
    runtime_buffers: tuple[RuntimeBuffer, ...]
    snapshot_fields: tuple[SnapshotField, ...]
    cpu_regions: tuple[TransferRegion, ...]
    pd_regions: tuple[TransferRegion, ...]


def window_regions(config, name, window, width, dtype):
    if not window:
        return ()
    shape = (len(config.logical_layer_ids), window, width)
    return (
        TransferRegion(
            field=name,
            byte_start=0,
            byte_end=prod(shape) * dtype.itemsize,
            dtype=dtype,
            axes=("layer", "window", "channel"),
            shape=shape,
            axis_mappings=(),
            write_enabled=config.tp_rank == 0,
        ),
    )
