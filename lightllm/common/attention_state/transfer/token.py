from __future__ import annotations

import torch

from lightllm.common.attention_state.tensor_views import merge_contiguous_tail_dims, view_max_copy_dtype
from lightllm.common.attention_state.kernels.token_copy import _copy_indexed_token_storages

from lightllm.common.attention_state.storage.token import TokenStorage


class TokenTransfer:
    def __init__(self, storage: TokenStorage):
        self.storage = storage
        self.field_axes = {field.name: field.axes for field in storage.spec.transfer_fields}

    def copy_tokens(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        assert src_indices.is_cuda and dst_indices.is_cuda, "token indices must reside on CUDA"
        assert (
            src_indices.ndim == dst_indices.ndim == 1 and src_indices.numel() == dst_indices.numel()
        ), "token copies require equal one-dimensional index tensors"
        assert src_indices.dtype in (torch.int32, torch.int64) and dst_indices.dtype in (
            torch.int32,
            torch.int64,
        ), "token indices must be integer tensors"
        assert self.storage.buffers and all(
            buffer.is_cuda for buffer in self.storage.buffers.values()
        ), "token copies require CUDA storage"
        assert (
            src_indices.numel() % self.storage.spec.block_tokens == 0
        ), "token copies require complete compressed blocks"
        if not src_indices.numel():
            return
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        for buffer in self.storage.buffers.values():
            copy_kv_buffer_to_kv_buffer(src_mem_index=src_indices, dst_mem_index=dst_indices, kv_buffer=buffer)

    def copy_from_peer(self, source: TokenStorage, src_indices, dst_indices) -> None:
        assert source.spec == self.storage.spec, "DP copy requires equivalent storage specifications"
        assert src_indices.is_cuda and dst_indices.is_cuda, "DP token indices must reside on CUDA"
        assert src_indices.numel() == dst_indices.numel(), "source and destination token counts differ"
        assert (
            src_indices.numel() % self.storage.spec.block_tokens == 0
        ), "DP token copies require complete compressed blocks"

        copy_token_storages(
            src_storages=tuple(source.buffers[field.name] for field in source.spec.fields),
            dst_storages=tuple(self.storage.buffers[field.name] for field in self.storage.spec.fields),
            src_token_indices=src_indices,
            dst_token_indices=dst_indices,
        )

    def pages(self, token_indices, page_buffers, page_indices, layout, write, page_readies=None) -> None:
        from lightllm.common.attention_state.transfer.pages import transfer_token_pages

        transfer_token_pages(
            fields=self.storage.spec.page_views(self.storage.buffers),
            field_axes=self.field_axes,
            token_indices=token_indices,
            page_buffers=page_buffers,
            page_indices=page_indices,
            layout=layout,
            write=write,
            page_readies=page_readies,
        )


class BlockTokenTransfer(TokenTransfer):
    def pages(self, token_indices, page_buffers, page_indices, layout, write, page_readies=None):
        alignment = self.storage.spec.block_tokens
        if page_buffers.device.type == "cpu":
            page_tokens = layout.tokens_per_page
            assert (
                page_tokens is not None and page_tokens > 0 and page_tokens % alignment == 0
            ), "CPU page token capacity must be a positive multiple of block_token_size"
            from math import prod

            page_bytes = prod(page_buffers.shape[1:]) * page_buffers.element_size()
            required_bytes = self.storage.spec.token_bytes * page_tokens
            assert (
                0 <= layout.byte_start < layout.byte_start + required_bytes <= layout.byte_end <= page_bytes
            ), "compressed State byte range must fit within one CPU page"
            assert all(0 <= page < page_buffers.shape[0] for page in page_indices), "CPU page index is out of bounds"
        else:
            padding = (-token_indices.numel()) % alignment
            if token_indices.numel() and padding:
                tail = token_indices[-1:] + torch.arange(
                    start=1, end=padding + 1, dtype=token_indices.dtype, device=token_indices.device
                )
                token_indices = torch.cat(tensors=(token_indices, tail))
        super().pages(token_indices, page_buffers, page_indices, layout, write, page_readies)


def copy_token_storages(
    src_storages: tuple[torch.Tensor, ...],
    dst_storages: tuple[torch.Tensor, ...],
    src_token_indices: torch.Tensor,
    dst_token_indices: torch.Tensor,
) -> None:
    """共享的四维 token 布局搬运，不创建 payload 或索引的中间 Tensor。

    在目标设备当前流发射；调用方保证远端存储及索引已可通过 peer/IPC 访问，
    并负责源数据就绪、存储生命周期和完成事件。目标 token 不重复，源目标区域不重叠。
    """
    token_count = src_token_indices.numel()
    if token_count == 0:
        return

    # 阶段一：先合并两侧连续的 head/width，保留各自 layer/token 容量和 stride。
    # head 轴保留为 1，沿用四维 token kernel；不连续时保持原布局。
    # 再按 8/4/2 字节搬运，保留 FP8、NaN、动态 scale 等原始位模式。
    sources, destinations = [], []
    for source, destination in zip(src_storages, dst_storages):
        source, destination = merge_contiguous_tail_dims(
            copies=[(source, destination)],
            start_dim=-2,
            keepdim=True,
        )[0]
        source, destination = view_max_copy_dtype(copies=[(source, destination)])[0]
        sources.append(source)
        destinations.append(destination)

    # 阶段二：Tensor tuple 在编译时展开为独立指针参数，不上传指针表。
    # 一次 launch 完成当前方案全部字段；每个 program 用跳步循环覆盖任意 token 数。
    # 保持 24 个 program，使用 4096 个搬运单元提高跨卡读取吞吐，循环流水深度为 1。
    _copy_indexed_token_storages[(24,)](
        SOURCES=tuple(sources),
        DESTINATIONS=tuple(destinations),
        SRC_TOKEN_IDS=src_token_indices,
        DST_TOKEN_IDS=dst_token_indices,
        token_count=token_count,
        SRC_INDEX_STRIDE=src_token_indices.stride(0),
        DST_INDEX_STRIDE=dst_token_indices.stride(0),
        SHAPES=tuple((tensor.shape[0], tensor.shape[2], tensor.shape[3]) for tensor in sources),
        SRC_STRIDES=tuple(tensor.stride() for tensor in sources),
        DST_STRIDES=tuple(tensor.stride() for tensor in destinations),
        ELEMENT_BYTES=tuple(tensor.element_size() for tensor in sources),
        BLOCK_ELEMENTS=4096,
        NUM_STAGES=1,
        num_warps=8,
    )
