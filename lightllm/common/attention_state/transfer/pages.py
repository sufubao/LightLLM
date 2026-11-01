import torch
import triton
from lightllm.common.attention_state.types import StateTransferLayout
from lightllm.common.attention_state.tensor_views import merge_contiguous_tail_dims, view_max_copy_dtype
from lightllm.common.attention_state.kernels.token_pages import _copy_token_pages


def transfer_token_pages(
    fields: dict[str, torch.Tensor],
    field_axes: dict[str, tuple[str, ...]],
    token_indices: torch.Tensor,
    page_buffers: torch.Tensor,
    page_indices: list[int],
    layout: StateTransferLayout,
    write: bool,
    grid_num: int = 24,
    page_readies: list[bool] | None = None,
    num_warps: int = 8,
    num_stages: int = 1,
) -> None:
    """直接搬运 K/V；CPU 传入页列表，PD 传入 [pd_page_index]，同步由调用方负责。

    外部页为 pinned/注册 CPU 内存或 CUDA PD buffer；跨卡指针由调用方提前建立 peer/IPC 映射。
    单页和多页统一通过页索引 Tensor 定位，共用布局和拷贝 kernel。
    grid_num 控制每个字段发射的 program 数，默认 24；kernel 跳步循环覆盖全部搬运单元。
    num_warps 控制每个 program 的 warp 数，num_stages 控制循环级流水深度。
    """
    assert grid_num > 0, "grid_num must be positive"
    assert fields and all(t.is_cuda for t in fields.values()), "page transfers require CUDA State storage"
    assert page_buffers.ndim >= 2, "page buffers must include a leading page dimension"
    assert isinstance(page_indices, list), "page indices must be a list"
    if page_buffers.device.type == "cpu":
        assert page_buffers.is_pinned(), "CPU pages must be pinned or registered"
    page_tokens = layout.tokens_per_page
    assert page_tokens is not None and page_tokens > 0, "KV transfers require a positive token capacity"
    assert token_indices.is_cuda, "token indices must reside on CUDA"
    token_count = token_indices.numel()
    assert len(page_indices) == triton.cdiv(x=token_count, y=page_tokens), "KV token and page counts differ"
    if write and page_readies is not None:
        assert len(page_readies) == len(page_indices), "page and ready counts differ"
        if all(page_readies):
            return
    if token_count == 0:
        return

    # 阶段一：准备索引，不预选或复制 K/V。
    # token_indices[i] 是第 i 个 token 的 GPU 物理槽位，直接使用传入的 dtype 和 stride。
    # 跨 GPU 的 KV 由传输进程通过现有 IPC peer 映射导入当前设备，kernel 沿当前流访问。
    # page_ids[j] 是第 j 个逻辑页对应的外部物理页 ID，允许不连续、不递增。
    # CPU/PD、单页/多页统一通过 pinned 索引异步提交到 GPU，临时显存只与页数有关。
    page_ids = torch.tensor(data=page_indices, dtype=torch.int64, pin_memory=True).to(
        device=torch.cuda.current_device(),
        non_blocking=True,
    )

    # ready 标记按逻辑页排列，保留完整 token/page 索引，避免跳过中间页后错位。
    # 本 State 的全部字段共享此 Tensor；PD 和读回路径不需要 ready 标记。
    page_readies_tensor = None
    if write and page_readies is not None:
        page_readies_tensor = torch.tensor(data=page_readies, dtype=torch.bool, pin_memory=True).to(
            device=torch.cuda.current_device(),
            non_blocking=True,
        )

    # 阶段二：构造全部字段的视图，完成后再统一提交 kernel。
    # page_bytes 是 [物理页数, 每页可见字节数]，第一维仍保留真实页跨度。
    # 一个联合页可能还保存其他 State 的字段，不能用当前 K/V 大小代替页跨度。
    page_bytes = page_buffers.view(torch.uint8).view(page_buffers.shape[0], -1)
    copy_views = []
    for region in layout.regions:
        if write and not region.write_enabled:
            continue
        assert region.field in fields and region.field in field_axes, f"unknown transfer field {region.field}"
        local = fields[region.field]
        axes = field_axes[region.field]
        # 从各页中取字段的字节区间，按 dtype 和布局形状解释，不搬运数据。
        external = page_bytes[:, region.byte_start : region.byte_end].view(region.dtype)
        external = external.view(page_buffers.shape[0], *region.shape)
        # 外部统一为 [page, layer, page_token, head, field_width]；+1 是因为多了 page 轴。
        external = external.permute(0, *(region.axes.index(axis) + 1 for axis in axes))
        local_slices = [slice(None)] * 4
        external_slices = [slice(None)] * 5
        for axis_mapping in region.axis_mappings:
            # local_start/external_start 固定表示本地/外部起点，不随读写方向改变。
            # 例：本地 heads [0, 2) 对应外部 heads [4, 6)，切片保留两侧真实 stride。
            axis = axes.index(axis_mapping.axis)
            local_slices[axis] = slice(axis_mapping.local_start, axis_mapping.local_start + axis_mapping.length)
            external_slices[axis + 1] = slice(
                axis_mapping.external_start, axis_mapping.external_start + axis_mapping.length
            )
        local = local[tuple(local_slices)]
        external = external[tuple(external_slices)]
        # token 轴不比较：local 是 GPU 槽位容量，external 是单页容量。
        assert local.dtype == external.dtype
        assert (local.shape[0], local.shape[2], local.shape[3]) == (
            external.shape[1],
            external.shape[3],
            external.shape[4],
        ), "local and external copy views must have the same shape"
        copy_views.append((local, external))

    # 阶段三：每个传输字段一次 launch，覆盖全部层、有效 token 和外部页。
    # elements 是转换后搬运单元数，不是字节数；每个字段固定发射 grid_num 个 program。
    # 超出一轮 grid 覆盖范围的数据由 kernel 跳步处理，没有 gather/scatter 中间存储。
    # stream/event 同步由调用方负责，函数返回不表示 CPU 页已经可读取或复用。
    # TODO：当前 CPU/PD 已直接按索引搬运，可进一步将 K/V 及动态 scale 的逐字段 launch 融合，
    # 共用 token/page 索引；保留各字段 dtype、stride、TP 切片、write_enabled 和页内 padding。
    for local, external in copy_views:
        # 先合并连续的 head/width，保留 layer/token/page；不能合并时沿用原视图。
        # head 轴保留为 1，继续使用原来的四维/五维 kernel 接口，再选择最大搬运 dtype。
        local, external = merge_contiguous_tail_dims(
            copies=[(local, external)],
            start_dim=-2,
            keepdim=True,
        )[0]
        local, external = view_max_copy_dtype(copies=[(local, external)])[0]
        elements = local.shape[0] * token_count * local.shape[2] * local.shape[3]
        _copy_token_pages[(grid_num,)](
            LOCAL=local,
            EXTERNAL=external,
            TOKEN_IDS=token_indices,
            TOKEN_STRIDE=token_indices.stride(0),
            PAGE_IDS=page_ids,
            PAGE_READIES=page_readies_tensor,
            HAS_PAGE_READIES=page_readies_tensor is not None,
            token_count=token_count,
            element_count=elements,
            LOCAL_STRIDES=local.stride(),
            EXTERNAL_STRIDES=external.stride(),
            PAGE_TOKENS=page_tokens,
            HEADS=local.shape[2],
            WIDTH=local.shape[3],
            ELEMENT_BYTES=local.element_size(),
            WRITE=write,
            BLOCK=1024,
            NUM_STAGES=num_stages,
            num_warps=num_warps,
        )
