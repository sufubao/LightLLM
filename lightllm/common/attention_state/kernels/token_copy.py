import triton
import triton.language as tl


@triton.jit
def _copy_indexed_token_storages(
    # 各字段的四维存储指针；K/V、scale 或 packed 行在一次 launch 中处理。
    SOURCES,
    DESTINATIONS,
    SRC_TOKEN_IDS,
    DST_TOKEN_IDS,
    token_count,
    # 索引 Tensor 可以非连续，直接使用调用方提供的 int32/int64 索引。
    SRC_INDEX_STRIDE: tl.constexpr,
    DST_INDEX_STRIDE: tl.constexpr,
    SHAPES: tl.constexpr,  # 各字段的 (layer_count, head_count, width)。
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    token_count = tl.cast(token_count, tl.int64)
    program_start = tl.program_id(0).to(tl.int64) * BLOCK_ELEMENTS
    grid_stride = tl.num_programs(0).to(tl.int64) * BLOCK_ELEMENTS
    lane = tl.arange(0, BLOCK_ELEMENTS).to(tl.int64)

    for field in tl.static_range(len(SHAPES)):
        # 每个字段独立计算地址；不同容量下 layer stride 不同，不能复用源端 stride。
        layers = SHAPES[field][0]
        heads = SHAPES[field][1]
        width = SHAPES[field][2]
        if ELEMENT_BYTES[field] == 8:
            copy_dtype = tl.uint64
        elif ELEMENT_BYTES[field] == 4:
            copy_dtype = tl.uint32
        elif ELEMENT_BYTES[field] == 2:
            copy_dtype = tl.uint16
        else:
            copy_dtype = tl.uint8
        source = SOURCES[field].to(tl.pointer_type(copy_dtype))
        destination = DESTINATIONS[field].to(tl.pointer_type(copy_dtype))
        element_count = token_count * layers * heads * width
        for block_start in tl.range(program_start, element_count, grid_stride, num_stages=NUM_STAGES):
            # 阶段一：按 [token, layer, head, width] 展开；除数均为布局常量，
            # 避免按动态 token_count 做逐元素的 int64 除法。
            element = block_start + lane
            valid = element < element_count
            column = element % width
            head = element // width % heads
            layer = element // (width * heads) % layers
            token = element // (width * heads * layers)

            # 阶段二：读取两端物理槽位；索引值及后续地址乘法使用 int64 防止溢出。
            src_token = tl.load(SRC_TOKEN_IDS + token * SRC_INDEX_STRIDE, mask=valid, other=0).to(tl.int64)
            dst_token = tl.load(DST_TOKEN_IDS + token * DST_INDEX_STRIDE, mask=valid, other=0).to(tl.int64)
            src_offset = (
                layer * tl.cast(SRC_STRIDES[field][0], tl.int64)
                + src_token * tl.cast(SRC_STRIDES[field][1], tl.int64)
                + head * tl.cast(SRC_STRIDES[field][2], tl.int64)
                + column * tl.cast(SRC_STRIDES[field][3], tl.int64)
            )
            dst_offset = (
                layer * tl.cast(DST_STRIDES[field][0], tl.int64)
                + dst_token * tl.cast(DST_STRIDES[field][1], tl.int64)
                + head * tl.cast(DST_STRIDES[field][2], tl.int64)
                + column * tl.cast(DST_STRIDES[field][3], tl.int64)
            )

            # 阶段三：直接从远端存储读出并写入目标槽位，无 gather 或跨设备中转存储。
            value = tl.load(source + src_offset, mask=valid, other=0)
            tl.store(destination + dst_offset, value, mask=valid)
