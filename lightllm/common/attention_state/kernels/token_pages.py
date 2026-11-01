import triton
import triton.language as tl


@triton.jit
def _copy_token_pages(
    # 数据指针：一次处理一个字段；调用方已应用 TP 切片，两侧 dtype 相同。
    LOCAL,  # GPU 视图：[layer, token_slot, head, dim]。
    EXTERNAL,  # CPU/PD 外部页视图：[page, layer, page_token, head, field_width]。
    # 索引指针：均位于 GPU；索引值是物理槽位/页 ID，允许不连续、不递增。
    TOKEN_IDS,  # TOKEN_IDS[token] 指向 GPU KV 存储中的 token 槽位。
    PAGE_IDS,  # GPU 页索引 Tensor，PAGE_IDS[token // PAGE_TOKENS] 给出外部物理页 ID。
    PAGE_READIES,  # 可选的 GPU bool 数组；True 表示该逻辑页已就绪，禁止覆盖。
    # 本次搬运范围：只覆盖有效 token，不包含尾页空余位置。
    token_count,  # 本次有效 token 数；调用方保证大于 0。
    element_count,  # layer 数 × token_count × HEADS × WIDTH，单位为转换后的搬运单元。
    # 编译期寻址配置：stride 的单位是元素数，不是字节数。
    TOKEN_STRIDE: tl.constexpr,  # token 索引步长，直接支持切片视图，不创建连续副本。
    LOCAL_STRIDES: tl.constexpr,  # 对应 LOCAL 的四个轴，保留原 GPU 存储步长。
    EXTERNAL_STRIDES: tl.constexpr,  # 对应 EXTERNAL 的五个轴，包含真实外部页跨度。
    PAGE_TOKENS: tl.constexpr,  # 每个外部页可容纳的 token 数。
    HEADS: tl.constexpr,  # 本次 TP 切片参与拷贝的 head 数。
    WIDTH: tl.constexpr,  # 字段末轴参与拷贝的元素数，如 dim、group 或 scale。
    # 编译期搬运配置。
    ELEMENT_BYTES: tl.constexpr,  # 单个元素字节数，用于选择同宽整数指针。
    HAS_PAGE_READIES: tl.constexpr,  # 无 ready 标记时编译掉检查，保持 PD/读回路径。
    WRITE: tl.constexpr,  # True：KV→外部页；False：外部页→KV。
    BLOCK: tl.constexpr,  # 每个 program 每轮处理的搬运单元数。
    NUM_STAGES: tl.constexpr,  # tl.range 的循环流水深度，不是 launch 的同名选项。
):
    # 阶段一：把两侧指针解释为同宽整数指针，逐位保留数据（包括 NaN 位模式）。
    # 这里只改变指针类型，不进行数值转换；元素宽度不变，因此原 stride 仍然有效。
    if ELEMENT_BYTES == 1:
        LOCAL = LOCAL.to(dtype=tl.pointer_type(element_ty=tl.uint8))
        EXTERNAL = EXTERNAL.to(dtype=tl.pointer_type(element_ty=tl.uint8))
    elif ELEMENT_BYTES == 2:
        LOCAL = LOCAL.to(dtype=tl.pointer_type(element_ty=tl.uint16))
        EXTERNAL = EXTERNAL.to(dtype=tl.pointer_type(element_ty=tl.uint16))
    elif ELEMENT_BYTES == 4:
        LOCAL = LOCAL.to(dtype=tl.pointer_type(element_ty=tl.uint32))
        EXTERNAL = EXTERNAL.to(dtype=tl.pointer_type(element_ty=tl.uint32))
    else:
        LOCAL = LOCAL.to(dtype=tl.pointer_type(element_ty=tl.uint64))
        EXTERNAL = EXTERNAL.to(dtype=tl.pointer_type(element_ty=tl.uint64))

    # 阶段二：确定当前 program 的起点和整张 grid 的步长。
    # 每轮处理 BLOCK 个元素，下一轮前进 grid_num * BLOCK，所有 program 共同覆盖完整范围。
    # 使用 int64 避免大容量寻址溢出；tl.cast 兼容 token_count=1 被特化为常量的情况。
    token_count = tl.cast(input=token_count, dtype=tl.int64)
    program_start = tl.program_id(axis=0).to(dtype=tl.int64) * BLOCK
    grid_stride = tl.num_programs(axis=0).to(dtype=tl.int64) * BLOCK
    block_offsets = tl.arange(start=0, end=BLOCK)
    # 起点超出 element_count 的 program 不进入循环；最后一块通过 valid 屏蔽尾部元素。
    for block_start in tl.range(program_start, element_count, grid_stride, num_stages=NUM_STAGES):
        offsets = block_start + block_offsets
        valid = offsets < element_count
        token = offsets // (WIDTH * HEADS) % token_count

        # 阶段三：把逻辑 token 位置映射到两侧物理存储。
        # slot 是 GPU token 槽位；page 是外部物理页；token % PAGE_TOKENS 是页内位置。
        # 例如 PAGE_TOKENS=16 时，逻辑 token 18 使用 TOKEN_IDS[18] 和 PAGE_IDS[1] 的第 2 个位置。
        # 索引读取同样使用 valid，避免尾部无效元素访问 TOKEN_IDS/PAGE_IDS 越界。
        # 写出 ready 页时同时屏蔽源 KV 读取和目标写入，不仅仅跳过 store。
        copy_block = True
        if WRITE and HAS_PAGE_READIES:
            ready = tl.load(pointer=PAGE_READIES + token // PAGE_TOKENS, mask=valid, other=1)
            valid = valid & ~ready
            # 一个 block 可能跨页：混合状态保留逐元素 mask；全部无需写出则跳过后续操作。
            copy_block = tl.sum(input=valid.to(dtype=tl.int32), axis=0) > 0
        if copy_block:
            dim = offsets % WIDTH
            head = offsets // WIDTH % HEADS
            storage_layer_index = offsets // (WIDTH * HEADS * token_count)
            slot = tl.load(pointer=TOKEN_IDS + token * TOKEN_STRIDE, mask=valid, other=0).to(dtype=tl.int64)
            page = tl.load(pointer=PAGE_IDS + token // PAGE_TOKENS, mask=valid, other=0).to(dtype=tl.int64)

            # 阶段四：按各轴真实 stride 计算元素偏移，兼容 TP 切片及轴顺序调整后的视图。
            # EXTERNAL 已指向当前字段在联合页中的起点；这里无需再次加 region.byte_start。
            # EXTERNAL_STRIDES[0] 使用整个外部页的跨度，不能替换为当前 传输字段的大小。
            local_offset = (
                storage_layer_index * LOCAL_STRIDES[0]
                + slot * LOCAL_STRIDES[1]
                + head * LOCAL_STRIDES[2]
                + dim * LOCAL_STRIDES[3]
            )
            external_offset = (
                page * EXTERNAL_STRIDES[0]
                + storage_layer_index * EXTERNAL_STRIDES[1]
                + (token % PAGE_TOKENS) * EXTERNAL_STRIDES[2]
                + head * EXTERNAL_STRIDES[3]
                + dim * EXTERNAL_STRIDES[4]
            )

            # 阶段五：按方向直接读写两侧存储，不生成 gather/scatter 中间 Tensor。
            # 两个方向使用相同的索引和有效范围；尾页空余、其他 TP 分片及其他字段保持不变。
            if WRITE:
                value = tl.load(pointer=LOCAL + local_offset, mask=valid, other=0)
                tl.store(pointer=EXTERNAL + external_offset, value=value, mask=valid)
            else:
                value = tl.load(pointer=EXTERNAL + external_offset, mask=valid, other=0)
                tl.store(pointer=LOCAL + local_offset, value=value, mask=valid)
