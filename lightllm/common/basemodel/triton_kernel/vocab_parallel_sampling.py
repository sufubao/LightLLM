"""Communicate vocabulary candidates instead of dense logits.

Temporary candidate buffers use int64 slots: logits are bitcast to
int32 before storage, while token IDs retain their integer representation.
Merges view buffers as [batch, part/rank, candidate slots]. Explicit strides
preserve coalesced local reads and describe gathered data without a copy.
All reductions read native projection strides and use only tile-sized fp32 data.
"""

import torch
import triton
import triton.language as tl

from lightllm.distributed.communication_op import all_gather_into_tensor


@triton.jit
def _local_top1(
    X,
    P,
    V: tl.constexpr,
    START: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr = 2048,
):
    batch, part = tl.program_id(0), tl.program_id(1)
    offsets = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets * S0 + batch * S1, offsets < V, other=-float("inf")).to(tl.float32)
    maximum = tl.max(x, 0)
    token = tl.min(tl.where((offsets < V) & (x == maximum), offsets.to(tl.int64) + START, 0x7FFFFFFFFFFFFFFF), 0)
    width: tl.constexpr = 2
    parts: tl.constexpr = triton.cdiv(V, BLOCK) if V > 0 else 1
    out = P + (batch * parts + part) * width
    tl.store(out, maximum.to(tl.int32, bitcast=True).to(tl.int64))
    tl.store(out + 1, token)


@triton.jit
def _merge_top1(P, OUT, PARTS: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr):
    batch = tl.program_id(0)
    width: tl.constexpr = 2
    offsets = tl.arange(0, triton.next_power_of_2(PARTS))
    ptr = P + batch * S0 + offsets * S1
    bits = tl.load(ptr, offsets < PARTS, other=0).to(tl.int32)
    x = tl.where(offsets < PARTS, bits.to(tl.float32, bitcast=True), -float("inf"))
    ids = tl.load(ptr + 1, offsets < PARTS, other=0x7FFFFFFFFFFFFFFF)
    maximum = tl.max(x, 0)
    token = tl.min(tl.where(x == maximum, ids, 0x7FFFFFFFFFFFFFFF), 0)
    tl.store(OUT + batch * width, maximum.to(tl.int32, bitcast=True).to(tl.int64))
    tl.store(OUT + batch * width + 1, token)


@triton.jit
def _unpack_top1(P, OUT, IDS, RANKS: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr):
    batch = tl.program_id(0)
    ranks = tl.arange(0, triton.next_power_of_2(RANKS))
    ptr = P + batch * S0 + ranks * S1
    values = tl.load(ptr, ranks < RANKS, other=0).to(tl.int32).to(tl.float32, bitcast=True)
    ids = tl.load(ptr + 1, ranks < RANKS, other=0)
    tl.store(OUT + batch * RANKS + ranks, values, ranks < RANKS)
    tl.store(IDS + batch * RANKS + ranks, ids, ranks < RANKS)


@triton.jit
def _candidate_key(values, ids):
    # Normalize signed zero so ties always prefer the lowest token ID.
    bits = tl.where(values == 0, 0.0, values).to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    return (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - ids.to(tl.uint32)).to(tl.uint64)


@triton.jit
def _topk_tiles(
    X,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    START: tl.constexpr = None,
    IDS=None,
    BLOCK: tl.constexpr = 1024,
):
    batch, part = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    offsets = part * BLOCK + lane
    if START is not None:
        values = tl.load(X + offsets * S0 + batch * S1, offsets < N, other=-float("inf")).to(tl.float32)
        ids = offsets.to(tl.int64) + START
    else:
        ptr = X + batch * S0 + offsets // K * S1 + offsets % K * 2
        values = tl.load(ptr, offsets < N, other=0).to(tl.int32).to(tl.float32, bitcast=True)
        ids = tl.load(ptr + 1, offsets < N, other=0x7FFFFFFFFFFFFFFF)
    valid = (offsets < N) & (ids <= 0xFFFFFFFF)
    keys = tl.where(valid, _candidate_key(values, ids), 0)
    if K == 1:
        keys = tl.max(keys, 0)[None]
    else:
        keys = tl.topk(keys, k=triton.next_power_of_2(K))
    lane = tl.arange(0, triton.next_power_of_2(K))
    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered)
    selected = tl.where(keys != 0, bits.to(tl.float32, bitcast=True), -float("inf"))
    selected_ids = tl.where(keys != 0, (0xFFFFFFFF - keys.to(tl.uint32)).to(tl.int64), 0x7FFFFFFFFFFFFFFF)
    if IDS is None:
        parts: tl.constexpr = triton.cdiv(N, BLOCK) if N > 0 else 1
        ptr = OUT + ((batch * parts + part) * K + lane) * 2
        tl.store(ptr, selected.to(tl.int32, bitcast=True).to(tl.int64), lane < K)
        tl.store(ptr + 1, selected_ids, lane < K)
    else:
        tl.store(OUT + batch * K + lane, selected, lane < K)
        tl.store(IDS + batch * K + lane, selected_ids, lane < K)


def vocab_parallel_candidates(
    local_logits: torch.Tensor,
    vocab_start: int,
    vocab_size: int,
    top_k: int = 1,
    group=None,
    world_size: int = 1,
    alloc_func=None,
):
    """Return candidate logits and global token IDs.

    ``local_logits`` is native ``[local_vocab, batch]`` with arbitrary strides.
    Top1 returns ``[batch, world_size]``, retaining each rank's local winner in
    rank order for downstream argmax and approximate probability calculation.
    Other supported values return global candidates of shape
    ``[batch, min(top_k, vocab_size)]``.
    Finite ties select the smallest global ID. Every rank must use the same
    arguments except its shard and ``vocab_start``. The supplied allocator follows ``torch.empty``.
    """
    assert local_logits.is_cuda and local_logits.ndim == 2
    assert top_k in (1, 16, 32, 64, 128, 256, 512)
    assert 0 < vocab_size <= 0xFFFFFFFF and world_size >= 1
    assert 0 <= vocab_start <= vocab_size
    assert vocab_start + local_logits.shape[0] <= vocab_size
    allocate = torch.empty if alloc_func is None else alloc_func

    def alloc(shape, dtype=torch.int64):
        return allocate(shape, dtype=dtype, device=local_logits.device)

    local_vocab, batch = local_logits.shape
    k = min(top_k, vocab_size)
    output_width = world_size if top_k == 1 else k
    values, ids = alloc((batch, output_width), torch.float32), alloc((batch, output_width))
    if batch == 0:
        return values, ids
    width = 2 * k
    block = 2048 if top_k == 1 else 1024
    parts = max(1, triton.cdiv(local_vocab, block))
    packed = alloc((batch, parts, width))
    if top_k == 1:
        _local_top1[(batch, parts)](local_logits, packed, local_vocab, vocab_start, *local_logits.stride())
        reduced = alloc((batch, width))
        _merge_top1[(batch,)](packed, reduced, parts, *packed.stride()[:2])
        packed = reduced
    else:
        _topk_tiles[(batch, parts)](local_logits, packed, local_vocab, k, *local_logits.stride(), START=vocab_start)
        while parts > 1:
            n = parts * k
            parts = triton.cdiv(n, block)
            reduced = alloc((batch, parts, width))
            _topk_tiles[(batch, parts)](packed, reduced, n, k, *packed.stride()[:2])
            packed = reduced
    packed = packed.view(batch, width)
    if world_size > 1:
        gathered = alloc((world_size * batch, width))
        all_gather_into_tensor(gathered, packed, group=group)
    else:
        gathered = packed
    gathered = gathered.view(world_size, batch, width).transpose(0, 1)
    if top_k == 1:
        _unpack_top1[(batch,)](gathered, values, ids, world_size, *gathered.stride()[:2])
    else:
        _topk_tiles[(batch, 1)](
            gathered,
            values,
            world_size * k,
            k,
            *gathered.stride()[:2],
            IDS=ids,
            BLOCK=triton.next_power_of_2(world_size * k),
        )
    return values, ids
