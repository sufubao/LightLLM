# SPDX-License-Identifier: Apache-2.0
"""CUDA/Triton kernels for Qwen4 experimental QSA.

LightLLM keeps token K/V in a flat physical cache instead of vLLM's paged
blocks.  The kernels below preserve QSA's scoring and sparse-attention math,
but resolve logical token positions through ``req_to_token_indexs`` directly.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024


@triton.jit
def _qsa_mrope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    section_ptr,
    stride_x_row,
    stride_x_head,
    stride_x_dim,
    stride_cos_axis,
    stride_cos_row,
    stride_sin_axis,
    stride_sin_row,
    INTERLEAVED: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
) -> None:
    head = tl.program_id(0)
    row = tl.program_id(1)
    offsets = tl.arange(0, ROTARY_DIM // 2)
    upper = offsets + ROTARY_DIM // 2

    if INTERLEAVED:
        section_h = tl.load(section_ptr + 1)
        section_w = tl.load(section_ptr + 2)
        h_mask = ((offsets % 3) == 1) & (offsets <= 3 * section_h)
        w_mask = ((offsets % 3) == 2) & (offsets <= 3 * section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        section_t = tl.load(section_ptr)
        section_h = tl.load(section_ptr + 1)
        t_end = section_t
        h_end = t_end + section_h
        t_mask = offsets < t_end
        h_mask = (offsets >= t_end) & (offsets < h_end)
        w_mask = offsets >= h_end

    cos_t = tl.load(
        cos_ptr + row * stride_cos_row + offsets,
        mask=t_mask,
        other=0.0,
    )
    cos_h = tl.load(
        cos_ptr + stride_cos_axis + row * stride_cos_row + offsets,
        mask=h_mask,
        other=0.0,
    )
    cos_w = tl.load(
        cos_ptr + 2 * stride_cos_axis + row * stride_cos_row + offsets,
        mask=w_mask,
        other=0.0,
    )
    sin_t = tl.load(
        sin_ptr + row * stride_sin_row + offsets,
        mask=t_mask,
        other=0.0,
    )
    sin_h = tl.load(
        sin_ptr + stride_sin_axis + row * stride_sin_row + offsets,
        mask=h_mask,
        other=0.0,
    )
    sin_w = tl.load(
        sin_ptr + 2 * stride_sin_axis + row * stride_sin_row + offsets,
        mask=w_mask,
        other=0.0,
    )
    cos = cos_t + cos_h + cos_w
    sin = sin_t + sin_h + sin_w

    base = x_ptr + row * stride_x_row + head * stride_x_head
    x0 = tl.load(base + offsets * stride_x_dim)
    x1 = tl.load(base + upper * stride_x_dim)
    tl.store(base + offsets * stride_x_dim, x0 * cos - x1 * sin)
    tl.store(base + upper * stride_x_dim, x0 * sin + x1 * cos)


def qsa_mrope_fwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: torch.Tensor,
    *,
    rotary_dim: int,
    is_interleaved: bool,
) -> torch.Tensor:
    """Apply Qwen's exact three-axis partial MRoPE in place."""

    if x.ndim != 3 or cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("QSA MRoPE expects x=[rows,heads,dim], cos/sin=[3,rows,dim/2]")
    if cos.shape[0] != 3 or cos.shape[1] != x.shape[0]:
        raise ValueError("QSA MRoPE position rows do not match the input")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > x.shape[2]:
        raise ValueError("invalid QSA rotary dimension")
    _qsa_mrope_kernel[(x.shape[1], x.shape[0])](
        x,
        cos,
        sin,
        mrope_section,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        INTERLEAVED=is_interleaved,
        NUM_HEADS=x.shape[1],
        ROTARY_DIM=rotary_dim,
        num_warps=1,
        num_stages=1,
    )
    return x


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    destinations_ptr,
    rows_ptr,
    valid_ptr,
    stride_cache_row,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    cache_rows,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    destination = tl.load(destinations_ptr + row)
    valid = (
        (row < num_rows)
        & tl.load(valid_ptr + row)
        & (destination >= 0)
        & (destination < cache_rows)
    )
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0.0,
    )
    tl.store(
        cache_ptr
        + tl.maximum(destination, 0).to(tl.int64) * stride_cache_row
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


def qsa_store_rows(
    cache: torch.Tensor,
    destinations: torch.Tensor,
    rows: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Store selected fixed-width rows into a flat physical-token cache."""

    if cache.ndim != 2 or rows.shape != (destinations.numel(), cache.shape[1]):
        raise ValueError("QSA cache store shapes are incompatible")
    if valid.shape != destinations.shape:
        raise ValueError("QSA cache store validity must match destinations")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        destinations,
        rows,
        valid,
        cache.stride(0),
        cache.stride(1),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        WIDTH=cache.shape[1],
        BLOCK_D=triton.next_power_of_2(cache.shape[1]),
        num_warps=4,
    )


@triton.jit
def _qsa_mqa_flat_kernel(
    q_ptr,
    compressed_cache_ptr,
    req_to_token_ptr,
    row_req_ids_ptr,
    query_positions_ptr,
    row_sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_row,
    stride_cache_dim,
    stride_req,
    stride_req_token,
    stride_logits_row,
    num_rows,
    num_columns,
    cache_rows,
    num_requests,
    score_divisor,
    REQ_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    MAX_N: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(row_req_ids_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(row_sequence_lengths_ptr + row)
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    if tile_start * BLOCK_N >= visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    query = tl.load(
        q_ptr
        + row * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(heads[None, :] < NUM_HEADS) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        logical_tokens = columns * COMPRESS_RATIO
        live = (columns < visible) & (logical_tokens < REQ_TABLE_WIDTH)
        physical_tokens = tl.load(
            req_to_token_ptr
            + safe_request * stride_req
            + logical_tokens * stride_req_token,
            mask=live & (request >= 0) & (request < num_requests),
            other=-1,
        )
        cache_valid = live & (physical_tokens >= 0) & (physical_tokens < cache_rows)
        safe_physical = tl.maximum(physical_tokens, 0).to(tl.int64)
        keys = tl.load(
            compressed_cache_ptr
            + safe_physical[:, None] * stride_cache_row
            + dims[None, :] * stride_cache_dim,
            mask=cache_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(
            heads[None, :] < NUM_HEADS,
            tl.maximum(scores, 0.0),
            0.0,
        )
        score = tl.sum(scores, axis=1) / score_divisor
        tl.store(
            logits_ptr + row * stride_logits_row + columns,
            tl.where(cache_valid, score, -float("inf")),
            mask=columns < num_columns,
        )


def qsa_mqa_flat(
    q: torch.Tensor,
    compressed_cache: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    row_req_ids: torch.Tensor,
    query_positions: torch.Tensor,
    row_sequence_lengths: torch.Tensor,
    *,
    compress_ratio: int,
    num_columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score compressed QSA groups held at group-start physical token rows."""

    if q.ndim != 3 or compressed_cache.ndim != 2:
        raise ValueError("QSA scoring expects q=[rows,heads,dim] and cache=[tokens,dim]")
    if compressed_cache.shape[1] != q.shape[2]:
        raise ValueError("QSA scoring dimensions do not match")
    rows = q.shape[0]
    for tensor in (row_req_ids, query_positions, row_sequence_lengths):
        if tensor.shape != (rows,):
            raise ValueError("QSA scoring metadata must have one element per row")
    # Unlike vLLM's persistent top-k operator, torch.topk has no per-row
    # length input.  Initialize the causal suffix so it can never be selected.
    logits = torch.full(
        (rows, num_columns),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    visible_blocks = torch.empty(rows, dtype=torch.int32, device=q.device)
    if not rows or not num_columns:
        return logits, visible_blocks
    block_n = 64
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    max_n = max(16, triton.next_power_of_2(q.shape[1]))
    tiles_per_program = 1 if rows <= 32 else 8
    _qsa_mqa_flat_kernel[
        (rows, triton.cdiv(num_columns, block_n * tiles_per_program))
    ](
        q,
        compressed_cache,
        req_to_token_indexs,
        row_req_ids,
        query_positions,
        row_sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        logits.stride(0),
        rows,
        num_columns,
        compressed_cache.shape[0],
        req_to_token_indexs.shape[0],
        float(math.sqrt(q.shape[2])),
        REQ_TABLE_WIDTH=req_to_token_indexs.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        MAX_N=max_n,
        COMPRESS_RATIO=compress_ratio,
        num_warps=2,
    )
    return logits, visible_blocks


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    row_sequence_lengths_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(row_sequence_lengths_ptr + row)
    complete_blocks = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr
        + row * stride_blocks_row
        + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows)
        & (columns < OUTPUT_WIDTH)
        & (is_expanded | is_tail)
        & (token >= 0)
        & (token < sequence_length)
    )
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    row_sequence_lengths: torch.Tensor,
    *,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices,
        query_positions,
        row_sequence_lengths,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )
    return out


def qsa_select_tokens(
    q: torch.Tensor,
    compressed_cache: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    row_req_ids: torch.Tensor,
    query_positions: torch.Tensor,
    row_sequence_lengths: torch.Tensor,
    *,
    max_sequence_length: int,
    token_topk: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Score, select, and expand fixed-width request-relative token indices."""

    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if not rows:
        return out
    columns = max_sequence_length // compress_ratio
    block_topk = token_topk // compress_ratio
    if columns < block_topk:
        raise ValueError("QSA selection requires at least token_topk visible capacity")
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    all_blocks = torch.arange(block_topk, dtype=torch.int32, device=q.device)
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits, _ = qsa_mqa_flat(
            q[row_slice],
            compressed_cache,
            req_to_token_indexs,
            row_req_ids[row_slice],
            query_positions[row_slice],
            row_sequence_lengths[row_slice],
            compress_ratio=compress_ratio,
            num_columns=columns,
        )
        if columns == block_topk:
            blocks = all_blocks.expand(row_end - row_start, -1)
        else:
            # Sorting guarantees that finite visible candidates precede the
            # -inf padding for early rows in a long packed prefill.
            blocks = torch.topk(
                logits,
                k=block_topk,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices.to(torch.int32)
        expand_qsa_block_indices(
            blocks,
            query_positions[row_slice],
            row_sequence_lengths[row_slice],
            compress_ratio=compress_ratio,
            token_topk=token_topk,
            out=out[row_slice],
        )
    return out


@triton.jit
def _qsa_sparse_flat_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    req_to_token_ptr,
    row_req_ids_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_k_row,
    stride_k_head,
    stride_k_dim,
    stride_v_row,
    stride_v_head,
    stride_v_dim,
    stride_indices_row,
    stride_req,
    stride_req_token,
    stride_output_row,
    stride_output_head,
    stride_output_dim,
    num_rows,
    cache_rows,
    num_requests,
    TOPK: tl.constexpr,
    REQ_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(row_req_ids_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_token < REQ_TABLE_WIDTH)
        )
        physical_token = tl.load(
            req_to_token_ptr
            + safe_request * stride_req
            + safe_token * stride_req_token,
            mask=valid,
            other=-1,
        )
        valid &= (physical_token >= 0) & (physical_token < cache_rows)
        safe_physical = tl.maximum(physical_token, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical[None, :] * stride_k_row
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_physical[:, None] * stride_v_row
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :],
            tl.math.exp2(scores - next_max[:, None]),
            0.0,
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :] * stride_output_dim,
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_dim,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr
        + (split_offsets * num_rows + row) * NUM_QUERY_HEADS
        + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr
        + row * stride_output_row
        + head * stride_output_head
        + dim_offsets * stride_output_dim,
        merged,
    )


def qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    row_req_ids: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run QSA sparse GQA over LightLLM's flat physical K/V cache."""

    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.shape[0] != q.shape[0] or row_req_ids.shape != (q.shape[0],):
        raise ValueError("QSA sparse metadata must have one row per query")
    if q.shape[2] != k_cache.shape[2] or q.shape[1] % k_cache.shape[1]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    if q.dtype != torch.bfloat16 or k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise ValueError("QSA sparse attention requires BF16 Q/K/V")
    if out is None:
        out = torch.empty_like(q)
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[1]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[1]
    small_profile_limit = 8 if block_m <= 8 else 4
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    _qsa_sparse_flat_gqa_splitk_kernel[
        (q.shape[0], k_cache.shape[1], num_splits)
    ](
        q,
        k_cache,
        v_cache,
        logical_indices,
        req_to_token_indexs,
        row_req_ids,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q.shape[0],
        k_cache.shape[0],
        req_to_token_indexs.shape[0],
        TOPK=logical_indices.shape[1],
        REQ_TABLE_WIDTH=req_to_token_indexs.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits == 1:
        return out
    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


__all__ = [
    "expand_qsa_block_indices",
    "qsa_mqa_flat",
    "qsa_mrope_fwd",
    "qsa_select_tokens",
    "qsa_sparse_attention",
    "qsa_store_rows",
]
