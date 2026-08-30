import math

import torch
import torch.nn.functional as F


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def reset_ple_new_request_state(
    req_ids: torch.Tensor,
    new_request_mask: torch.Tensor,
    state_indices: torch.Tensor,
    context_buffer: torch.Tensor,
    conv_buffer: torch.Tensor,
    eos_token_id: int,
) -> None:
    """Reset new PLE requests without data-dependent host control flow.

    All gathers and writes retain the fixed request shape, which keeps the
    operation safe inside prefill CUDA Graph capture. Cached requests write
    their existing values back unchanged.
    """
    req_ids = req_ids.long()
    new_request_mask = new_request_mask.bool()
    if req_ids.ndim != 1 or new_request_mask.shape != req_ids.shape:
        raise ValueError("req_ids and new_request_mask must be matching 1-D tensors")
    if context_buffer.shape[:2] != conv_buffer.shape[:2]:
        raise ValueError("PLE context and convolution buffers must share request/state dimensions")

    current_indices = state_indices.index_select(0, req_ids)
    reset_indices = torch.where(
        new_request_mask,
        torch.zeros_like(current_indices),
        current_indices,
    )
    state_indices.index_copy_(0, req_ids, reset_indices)

    state_width = context_buffer.shape[1]
    initial_slots = req_ids * state_width
    flat_context = context_buffer.flatten(0, 1)
    current_context = flat_context.index_select(0, initial_slots)
    context_mask = new_request_mask.view(
        new_request_mask.shape[0], *([1] * (current_context.ndim - 1))
    )
    reset_context = torch.where(
        context_mask,
        torch.full_like(current_context, eos_token_id),
        current_context,
    )
    flat_context.index_copy_(0, initial_slots, reset_context)

    flat_conv = conv_buffer.flatten(0, 1)
    current_conv = flat_conv.index_select(0, initial_slots)
    conv_mask = new_request_mask.view(
        new_request_mask.shape[0], *([1] * (current_conv.ndim - 1))
    )
    reset_conv = torch.where(conv_mask, torch.zeros_like(current_conv), current_conv)
    flat_conv.index_copy_(0, initial_slots, reset_conv)


def packed_ple_conv1d(
    conv_input: torch.Tensor,
    base_conv_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    conv_weight: torch.Tensor,
    *,
    dilation: int,
    max_query_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run PLE dilated convolution for packed prefill requests.

    This is the fixed-shape, graph-safe equivalent of invoking ``conv1d`` once
    per request. It follows the packing scheme used by vLLM's Qwen4 PLE path
    and avoids both CUDA-to-host synchronizations and Python launch loops.
    """
    if conv_input.ndim != 2 or base_conv_states.ndim != 3:
        raise ValueError("conv_input must be 2-D and base_conv_states must be 3-D")
    if conv_input.shape[1] != base_conv_states.shape[1]:
        raise ValueError("PLE input and cached state hidden dimensions differ")
    request_num = base_conv_states.shape[0]
    if cu_seqlens.shape != (request_num + 1,):
        raise ValueError("cu_seqlens must contain one boundary per request plus one")
    if max_query_len <= 0:
        raise ValueError("max_query_len must be positive")

    q_starts = cu_seqlens.to(torch.int64)
    lengths = q_starts[1:] - q_starts[:-1]
    positions = torch.arange(
        conv_input.shape[0], device=conv_input.device, dtype=torch.int64
    )
    request_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    column_indices = positions - q_starts.index_select(0, request_indices)

    packed = conv_input.new_zeros(
        (request_num, max_query_len, conv_input.shape[1])
    )
    packed[request_indices, column_indices] = conv_input
    packed = packed.transpose(1, 2).contiguous()
    history = torch.cat((base_conv_states.to(conv_input.dtype), packed), dim=-1)

    conv_output = F.conv1d(
        history,
        conv_weight,
        groups=conv_input.shape[1],
        dilation=dilation,
    )
    conv_output = F.silu(conv_output).transpose(1, 2).contiguous()
    output = conv_output[request_indices, column_indices]

    state_len = base_conv_states.shape[-1]
    state_offsets = torch.arange(
        state_len, device=history.device, dtype=torch.int64
    ).view(1, 1, state_len)
    next_states = history.gather(
        2,
        (lengths.view(request_num, 1, 1) + state_offsets).expand(
            -1, history.shape[1], -1
        ),
    )
    return output, next_states.to(base_conv_states.dtype)


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(
    unigram_vocab_size: int,
    ngram_size: int,
    ple_layer_index: int,
    seed: int = 1234,
) -> torch.Tensor:
    max_multiplier = ((1 << 63) - 1) // max(unigram_vocab_size, 1)
    half_bound = max(1, max_multiplier // 2)
    base_seed = seed + _PLE_LAYER_PRIME * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


def build_ngram_vocab_layout(
    *,
    ngram_size: int,
    heads_per_ngram: int,
    ngram_vocab_size_base: int,
    ple_layer_index: int,
    make_divisible_by: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    sizes = []
    offsets = []
    offset = 0
    for local_head in range(ngram_heads):
        global_head = ple_layer_index * ngram_heads + local_head
        size = nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
        sizes.append(size)
        offsets.append(offset)
        offset += size
    padded_vocab_size = math.ceil(offset / make_divisible_by) * make_divisible_by
    return (
        torch.tensor(sizes, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        padded_vocab_size,
    )


def _shift_right_ignore_eos(
    token_ids: torch.Tensor,
    shift: int,
    eos_token_id: int,
) -> torch.Tensor:
    if shift == 0:
        return token_ids
    positions = torch.arange(
        token_ids.numel(), device=token_ids.device, dtype=torch.long
    )
    eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=0).values
    previous_eos = torch.cat(
        [eos_positions.new_full((1,), -1), previous_eos_inclusive[:-1]]
    )
    position_in_segment = positions - previous_eos - 1
    source_positions = positions - shift
    shifted = token_ids.gather(0, source_positions.clamp_min(0))
    valid = (source_positions >= 0) & (position_in_segment >= shift)
    return torch.where(valid, shifted, token_ids.new_full((), eos_token_id))


def build_packed_ngram_ids(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    context_ids: torch.Tensor,
    *,
    layer_multipliers: torch.Tensor,
    head_vocab_sizes: torch.Tensor,
    head_offsets: torch.Tensor,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build PLE ids for packed requests and return their updated token context.

    ``context_ids`` contains the previous ``ngram_size - 1`` tokens for each
    request.  EOS boundaries are honored exactly as in the Transformers
    implementation, so n-grams never cross conversations or turns separated
    by the model EOS token.
    """

    input_ids = input_ids.long().reshape(-1)
    cu_seqlens = cu_seqlens.long().reshape(-1)
    context_len = ngram_size - 1
    batch_size = cu_seqlens.numel() - 1
    expected_context_shape = (batch_size, context_len)
    if tuple(context_ids.shape) != expected_context_shape:
        raise ValueError(
            f"context_ids shape {tuple(context_ids.shape)} must be {expected_context_shape}"
        )
    multipliers = layer_multipliers.to(device=input_ids.device, dtype=torch.long)
    vocab_sizes = head_vocab_sizes.to(device=input_ids.device, dtype=torch.long)
    offsets = head_offsets.to(device=input_ids.device, dtype=torch.long)

    # Pack every request into a fixed-width matrix.  This mirrors vLLM's
    # Qwen4 PLE implementation and, unlike slicing with ``int(cu_seqlens[i])``,
    # remains entirely on the device during CUDA Graph capture.
    token_positions = torch.arange(
        input_ids.numel(), device=input_ids.device, dtype=torch.long
    )
    request_indices = torch.searchsorted(cu_seqlens[1:], token_positions, right=True)
    columns = token_positions - cu_seqlens.index_select(0, request_indices)
    request_tokens = input_ids.new_full(
        (batch_size, input_ids.numel()), eos_token_id
    )
    request_tokens[request_indices, columns] = input_ids
    context = torch.cat((context_ids.long(), request_tokens), dim=1)

    context_positions = torch.arange(
        context.shape[1], device=input_ids.device, dtype=torch.long
    )
    eos_positions = torch.where(
        context == eos_token_id,
        context_positions.unsqueeze(0),
        -1,
    )
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        (
            eos_positions.new_full((batch_size, 1), -1),
            previous_eos_inclusive[:, :-1],
        ),
        dim=1,
    )
    position_in_segment = context_positions.unsqueeze(0) - previous_eos - 1
    shifted_tokens = [context]
    for shift in range(1, ngram_size):
        source = context_positions - shift
        shifted = context.gather(
            1, source.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        )
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        shifted_tokens.append(
            torch.where(valid, shifted, context.new_full((), eos_token_id))
        )

    adjusted_columns = columns + context_len
    id_blocks = []
    for ngram in range(2, ngram_size + 1):
        head_start = (ngram - 2) * heads_per_ngram
        head_end = head_start + heads_per_ngram
        mixed_ids = shifted_tokens[0] * multipliers[0]
        for position in range(1, ngram):
            mixed_ids = torch.bitwise_xor(
                mixed_ids, shifted_tokens[position] * multipliers[position]
            )
        ids = torch.remainder(
            mixed_ids.unsqueeze(-1), vocab_sizes[head_start:head_end]
        ) + offsets[head_start:head_end]
        id_blocks.append(ids[request_indices, adjusted_columns])

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    next_context_columns = lengths[:, None] + torch.arange(
        context_len, device=input_ids.device, dtype=torch.long
    )[None, :]
    next_context = context.gather(1, next_context_columns)
    return torch.cat(id_blocks, dim=-1), next_context


def build_decode_ngram_ids(
    input_ids: torch.Tensor,
    context_ids: torch.Tensor,
    *,
    layer_multipliers: torch.Tensor,
    head_vocab_sizes: torch.Tensor,
    head_offsets: torch.Tensor,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized one-token PLE hashing for graph-safe decode."""

    input_ids = input_ids.long().reshape(-1)
    context_len = ngram_size - 1
    if tuple(context_ids.shape) != (input_ids.numel(), context_len):
        raise ValueError(
            f"context_ids shape {tuple(context_ids.shape)} must be "
            f"{(input_ids.numel(), context_len)}"
        )

    multipliers = layer_multipliers.to(device=input_ids.device, dtype=torch.long)
    vocab_sizes = head_vocab_sizes.to(device=input_ids.device, dtype=torch.long)
    offsets = head_offsets.to(device=input_ids.device, dtype=torch.long)
    shifted_tokens = [input_ids]
    for shift in range(1, ngram_size):
        token = context_ids[:, -shift].long()
        if shift > 1:
            crossed_eos = (context_ids[:, -shift:] == eos_token_id).any(dim=-1)
            token = torch.where(
                crossed_eos, token.new_full((), eos_token_id), token
            )
        shifted_tokens.append(token)

    blocks = []
    for ngram in range(2, ngram_size + 1):
        head_start = (ngram - 2) * heads_per_ngram
        head_end = head_start + heads_per_ngram
        mixed_ids = shifted_tokens[0] * multipliers[0]
        for position in range(1, ngram):
            mixed_ids = torch.bitwise_xor(
                mixed_ids, shifted_tokens[position] * multipliers[position]
            )
        ids = torch.remainder(
            mixed_ids.unsqueeze(-1), vocab_sizes[head_start:head_end]
        )
        blocks.append(ids + offsets[head_start:head_end])

    next_context = torch.cat([context_ids[:, 1:], input_ids[:, None]], dim=-1)
    return torch.cat(blocks, dim=-1), next_context


def expand_mtp_decode_contexts(
    input_ids: torch.Tensor,
    base_context_ids: torch.Tensor,
    b_mtp_index: torch.Tensor,
) -> torch.Tensor:
    """Build the causal pre-token context for every MTP verification row.

    ``base_context_ids`` is the persisted context at the start of the verify
    step, repeated once per row. Rows belonging to one request are consecutive
    and ``b_mtp_index`` is ``0, 1, ...`` within that run. This reconstructs the
    contexts produced by feeding those rows sequentially without a CPU loop or
    mutating the persisted state, which keeps CUDA-graph replay safe.
    """

    input_ids = input_ids.long().reshape(-1)
    b_mtp_index = b_mtp_index.long().reshape(-1)
    context_len = base_context_ids.shape[-1]
    expected_shape = (input_ids.numel(), context_len)
    if tuple(base_context_ids.shape) != expected_shape:
        raise ValueError(
            f"base_context_ids shape {tuple(base_context_ids.shape)} must be "
            f"{expected_shape}"
        )
    if b_mtp_index.shape != input_ids.shape:
        raise ValueError("b_mtp_index must have one entry per input token")

    row_ids = torch.arange(input_ids.numel(), device=input_ids.device)
    columns = []
    for shift in range(context_len, 0, -1):
        use_verify_row = b_mtp_index >= shift
        previous_row = (row_ids - shift).clamp_min(0)
        verify_token = input_ids.index_select(0, previous_row)
        base_column = (context_len + b_mtp_index - shift).clamp(
            min=0, max=context_len - 1
        )
        base_token = base_context_ids.gather(1, base_column[:, None]).squeeze(1)
        columns.append(torch.where(use_verify_row, verify_token, base_token))
    return torch.stack(columns, dim=-1)


def build_mtp_conv_window(
    conv_input: torch.Tensor,
    base_history: torch.Tensor,
    b_mtp_index: torch.Tensor,
) -> torch.Tensor:
    """Return each verify row's causal PLE convolution window.

    The returned window contains the persisted history followed by the current
    row after accounting for earlier rows in the same MTP request run. Its last
    ``base_history.shape[-1]`` entries are the candidate state to persist if
    that row is the final accepted row.
    """

    if conv_input.ndim != 2 or base_history.ndim != 3:
        raise ValueError("conv_input must be 2-D and base_history must be 3-D")
    if conv_input.shape[:2] != base_history.shape[:2]:
        raise ValueError("conv_input and base_history batch/channel shapes differ")
    b_mtp_index = b_mtp_index.long().reshape(-1)
    if b_mtp_index.shape[0] != conv_input.shape[0]:
        raise ValueError("b_mtp_index must have one entry per convolution row")

    row_ids = torch.arange(conv_input.shape[0], device=conv_input.device)
    history_len = base_history.shape[-1]
    columns = []
    for back in range(history_len, -1, -1):
        use_verify_row = b_mtp_index >= back
        previous_row = (row_ids - back).clamp_min(0)
        verify_value = conv_input.index_select(0, previous_row)
        base_column = (history_len + b_mtp_index - back).clamp(
            min=0, max=history_len - 1
        )
        base_value = base_history.gather(
            2,
            base_column[:, None, None].expand(-1, base_history.shape[1], 1),
        ).squeeze(-1)
        columns.append(
            torch.where(use_verify_row[:, None], verify_value, base_value)
        )
    return torch.stack(columns, dim=-1)


def compute_shard_overlap(
    *,
    checkpoint_start: int,
    checkpoint_rows: int,
    tp_start: int,
    tp_end: int,
) -> tuple[int, int, int] | None:
    overlap_start = max(checkpoint_start, tp_start)
    overlap_end = min(checkpoint_start + checkpoint_rows, tp_end)
    if overlap_start >= overlap_end:
        return None
    return (
        overlap_start - checkpoint_start,
        overlap_start - tp_start,
        overlap_end - overlap_start,
    )
