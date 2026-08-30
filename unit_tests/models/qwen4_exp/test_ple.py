import torch
import torch.nn.functional as F
import pytest

from lightllm.models.qwen4_exp.ple import (
    build_decode_ngram_ids,
    build_layer_multipliers,
    build_mtp_conv_window,
    build_ngram_vocab_layout,
    build_packed_ngram_ids,
    compute_shard_overlap,
    expand_mtp_decode_contexts,
    packed_ple_conv1d,
    reset_ple_new_request_state,
)


def test_qwen38_flash_next_published_ple_layout():
    sizes, offsets, padded_vocab_size = build_ngram_vocab_layout(
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        ple_layer_index=0,
        make_divisible_by=128,
    )
    assert sizes.shape == offsets.shape == (16,)
    assert padded_vocab_size == 320_001_536
    assert padded_vocab_size // 128 == 2_500_012
    assert torch.equal(offsets[1:], sizes.cumsum(0)[:-1])


def test_packed_ngram_ids_match_per_request_reference_and_update_context():
    eos = 99
    input_ids = torch.tensor([10, 11, eos, 12, 20, 21, 22, 23])
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    contexts = torch.tensor([[7, 8], [eos, 19]])
    multipliers = build_layer_multipliers(128, 3, 0)
    sizes = torch.tensor([101, 103, 107, 109])
    offsets = torch.tensor([0, 101, 204, 311])

    actual, next_context = build_packed_ngram_ids(
        input_ids,
        cu_seqlens,
        contexts,
        layer_multipliers=multipliers,
        head_vocab_sizes=sizes,
        head_offsets=offsets,
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=eos,
    )

    # The first token after EOS must not hash with tokens from the prior segment.
    isolated, _ = build_packed_ngram_ids(
        torch.tensor([12]),
        torch.tensor([0, 1]),
        torch.tensor([[eos, eos]]),
        layer_multipliers=multipliers,
        head_vocab_sizes=sizes,
        head_offsets=offsets,
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=eos,
    )
    torch.testing.assert_close(actual[3], isolated[0])
    assert torch.equal(next_context, torch.tensor([[eos, 12], [22, 23]]))

    # Packed execution is exactly the concatenation of independent requests.
    independently = []
    for index, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        ids, _ = build_packed_ngram_ids(
            input_ids[start:end],
            torch.tensor([0, end - start]),
            contexts[index : index + 1],
            layer_multipliers=multipliers,
            head_vocab_sizes=sizes,
            head_offsets=offsets,
            ngram_size=3,
            heads_per_ngram=2,
            eos_token_id=eos,
        )
        independently.append(ids)
    assert torch.equal(actual, torch.cat(independently))


def test_checkpoint_shard_overlap():
    assert compute_shard_overlap(
        checkpoint_start=100, checkpoint_rows=50, tp_start=120, tp_end=180
    ) == (20, 0, 30)
    assert (
        compute_shard_overlap(
            checkpoint_start=100, checkpoint_rows=50, tp_start=0, tp_end=100
        )
        is None
    )


def test_packed_ngram_ids_preserve_empty_request_context():
    multipliers = build_layer_multipliers(128, 3, 0)
    ids, next_context = build_packed_ngram_ids(
        torch.tensor([7]),
        torch.tensor([0, 0, 1]),
        torch.tensor([[90, 91], [92, 93]]),
        layer_multipliers=multipliers,
        head_vocab_sizes=torch.tensor([101, 103]),
        head_offsets=torch.tensor([0, 101]),
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=99,
    )
    assert ids.shape == (1, 2)
    assert torch.equal(next_context[0], torch.tensor([90, 91]))
    assert torch.equal(next_context[1], torch.tensor([93, 7]))


def test_vectorized_decode_ngram_matches_packed_path():
    eos = 99
    contexts = torch.tensor([[99, 5], [4, 99], [4, 5], [99, 99]])
    current = torch.tensor([6, 7, 8, 9])
    multipliers = torch.tensor([3, 5, 7])
    vocab_sizes = torch.tensor([101, 103, 107, 109])
    offsets = torch.tensor([0, 101, 204, 311])
    decode_ids, next_context = build_decode_ngram_ids(
        current,
        contexts,
        layer_multipliers=multipliers,
        head_vocab_sizes=vocab_sizes,
        head_offsets=offsets,
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=eos,
    )

    expected = []
    expected_context = []
    for index in range(current.numel()):
        packed_ids, packed_context = build_packed_ngram_ids(
            current[index : index + 1],
            torch.tensor([0, 1]),
            contexts[index : index + 1],
            layer_multipliers=multipliers,
            head_vocab_sizes=vocab_sizes,
            head_offsets=offsets,
            ngram_size=3,
            heads_per_ngram=2,
            eos_token_id=eos,
        )
        expected.append(packed_ids)
        expected_context.append(packed_context)
    torch.testing.assert_close(decode_ids, torch.cat(expected))
    torch.testing.assert_close(next_context, torch.cat(expected_context))


def test_mtp_decode_contexts_match_sequential_request_runs():
    input_ids = torch.tensor([10, 11, 12, 20, 21])
    mtp_index = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)
    base_contexts = torch.tensor(
        [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4]]
    )

    contexts = expand_mtp_decode_contexts(
        input_ids, base_contexts, mtp_index
    )

    assert torch.equal(
        contexts,
        torch.tensor([[1, 2], [2, 10], [10, 11], [3, 4], [4, 20]]),
    )


def test_mtp_conv_window_matches_sequential_request_runs():
    conv_input = torch.tensor([[10.0], [11.0], [12.0], [20.0], [21.0]])
    mtp_index = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)
    base_history = torch.tensor(
        [
            [[1.0, 2.0, 3.0]],
            [[1.0, 2.0, 3.0]],
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
            [[4.0, 5.0, 6.0]],
        ]
    )

    window = build_mtp_conv_window(conv_input, base_history, mtp_index)

    assert torch.equal(
        window.squeeze(1),
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 10.0],
                [2.0, 3.0, 10.0, 11.0],
                [3.0, 10.0, 11.0, 12.0],
                [4.0, 5.0, 6.0, 20.0],
                [5.0, 6.0, 20.0, 21.0],
            ]
        ),
    )


def test_reset_ple_new_request_state_preserves_cached_rows():
    state_indices = torch.tensor([2, 3, 1], dtype=torch.int32)
    context = torch.arange(3 * 4 * 2).view(3, 4, 2)
    conv = torch.arange(3 * 4 * 2 * 3, dtype=torch.float32).view(3, 4, 2, 3)
    original_context = context.clone()
    original_conv = conv.clone()

    reset_ple_new_request_state(
        req_ids=torch.tensor([0, 1]),
        new_request_mask=torch.tensor([True, False]),
        state_indices=state_indices,
        context_buffer=context,
        conv_buffer=conv,
        eos_token_id=99,
    )

    assert torch.equal(state_indices, torch.tensor([0, 3, 1], dtype=torch.int32))
    assert torch.equal(context[0, 0], torch.tensor([99, 99]))
    assert torch.equal(conv[0, 0], torch.zeros_like(conv[0, 0]))
    assert torch.equal(context[1], original_context[1])
    assert torch.equal(conv[1], original_conv[1])


def test_packed_ple_conv_matches_independent_requests():
    torch.manual_seed(7)
    lengths = [3, 1, 4]
    hidden_size = 6
    state_len = 6
    dilation = 2
    kernel_size = 4
    conv_input = torch.randn(sum(lengths), hidden_size)
    base_states = torch.randn(len(lengths), hidden_size, state_len)
    weight = torch.randn(hidden_size, 1, kernel_size)
    cu_seqlens = torch.tensor([0, 3, 4, 8], dtype=torch.int32)

    output, next_states = packed_ple_conv1d(
        conv_input,
        base_states,
        cu_seqlens,
        weight,
        dilation=dilation,
        max_query_len=max(lengths),
    )

    expected_output = []
    expected_states = []
    for request_index, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        current = conv_input[start:end].transpose(0, 1)
        history = torch.cat((base_states[request_index], current), dim=-1)
        expected_states.append(history[:, -state_len:])
        expected_output.append(
            F.silu(
                F.conv1d(
                    history.unsqueeze(0),
                    weight,
                    groups=hidden_size,
                    dilation=dilation,
                ).squeeze(0).transpose(0, 1)
            )
        )

    torch.testing.assert_close(output, torch.cat(expected_output))
    torch.testing.assert_close(next_states, torch.stack(expected_states))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graph support")
def test_ple_prefill_helpers_are_cuda_graph_safe():
    device = torch.device("cuda")
    input_ids = torch.tensor([10, 11, 12, 20], device=device)
    cu_seqlens = torch.tensor([0, 3, 4], dtype=torch.int32, device=device)
    contexts = torch.tensor([[99, 7], [99, 19]], device=device)
    multipliers = torch.tensor([3, 5, 7], device=device)
    vocab_sizes = torch.tensor([101, 103, 107, 109], device=device)
    offsets = torch.tensor([0, 101, 204, 311], device=device)
    conv_input = torch.randn(4, 6, dtype=torch.float16, device=device)
    conv_states = torch.randn(2, 6, 6, dtype=torch.float16, device=device)
    conv_weight = torch.randn(6, 1, 4, dtype=torch.float16, device=device)

    # Warm kernels and allocator paths before capture.
    build_packed_ngram_ids(
        input_ids,
        cu_seqlens,
        contexts,
        layer_multipliers=multipliers,
        head_vocab_sizes=vocab_sizes,
        head_offsets=offsets,
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=99,
    )
    packed_ple_conv1d(
        conv_input,
        conv_states,
        cu_seqlens,
        conv_weight,
        dilation=2,
        max_query_len=3,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_ids, graph_context = build_packed_ngram_ids(
            input_ids,
            cu_seqlens,
            contexts,
            layer_multipliers=multipliers,
            head_vocab_sizes=vocab_sizes,
            head_offsets=offsets,
            ngram_size=3,
            heads_per_ngram=2,
            eos_token_id=99,
        )
        graph_conv, graph_state = packed_ple_conv1d(
            conv_input,
            conv_states,
            cu_seqlens,
            conv_weight,
            dilation=2,
            max_query_len=3,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert graph_ids.shape == (4, 4)
    assert graph_context.shape == (2, 2)
    assert graph_conv.shape == conv_input.shape
    assert graph_state.shape == conv_states.shape
