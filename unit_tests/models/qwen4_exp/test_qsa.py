import math

import pytest
import torch

from lightllm.models.qwen4_exp.triton_kernel.qsa import (
    qsa_mqa_flat,
    qsa_select_tokens,
    qsa_sparse_attention,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _physical_layout(length: int):
    generator = torch.Generator().manual_seed(9301)
    physical = torch.randperm(length, generator=generator, dtype=torch.int32)
    req_to_token = physical.cuda().unsqueeze(0)
    return physical, req_to_token


def test_qsa_flat_scoring_and_selection_match_reference():
    torch.manual_seed(41)
    sequence_length = 2304
    columns = sequence_length // 4
    _, req_to_token = _physical_layout(sequence_length)
    cache = torch.zeros(
        sequence_length, 128, dtype=torch.bfloat16, device="cuda"
    )
    group_starts = torch.arange(columns, device="cuda", dtype=torch.int64) * 4
    group_physical = req_to_token[0].index_select(0, group_starts).long()
    compressed = torch.randn(
        columns, 128, dtype=torch.bfloat16, device="cuda"
    )
    cache.index_copy_(0, group_physical, compressed)

    query_positions = torch.tensor([2048, 2055, 2303], dtype=torch.int32, device="cuda")
    row_lengths = torch.full_like(query_positions, sequence_length)
    row_req_ids = torch.zeros_like(query_positions)
    query = torch.randn(3, 4, 128, dtype=torch.bfloat16, device="cuda")

    logits, visible = qsa_mqa_flat(
        query,
        cache,
        req_to_token,
        row_req_ids,
        query_positions,
        row_lengths,
        compress_ratio=4,
        num_columns=columns,
    )
    expected_visible = (query_positions + 1) // 4
    torch.testing.assert_close(visible, expected_visible)
    reference = torch.einsum(
        "cd,rhd->rch", compressed.float(), query.float()
    ).relu().sum(-1) / math.sqrt(128)
    for row in range(query.shape[0]):
        width = int(expected_visible[row])
        torch.testing.assert_close(
            logits[row, :width], reference[row, :width], rtol=0, atol=0.08
        )
        assert torch.isneginf(logits[row, width:]).all()

    selected = qsa_select_tokens(
        query,
        cache,
        req_to_token,
        row_req_ids,
        query_positions,
        row_lengths,
        max_sequence_length=sequence_length,
        token_topk=2048,
        compress_ratio=4,
    )
    assert selected.shape == (3, 2051)
    for row in range(query.shape[0]):
        complete = min(int(expected_visible[row]), 512)
        expanded = selected[row, : complete * 4].view(-1, 4)
        assert torch.equal(
            expanded[:, 1:] - expanded[:, :-1],
            torch.ones_like(expanded[:, 1:]),
        )
        expected_blocks = torch.topk(
            reference[row, : int(expected_visible[row])],
            k=complete,
            sorted=False,
        ).indices
        assert set((expanded[:, 0] // 4).cpu().tolist()) == set(
            expected_blocks.cpu().tolist()
        )
        valid = selected[row][selected[row] >= 0]
        assert (valid <= query_positions[row]).all()


def test_qsa_sparse_attention_matches_dense_selected_reference():
    torch.manual_seed(73)
    cache_rows = 2400
    _, req_to_token = _physical_layout(cache_rows)
    key_cache = torch.randn(
        cache_rows, 1, 256, dtype=torch.bfloat16, device="cuda"
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(2, 6, 256, dtype=torch.bfloat16, device="cuda")
    logical = torch.stack(
        [
            torch.randperm(2200, device="cuda")[:2051],
            torch.randperm(2300, device="cuda")[:2051],
        ]
    ).to(torch.int32)
    row_req_ids = torch.zeros(2, dtype=torch.int32, device="cuda")

    actual = qsa_sparse_attention(
        query,
        key_cache,
        value_cache,
        logical,
        req_to_token,
        row_req_ids,
    )
    references = []
    for row in range(query.shape[0]):
        physical = req_to_token[0].index_select(0, logical[row].long()).long()
        keys = key_cache[physical, 0].float()
        values = value_cache[physical, 0].float()
        scores = query[row].float() @ keys.T / math.sqrt(256)
        probabilities = torch.softmax(scores, dim=-1).to(torch.bfloat16).float()
        references.append(probabilities @ values)
    reference = torch.stack(references).to(torch.bfloat16)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0.03)
