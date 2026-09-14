import pytest
import torch

from lightllm.common.basemodel.triton_kernel.vocab_parallel_sampling import vocab_parallel_candidates


@pytest.mark.parametrize("top_k", [16, 32, 64, 128, 256, 512])
def test_vocab_parallel_candidates_supports_configured_topk_widths(top_k):
    torch.manual_seed(1534 + top_k)
    local_logits = torch.randn(777, 3, dtype=torch.float32, device="cuda")

    values, token_ids = vocab_parallel_candidates(
        local_logits=local_logits,
        vocab_start=0,
        vocab_size=local_logits.shape[0],
        top_k=top_k,
        world_size=1,
    )

    expected = torch.topk(local_logits.T, top_k, dim=-1).values
    actual_sorted = torch.sort(values, dim=-1, descending=True).values
    torch.testing.assert_close(actual_sorted, expected)
    rows = torch.arange(local_logits.shape[1], device="cuda")[:, None]
    torch.testing.assert_close(values, local_logits.T[rows, token_ids])
