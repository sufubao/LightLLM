from types import SimpleNamespace

import torch

from lightllm.models.qwen4_exp.layer_infer.post_layer_infer import (
    Qwen4ExpPostLayerInfer,
)


class _FakeLMHead:
    def __init__(self, local_logits: torch.Tensor, vocab_size: int):
        self.local_logits = local_logits
        self.vocab_size = vocab_size

    def batch_major_forward(self, input, alloc_func):
        assert input.shape[0] == self.local_logits.shape[0]
        return self.local_logits


def _alloc_tensor(shape, dtype, device=None):
    return torch.empty(shape, dtype=dtype, device=device)


def test_qwen4_lm_head_gather_restores_token_major_vocab_order(monkeypatch):
    rank0_logits = torch.tensor(
        [[1, 2, 3], [4, 5, 6]],
        dtype=torch.bfloat16,
    )
    rank1_logits = torch.tensor(
        [[7, 8, 9], [10, 11, 12]],
        dtype=torch.bfloat16,
    )

    def fake_all_gather_into_tensor(output, input, group, async_op):
        assert group == "tp-group"
        assert async_op is False
        output[:2].copy_(input)
        output[2:].copy_(rank1_logits)

    monkeypatch.setattr(
        "lightllm.models.qwen4_exp.layer_infer.post_layer_infer.all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )
    post_infer = SimpleNamespace(
        tp_world_size_=2,
        alloc_tensor=_alloc_tensor,
        _norm=lambda hidden, infer_state, layer_weight: hidden,
    )
    layer_weight = SimpleNamespace(
        lm_head_weight_=_FakeLMHead(rank0_logits, vocab_size=6)
    )
    infer_state = SimpleNamespace(dist_group="tp-group")

    actual = Qwen4ExpPostLayerInfer._lm_head_and_gather(
        post_infer,
        hidden=torch.empty((2, 4), dtype=torch.bfloat16),
        token_num=2,
        layer_weight=layer_weight,
        infer_state=infer_state,
    )

    expected = torch.tensor(
        [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen4_lm_head_single_rank_keeps_batch_major_layout():
    local_logits = torch.tensor(
        [[1, 2, 3], [4, 5, 6]],
        dtype=torch.bfloat16,
    )
    post_infer = SimpleNamespace(
        tp_world_size_=1,
        alloc_tensor=_alloc_tensor,
        _norm=lambda hidden, infer_state, layer_weight: hidden,
    )
    layer_weight = SimpleNamespace(
        lm_head_weight_=_FakeLMHead(local_logits, vocab_size=3)
    )

    actual = Qwen4ExpPostLayerInfer._lm_head_and_gather(
        post_infer,
        hidden=torch.empty((2, 4), dtype=torch.bfloat16),
        token_num=2,
        layer_weight=layer_weight,
        infer_state=SimpleNamespace(),
    )

    torch.testing.assert_close(actual, local_logits.float(), rtol=0, atol=0)
