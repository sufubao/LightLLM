import torch

from lightllm.common.basemodel.layer_weights.meta_weights.embedding_weight import (
    LMHeadWeight,
)


def _make_lm_head(weight: torch.Tensor) -> LMHeadWeight:
    lm_head = object.__new__(LMHeadWeight)
    lm_head.weight = weight
    return lm_head


def test_lm_head_batch_major_forward_matches_linear_layout():
    weight = torch.randn((11, 7), dtype=torch.float32)
    input = torch.randn((3, 7), dtype=torch.float32)
    expected = torch.nn.functional.linear(input, weight)

    lm_head = _make_lm_head(weight)
    actual = lm_head.batch_major_forward(input)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_lm_head_batch_major_forward_honors_supplied_output():
    weight = torch.randn((11, 7), dtype=torch.float32)
    input = torch.randn((3, 7), dtype=torch.float32)
    out = torch.empty((3, 11), dtype=torch.float32)
    lm_head = _make_lm_head(weight)

    result = lm_head.batch_major_forward(input, out=out)

    assert result is out
    torch.testing.assert_close(
        result, torch.nn.functional.linear(input, weight), rtol=0, atol=0
    )
