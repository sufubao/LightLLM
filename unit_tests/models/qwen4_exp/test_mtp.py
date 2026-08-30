from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.models.qwen4_exp.hyperconnection import grouped_gemma_rmsnorm
from lightllm.models.qwen4_exp_mtp.layer_infer.pre_layer_infer import (
    Qwen4ExpMTPPreLayerInfer,
)
from lightllm.models.qwen4_exp_mtp.layer_infer.post_layer_infer import (
    Qwen4ExpMTPPostLayerInfer,
    select_global_argmax,
)
from lightllm.models.qwen4_exp_mtp.layer_weights.transformer_layer_weight import (
    rename_qwen4_mtp_layer_weight_keys,
)
from lightllm.models.qwen4_exp_mtp.model import Qwen4ExpMTPModel


class _Linear:
    def __init__(self, weight):
        self.weight = weight

    def mm(self, value):
        return F.linear(value, self.weight)


def test_qwen4_mtp_residual_linear_shared_matches_reference():
    torch.manual_seed(91)
    rows, hidden_size, hc_count = 5, 12, 4
    token_embeddings = torch.randn(rows, hidden_size, dtype=torch.bfloat16)
    target_hidden = torch.randn(rows, hidden_size * hc_count, dtype=torch.bfloat16)
    embedding_norm = torch.randn(hidden_size, dtype=torch.bfloat16)
    hidden_norm = torch.randn(hidden_size * hc_count, dtype=torch.bfloat16)
    embedding_fc = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16)
    hidden_fc = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16)
    weights = SimpleNamespace(
        pre_fc_norm_embedding=SimpleNamespace(weight=embedding_norm),
        pre_fc_norm_hidden=SimpleNamespace(weight=hidden_norm),
        fc_embedding=_Linear(embedding_fc),
        fc_hidden=_Linear(hidden_fc),
    )
    state = SimpleNamespace(mtp_draft_input_hiddens=target_hidden)
    infer = Qwen4ExpMTPPreLayerInfer.__new__(Qwen4ExpMTPPreLayerInfer)
    infer.eps_ = 1e-6
    infer.hidden_size = hidden_size
    infer.hc_count = hc_count

    actual = infer._mtp_fuse(token_embeddings, state, weights)

    expected_embedding = F.linear(
        grouped_gemma_rmsnorm(
            token_embeddings,
            embedding_norm,
            hidden_size=hidden_size,
            eps=1e-6,
        ),
        embedding_fc,
    )
    expected_hidden = F.linear(
        grouped_gemma_rmsnorm(
            target_hidden,
            hidden_norm,
            hidden_size=hidden_size,
            eps=1e-6,
        ).view(-1, hidden_size),
        hidden_fc,
    ).view(rows, hc_count, hidden_size)
    expected = (expected_hidden + expected_embedding.unsqueeze(1)).flatten(-2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen4_mtp_requires_multi_stream_target_hidden():
    infer = Qwen4ExpMTPPreLayerInfer.__new__(Qwen4ExpMTPPreLayerInfer)
    infer.eps_ = 1e-6
    infer.hidden_size = 8
    infer.hc_count = 4
    state = SimpleNamespace(mtp_draft_input_hiddens=torch.zeros(2, 8))

    with pytest.raises(ValueError, match="expects hidden width 32"):
        infer._mtp_fuse(torch.zeros(2, 8), state, SimpleNamespace())


def test_qwen4_mtp_layer_key_remapping_is_scoped():
    mtp = torch.tensor([1.0])
    main = torch.tensor([2.0])
    pre = torch.tensor([3.0])
    weights = {
        "mtp.layers.0.self_attn.q_proj.weight": mtp,
        "model.layers.0.self_attn.q_proj.weight": main,
        "mtp.fc_embedding.weight": pre,
    }

    rename_qwen4_mtp_layer_weight_keys(weights)

    assert weights["model.layers.0.self_attn.q_proj.weight"] is mtp
    assert weights["mtp.fc_embedding.weight"] is pre
    assert "mtp.layers.0.self_attn.q_proj.weight" not in weights


def test_qwen4_mtp_warmup_hidden_uses_all_hc_streams(monkeypatch):
    model = Qwen4ExpMTPModel.__new__(Qwen4ExpMTPModel)
    model.config = {"hidden_size": 12, "hc_count": 4}
    model.data_type = torch.bfloat16
    monkeypatch.setattr(torch, "randn", lambda *shape, **kwargs: (shape, kwargs))

    generated = model._gen_special_model_input(7)["mtp_draft_input_hiddens"]

    assert generated[0] == (7, 48)
    assert generated[1]["dtype"] == torch.bfloat16
    assert generated[1]["device"] == "cuda"


def test_qwen4_mtp_global_argmax_matches_full_vocab_with_rank_ties():
    # Rank-major pairs. Equal maxima must choose the earlier TP rank, matching
    # torch.argmax over the concatenated vocabulary.
    gathered_winners = torch.tensor(
        [
            [[4.0, 2.0], [8.0, 5.0], [3.0, 7.0]],
            [[9.0, 11.0], [8.0, 13.0], [6.0, 17.0]],
        ]
    )

    actual = select_global_argmax(gathered_winners)

    torch.testing.assert_close(actual, torch.tensor([11, 5, 17]))


def test_qwen4_mtp_local_argmax_consumes_batch_major_logits(monkeypatch):
    local_logits = torch.tensor(
        [[1.0, 5.0, 3.0], [9.0, 2.0, 1.0]],
        dtype=torch.bfloat16,
    )

    class LMHead:
        tp_vocab_start_id = 10

        def batch_major_forward(self, input, alloc_func):
            assert input.shape[0] == 2
            return local_logits

    class Collector:
        def add_mtp_outputs(self, **kwargs):
            self.outputs = kwargs

    def fake_all_gather_into_tensor(output, input, group, async_op):
        output[:2].copy_(input)
        output[2:].copy_(torch.tensor([[6.0, 20.0], [8.0, 22.0]]))

    monkeypatch.setattr(
        "lightllm.models.qwen4_exp_mtp.layer_infer.post_layer_infer."
        "all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )
    collector = Collector()
    infer_state = SimpleNamespace(
        dist_group="tp-group",
        hidden_collector=collector,
    )
    post_infer = SimpleNamespace(
        tp_world_size_=2,
        alloc_tensor=lambda shape, dtype: torch.empty(shape, dtype=dtype),
        _slice_get_last_input=lambda hidden, state: (hidden, 2),
        _norm=lambda hidden, state, weight: hidden,
    )

    output = Qwen4ExpMTPPostLayerInfer._local_argmax_token_forward(
        post_infer,
        input_embdings=torch.empty((2, 4)),
        infer_state=infer_state,
        layer_weight=SimpleNamespace(lm_head_weight_=LMHead()),
    )

    torch.testing.assert_close(
        collector.outputs["draft_token_ids"], torch.tensor([20, 10])
    )
    assert collector.outputs["confidence_logits"] is None
    assert output.shape == (2, 1)
