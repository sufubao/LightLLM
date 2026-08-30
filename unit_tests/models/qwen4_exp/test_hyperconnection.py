import pytest
import torch
import torch.nn.functional as F

from lightllm.models.qwen4_exp.layer_weights import (
    hyperconnection as hyperconnection_weight,
)
from lightllm.models.qwen4_exp.hyperconnection import (
    grouped_gemma_rmsnorm,
    hyperconnection_combine,
    hyperconnection_mix,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_gemma_rmsnorm_matches_reference(dtype):
    torch.manual_seed(17)
    tokens, hc_count, hidden_size = 7, 4, 32
    states = torch.randn(tokens, hc_count * hidden_size, dtype=dtype)
    weight = torch.randn(hc_count * hidden_size, dtype=dtype)

    actual = grouped_gemma_rmsnorm(states, weight, hidden_size=hidden_size, eps=1e-6)

    grouped = states.float().view(tokens, hc_count, hidden_size)
    expected = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected * (1 + weight.float().view(hc_count, hidden_size))
    expected = expected.flatten(-2).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gated_residual_matches_qwen4_reference(dtype):
    torch.manual_seed(29)
    tokens, hc_count, hidden_size, lowrank = 9, 4, 24, 13
    hyper_size = hc_count * hidden_size
    states = torch.randn(tokens, hyper_size, dtype=dtype)
    norm_weight = torch.randn(hyper_size, dtype=dtype)
    down_weight = torch.randn(lowrank, hyper_size, dtype=dtype)
    up_weight = torch.randn(hyper_size, lowrank, dtype=dtype)
    inject_weight = torch.randn(hc_count, hyper_size, dtype=dtype)
    block_output = torch.randn(tokens, hidden_size, dtype=dtype)

    normalized = grouped_gemma_rmsnorm(
        states, norm_weight, hidden_size=hidden_size, eps=1e-6
    )
    lowrank_states = F.silu(F.linear(normalized, down_weight) / hc_count)
    gate_logits = F.linear(lowrank_states, up_weight)
    mixed = hyperconnection_mix(normalized, gate_logits, hc_count=hc_count)
    injection_logits = F.linear(normalized, inject_weight)
    combined = hyperconnection_combine(
        states, block_output, injection_logits, hc_count=hc_count
    )

    normalized_ref = normalized.view(tokens, hc_count, hidden_size)
    gate_ref = torch.sigmoid(gate_logits).view(tokens, hc_count, hidden_size)
    mixed_ref = (normalized_ref * gate_ref).mean(dim=1)
    injection_ref = 2 * torch.sigmoid(injection_logits.float() / hc_count)
    combined_ref = states.float().view(tokens, hc_count, hidden_size)
    combined_ref = combined_ref + block_output.float().unsqueeze(
        1
    ) * injection_ref.unsqueeze(-1)

    atol = 0 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(mixed, mixed_ref, rtol=0, atol=atol)
    torch.testing.assert_close(
        combined, combined_ref.flatten(-2).to(dtype), rtol=0, atol=atol
    )


def test_hyperconnection_shape_validation():
    with pytest.raises(ValueError, match="not divisible"):
        grouped_gemma_rmsnorm(
            torch.zeros(2, 15), torch.zeros(15), hidden_size=4, eps=1e-6
        )
    with pytest.raises(ValueError, match="gate shape"):
        hyperconnection_mix(torch.zeros(2, 16), torch.zeros(2, 8), hc_count=4)
    with pytest.raises(ValueError, match="block output shape"):
        hyperconnection_combine(
            torch.zeros(2, 16), torch.zeros(2, 5), torch.zeros(2, 4), hc_count=4
        )


def test_gated_residual_weight_zero_pads_merged_projection(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")

    class StubNormWeight:
        def __init__(self, **kwargs):
            pass

    class StubPack:
        def __init__(self, out_dim, in_dim):
            self.weight = torch.full((1, 1), 7.0)
            self.load_ok = [False, True, True]

    class StubRowMMWeight:
        def __init__(self, *, in_dim, out_dims, weight_names, **kwargs):
            self.out_dims = out_dims
            self.weight_names = weight_names
            self.mm_param_list = [StubPack(out_dim, in_dim) for out_dim in out_dims]

    monkeypatch.setattr(hyperconnection_weight, "RMSNormWeight", StubNormWeight)
    monkeypatch.setattr(hyperconnection_weight, "ROWMMWeight", StubRowMMWeight)

    weight = hyperconnection_weight.Qwen4ExpGatedResidualWeight(
        prefix="model.layers.0.attn_hyper_connection",
        hidden_size=24,
        hc_count=4,
        hc_lowrank=13,
        data_type=torch.bfloat16,
    )

    merged = weight.input_mix_weight_down_block_inject
    assert weight.padding_size == 15
    assert merged.out_dims == [13, 4, 15]
    assert merged.weight_names[-1].endswith(".__padding__")
    assert torch.count_nonzero(merged.mm_param_list[-1].weight) == 0
    assert merged.mm_param_list[-1].load_ok[0]

    wide_weight = hyperconnection_weight.Qwen4ExpGatedResidualWeight(
        prefix="model.layers.0.attn_hyper_connection",
        hidden_size=2560,
        hc_count=4,
        hc_lowrank=320,
        data_type=torch.bfloat16,
    )
    wide_merged = wide_weight.input_mix_weight_down_block_inject
    assert wide_weight.padding_size == 124
    assert wide_merged.out_dims == [320, 4, 124]
