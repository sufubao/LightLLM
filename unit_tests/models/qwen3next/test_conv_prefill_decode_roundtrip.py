import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _eager_conv_update(x_seq, conv_state, weight, bias, activation):
    # x_seq: (dim, seqlen) tokens to roll in, conv_state: (dim, width-1) history
    state = conv_state.clone()
    outs = []
    for t in range(x_seq.shape[1]):
        window = torch.cat([state, x_seq[:, t : t + 1]], dim=1)  # (dim, width)
        y = (window * weight).sum(dim=1)
        if bias is not None:
            y = y + bias
        if activation in ("silu", "swish"):
            y = torch.nn.functional.silu(y)
        outs.append(y)
        state = window[:, 1:]
    return torch.stack(outs, dim=1), state


@pytest.mark.parametrize("S", [1, 2, 3])
def test_prefill_writes_first_columns_then_decode_reads_them(S):
    from lightllm.models.qwen3next.triton_kernel.causal_conv1d import causal_conv1d_fn
    from lightllm.models.qwen3next.triton_kernel.causal_conv1d_spec import causal_conv1d_update

    torch.manual_seed(0)
    dim, width = 64, 4
    prefill_len = 7
    state_len = (width - 1) + S  # widened slot
    device, dtype = "cuda", torch.float32

    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)

    # ---- PREFILL: populate one widened conv slot from a fresh (no initial state) sequence ----
    conv_states = torch.zeros(1, dim, state_len, device=device, dtype=dtype)
    x_prefill = torch.randn(dim, prefill_len, device=device, dtype=dtype)  # (dim, total_tokens)
    causal_conv1d_fn(
        x_prefill.clone(),
        weight,
        bias=bias,
        query_start_loc=torch.tensor([0, prefill_len], dtype=torch.int32, device=device),
        cache_indices=torch.zeros(1, dtype=torch.int32, device=device),
        has_initial_state=torch.zeros(1, dtype=torch.bool, device=device),
        conv_states=conv_states,
        activation="silu",
    )

    # Contract (a): committed state lands in the FIRST width-1 columns; widened tail untouched.
    committed_hist = conv_states[0, :, : width - 1].clone()
    expected_hist = x_prefill[:, -(width - 1) :]  # trailing window for a fresh causal conv
    torch.testing.assert_close(committed_hist, expected_hist, rtol=1e-3, atol=1e-3)
    if state_len > width - 1:
        assert torch.count_nonzero(conv_states[0, :, width - 1 :]) == 0, "widened tail must be untouched by prefill"

    # ---- FIRST DECODE: verify reads at offset accept_len-1 == 0 -> columns [0:width-1] ----
    seqlen = S + 1
    x_decode = torch.randn(seqlen, dim, device=device, dtype=dtype)
    out = causal_conv1d_update(
        x_decode.clone(),
        conv_states,
        weight,
        bias=bias,
        activation="silu",
        conv_state_indices=torch.zeros(1, dtype=torch.int32, device=device),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),  # offset 0
        query_start_loc=torch.tensor([0, seqlen], dtype=torch.int32, device=device),
    )

    # Contract (b): decode output must match an eager conv seeded from the prefill-written history.
    ref_out, _ = _eager_conv_update(x_decode.t(), committed_hist, weight, bias, "silu")
    torch.testing.assert_close(out.t(), ref_out, rtol=1e-3, atol=1e-3)
