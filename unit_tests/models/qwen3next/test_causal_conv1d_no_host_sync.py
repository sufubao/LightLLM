import pathlib
import re

import pytest
import torch

SPEC = pathlib.Path(__file__).resolve().parents[3] / "lightllm/models/qwen3next/triton_kernel/causal_conv1d_spec.py"


def test_no_query_start_loc_item_sync_in_seqlen():
    src = SPEC.read_text()
    # The per-step D2H sync on query_start_loc must be gone from the seqlen computation (#8a).
    assert not re.search(r"query_start_loc\[1:\]\s*-\s*query_start_loc\[:-1\]\)\.max\(\)\.item\(\)", src), (
        "causal_conv1d_update still computes seqlen via a .item() D2H sync on query_start_loc; "
        "use x.size(0) // batch unconditionally (#8a)."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_eager_varlen_multi_request_matches_eager_reference():
    from lightllm.models.qwen3next.triton_kernel.causal_conv1d_spec import causal_conv1d_update

    torch.manual_seed(0)
    dim, width, S = 64, 4, 2
    seqlen = S + 1
    n_req = 3
    state_len = (width - 1) + S
    device, dtype = "cuda", torch.float32

    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)

    conv_state = torch.zeros(n_req, dim, state_len, device=device, dtype=dtype)
    hist = torch.randn(n_req, dim, width - 1, device=device, dtype=dtype)
    conv_state[:, :, : width - 1] = hist

    x = torch.randn(n_req * seqlen, dim, device=device, dtype=dtype)  # packed varlen, uniform S+1
    query_start_loc = torch.arange(0, (n_req + 1) * seqlen, seqlen, dtype=torch.int32, device=device)

    out = causal_conv1d_update(
        x.clone(),
        conv_state,
        weight,
        bias=bias,
        activation="silu",
        conv_state_indices=torch.arange(n_req, dtype=torch.int32, device=device),
        num_accepted_tokens=torch.ones(n_req, dtype=torch.int32, device=device),  # offset 0
        query_start_loc=query_start_loc,
    )

    def _eager(x_seq, h):
        state = h.clone()
        outs = []
        for t in range(x_seq.shape[1]):
            window = torch.cat([state, x_seq[:, t : t + 1]], dim=1)
            y = torch.nn.functional.silu((window * weight).sum(dim=1) + bias)
            outs.append(y)
            state = window[:, 1:]
        return torch.stack(outs, dim=1)

    for r in range(n_req):
        xr = x[r * seqlen : (r + 1) * seqlen].t()  # (dim, seqlen)
        ref = _eager(xr, hist[r])
        got = out[r * seqlen : (r + 1) * seqlen].t()
        torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)
