import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _eager_conv_update(x_seq, conv_state, weight, bias, activation):
    # x_seq: (dim, seqlen) tokens to roll in, conv_state: (dim, width-1) history
    dim, width = weight.shape
    state = conv_state.clone()  # (dim, width-1)
    outs = []
    for t in range(x_seq.shape[1]):
        window = torch.cat([state, x_seq[:, t : t + 1]], dim=1)  # (dim, width)
        y = (window * weight).sum(dim=1)  # depthwise conv
        if bias is not None:
            y = y + bias
        if activation in ("silu", "swish"):
            y = torch.nn.functional.silu(y)
        outs.append(y)
        state = window[:, 1:]  # slide
    return torch.stack(outs, dim=1), state


@pytest.mark.parametrize("S", [0, 1, 2, 3])
def test_spec_conv_matches_eager_after_partial_accept(S):
    from lightllm.models.qwen3next.triton_kernel.causal_conv1d_spec import causal_conv1d_update

    torch.manual_seed(0)
    dim, width = 64, 4
    seqlen = S + 1
    state_len = (width - 1) + S
    device = "cuda"
    dtype = torch.float32

    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)

    conv_state = torch.zeros(1, dim, state_len, device=device, dtype=dtype)
    committed_hist = torch.randn(dim, width - 1, device=device, dtype=dtype)
    conv_state[0, :, : width - 1] = committed_hist

    x = torch.randn(seqlen, dim, device=device, dtype=dtype)  # candidate tokens

    out = causal_conv1d_update(
        x.clone(),
        conv_state,
        weight,
        bias=bias,
        activation="silu",
        conv_state_indices=torch.zeros(1, dtype=torch.int32, device=device),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),  # fresh: read offset 0
        query_start_loc=torch.tensor([0, seqlen], dtype=torch.int32, device=device),
    )

    ref_out, _ = _eager_conv_update(x.t(), committed_hist, weight, bias, "silu")
    torch.testing.assert_close(out.t(), ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("S", [1, 2, 3])
def test_spec_conv_reads_from_partial_accept_offset(S):
    # Exercise the nonzero read offset: num_accepted_tokens=2 -> read offset 1.
    # The widened slot front-loads a STALE token then the real committed history;
    # the kernel must read history starting at (num_accepted_tokens-1)==1, i.e.
    # conv_state[:, 1:width], NOT the stale token at index 0.
    from lightllm.models.qwen3next.triton_kernel.causal_conv1d_spec import causal_conv1d_update

    torch.manual_seed(0)
    dim, width = 64, 4
    seqlen = S + 1
    state_len = (width - 1) + S
    device = "cuda"
    dtype = torch.float32

    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)

    conv_state = torch.zeros(1, dim, state_len, device=device, dtype=dtype)
    # tokens [0 .. width-1] hold [stale, h1, h2, ...]: a stale front token then history
    seed = torch.randn(dim, width, device=device, dtype=dtype)
    conv_state[0, :, :width] = seed
    stale_front = conv_state[0, :, :width].clone()  # snapshot of the seeded window

    x = torch.randn(seqlen, dim, device=device, dtype=dtype)  # candidate tokens

    out = causal_conv1d_update(
        x.clone(),
        conv_state,
        weight,
        bias=bias,
        activation="silu",
        conv_state_indices=torch.zeros(1, dtype=torch.int32, device=device),
        num_accepted_tokens=2 * torch.ones(1, dtype=torch.int32, device=device),  # read offset 1
        query_start_loc=torch.tensor([0, seqlen], dtype=torch.int32, device=device),
    )

    # Eager reference starts from the offset-1 window: committed history excluding
    # the stale front token == conv_state[:, 1:width].
    committed_hist = stale_front[:, 1:width]
    ref_out, _ = _eager_conv_update(x.t(), committed_hist, weight, bias, "silu")
    torch.testing.assert_close(out.t(), ref_out, rtol=1e-3, atol=1e-3)


def test_spec_conv_varlen_update_is_cuda_graph_capturable():
    from lightllm.models.qwen3next.triton_kernel.causal_conv1d_spec import causal_conv1d_update

    torch.manual_seed(0)
    dim, width, S = 64, 4, 1
    seqlen = S + 1
    state_len = (width - 1) + S
    device = "cuda"
    dtype = torch.float32

    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_state = torch.zeros(1, dim, state_len, device=device, dtype=dtype)
    x = torch.randn(seqlen, dim, device=device, dtype=dtype)
    conv_state_indices = torch.zeros(1, dtype=torch.int32, device=device)
    num_accepted_tokens = torch.ones(1, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

    # Compile/warm the Triton kernel before capture; the regression is the wrapper's
    # host sync on query_start_loc during capture, not first-use compilation.
    causal_conv1d_update(
        x.clone(),
        conv_state,
        weight,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    static_x = x.clone()
    with torch.cuda.graph(graph):
        causal_conv1d_update(
            static_x,
            conv_state,
            weight,
            bias=bias,
            activation="silu",
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
        )
