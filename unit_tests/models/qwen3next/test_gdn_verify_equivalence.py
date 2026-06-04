import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("S", [1, 2, 3])
def test_gdn_verify_state_equals_sequential_decode(S):
    from lightllm.models.qwen3next.triton_kernel.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    torch.manual_seed(0)
    HV, K, V = 4, 16, 16
    T = S + 1
    device = "cuda"

    def rand_qkv(t):
        q = torch.randn(1, t, HV, K, device=device)
        k = torch.nn.functional.normalize(torch.randn(1, t, HV, K, device=device), dim=-1)
        v = torch.randn(1, t, HV, V, device=device)
        g = torch.nn.functional.logsigmoid(torch.rand(1, t, HV, device=device))
        beta = torch.rand(1, t, HV, device=device).sigmoid()
        return q, k, v, g, beta

    q, k, v, g, beta = rand_qkv(T)

    ref_state = torch.zeros(1, HV, K, V, device=device)
    for t in range(T):
        _, ref_state = fused_recurrent_gated_delta_rule(
            q=q[:, t : t + 1],
            k=k[:, t : t + 1],
            v=v[:, t : t + 1],
            g=g[:, t : t + 1],
            beta=beta[:, t : t + 1],
            initial_state=ref_state,
            inplace_final_state=False,
        )

    block = torch.zeros(T, HV, K, V, device=device)
    ssm_idx = torch.arange(T, device=device).view(1, T)
    fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=block,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, T], dtype=torch.long, device=device),
        ssm_state_indices=ssm_idx,
        ssm_state_write_indices=ssm_idx,
        num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(block[T - 1], ref_state[0], rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("S", [1, 2, 3])
def test_gdn_verify_output_equals_sequential_decode_fused(S):
    """H1: the LIVE verify combination - varlen + FUSED gating (A_log/dt_bias/a_raw/b_raw)
    + spec-decode - must produce per-position OUTPUT o[t] identical to running the proven
    T=1 decode recurrence sequentially. The original test only checked the final SSM state
    with EXPLICIT g/beta; it never verified o[t] nor the fused-gating path that
    _gdn_verify_kernel actually uses."""
    from lightllm.models.qwen3next.triton_kernel.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    torch.manual_seed(0)
    HV, K, V = 4, 16, 16
    H = HV
    T = S + 1
    device = "cuda"

    q = torch.randn(1, T, H, K, device=device)
    k = torch.nn.functional.normalize(torch.randn(1, T, H, K, device=device), dim=-1)
    v = torch.randn(1, T, HV, V, device=device)
    # Raw gating inputs (pre-activation), exactly as the model feeds the fused path.
    a_raw = torch.randn(T, HV, device=device)
    b_raw = torch.randn(T, HV, device=device)
    A_log = torch.randn(HV, device=device)
    dt_bias = torch.randn(HV, device=device)

    # Reference: sequential T=1 decode through the proven non-varlen fused path.
    ref_state = torch.zeros(1, HV, K, V, device=device)
    ref_o = torch.zeros(T, HV, V, device=device)
    for t in range(T):
        o_t, ref_state = fused_recurrent_gated_delta_rule(
            q=q[:, t : t + 1],
            k=k[:, t : t + 1],
            v=v[:, t : t + 1],
            initial_state=ref_state,
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            a_raw=a_raw[t : t + 1],
            b_raw=b_raw[t : t + 1],
        )
        ref_o[t] = o_t[0, 0]

    # Verify path: single varlen call with fused gating + spec-decode indices,
    # mirroring _gdn_verify_kernel for a single request, num_accepted=1.
    block = torch.zeros(T, HV, K, V, device=device)
    ssm_idx = torch.arange(T, device=device).view(1, T)
    o, _ = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        initial_state=block,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, T], dtype=torch.long, device=device),
        ssm_state_indices=ssm_idx,
        ssm_state_write_indices=ssm_idx,
        num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),
        use_qk_l2norm_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
        a_raw=a_raw,
        b_raw=b_raw,
    )
    o = o.view(T, HV, V)
    torch.testing.assert_close(o, ref_o, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(block[T - 1], ref_state[0], rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_accepted", [1, 2])
def test_gdn_verify_reads_committed_slot_by_num_accepted(num_accepted):
    """The verify kernel must read the per-request initial state from the SSM block
    slot at offset (num_accepted-1) -- i.e. the state committed after the previous
    step's last accepted token. This is the read path exercised by the FIRST decode
    after an accept-`num_accepted` step. A decoy is written into the OTHER block slot
    to prove the kernel reads the correct one and ignores the rest of the (S+1) block."""
    from lightllm.models.qwen3next.triton_kernel.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    torch.manual_seed(0)
    HV, K, V = 4, 16, 16
    S = 1
    T = S + 1
    device = "cuda"

    q = torch.randn(1, T, HV, K, device=device)
    k = torch.nn.functional.normalize(torch.randn(1, T, HV, K, device=device), dim=-1)
    v = torch.randn(1, T, HV, V, device=device)
    a_raw = torch.randn(T, HV, device=device)
    b_raw = torch.randn(T, HV, device=device)
    A_log = torch.randn(HV, device=device)
    dt_bias = torch.randn(HV, device=device)

    # (S+1) block: the committed slot is (num_accepted-1); the others hold decoys
    # that MUST NOT be read.
    block = torch.randn(T, HV, K, V, device=device) * 5.0
    committed = torch.randn(1, HV, K, V, device=device)
    block[num_accepted - 1] = committed[0]

    ref_state = committed.clone()
    ref_o = torch.zeros(T, HV, V, device=device)
    for t in range(T):
        o_t, ref_state = fused_recurrent_gated_delta_rule(
            q=q[:, t : t + 1],
            k=k[:, t : t + 1],
            v=v[:, t : t + 1],
            initial_state=ref_state,
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            a_raw=a_raw[t : t + 1],
            b_raw=b_raw[t : t + 1],
        )
        ref_o[t] = o_t[0, 0]

    blk = block.clone()
    ssm_idx = torch.arange(T, device=device).view(1, T)
    o, _ = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        initial_state=blk,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, T], dtype=torch.long, device=device),
        ssm_state_indices=ssm_idx,
        ssm_state_write_indices=ssm_idx,
        num_accepted_tokens=torch.tensor([num_accepted], dtype=torch.int32, device=device),
        use_qk_l2norm_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
        a_raw=a_raw,
        b_raw=b_raw,
    )
    o = o.view(T, HV, V)
    torch.testing.assert_close(o, ref_o, rtol=2e-2, atol=2e-2)
