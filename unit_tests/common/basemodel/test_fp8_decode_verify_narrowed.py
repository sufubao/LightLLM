import types
import torch
import pytest

import lightllm.common.basemodel.attention.fa3.fp8 as fp8_mod
from lightllm.common.basemodel.attention.fa3.fp8 import Fp8Fa3DecodeAttState


def _make_verify_state(n_real, mtp_size, head_num=2, head_dim=8):
    """Build an Fp8Fa3DecodeAttState as init_state would leave it in MTP-verify mode,
    bypassing init_state. b_att_seq_len/page_table are NARROW (n_real); infer_state.b_seq_len
    is the FULL expanded tensor (n_real*mtp_size) that must NOT be used as cache_seqlens."""
    state = object.__new__(Fp8Fa3DecodeAttState)
    batch = n_real * mtp_size
    state.b_att_seq_len = torch.full((n_real,), 16, dtype=torch.int32)
    state.page_table = torch.zeros((n_real, 16), dtype=torch.int32)
    state.cu_seqlens_q = torch.arange(0, (n_real + 1) * mtp_size, mtp_size, dtype=torch.int32)
    state.cu_seqlens_k = torch.zeros((n_real + 1,), dtype=torch.int32)
    state.decode_max_q_seq_len = mtp_size
    state.infer_state = types.SimpleNamespace(
        b_seq_len=torch.full((batch,), 16, dtype=torch.int32),
        batch_size=batch,
    )
    # k/v descale sized per real request (att_batch_size), indexed by layer
    state.k_descale = torch.ones((1, n_real, head_num))
    state.v_descale = torch.ones((1, n_real, head_num))
    state.backend = types.SimpleNamespace(_find_layer_index=lambda k, v, att_state: 0)
    return state, batch


def test_fp8_decode_uses_narrowed_cache_seqlens_and_causal(monkeypatch):
    n_real, mtp_size, head_num, head_dim = 3, 4, 2, 8
    state, batch = _make_verify_state(n_real, mtp_size, head_num, head_dim)

    captured = {}

    def fake_flash(**kwargs):
        captured.update(kwargs)
        q = kwargs["q"]
        return torch.zeros((q.shape[0], q.shape[1], q.shape[2]))

    def fake_quant(x, use_per_token_if_dynamic=True):
        return x, torch.ones((x.shape[0], 1))

    monkeypatch.setattr(fp8_mod, "flash_attn_with_kvcache", fake_flash)
    monkeypatch.setattr(fp8_mod, "scaled_fp8_quant", fake_quant)

    q = torch.randn((batch, head_num, head_dim))
    k = torch.randn((batch, head_num, head_dim))
    v = torch.randn((batch, head_num, head_dim))

    state._fp8_decode_att(q=q, k=k, v=v)

    # The KV-side seqlens must be the NARROW per-real-request tensor, matching page_table rows.
    assert captured["cache_seqlens"] is state.b_att_seq_len
    assert captured["cache_seqlens"].shape[0] == n_real
    assert captured["cache_seqlens"].shape[0] == captured["page_table"].shape[0]
    # Verify decode must be causal, like the non-fp8 sibling.
    assert captured["causal"] is True
