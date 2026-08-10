from types import SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.attention.linear.gdn as gdn
import lightllm.common.basemodel.triton_kernel.linear_att.gdn_prefill_backend as gdn_prefill_backend


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_prefill_casts_final_state_to_cache_dtype(monkeypatch, cache_dtype):
    ssm_states = torch.zeros((1, 1, 2, 2), dtype=cache_dtype)
    final_state = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(gdn, "fused_gdn_gating", lambda _log, a, b, _bias: (a, b))
    monkeypatch.setattr(gdn, "causal_conv1d_fn", lambda mixed, *args, **kwargs: mixed)
    qkv = torch.zeros((1, 3), dtype=cache_dtype)
    q = torch.zeros((1, 1, 1, 1), dtype=cache_dtype)
    backend = SimpleNamespace(
        mtp_step=0,
        activation="silu",
        ssm_state_dtype=cache_dtype,
        _rearrange_mixed_qkv=lambda mixed: (q, q, q),
        _gdn_prefill_backend=lambda *args, **kwargs: (None, final_state),
    )
    state = gdn.LinearAttPrefillAttState(
        backend=backend,
        infer_state=SimpleNamespace(
            b1_cu_q_seq_len=torch.tensor([0, 1], dtype=torch.int32),
            b_ready_cache_len=0,
        ),
    )
    state.b_conv_buffer_idx = torch.tensor([0], dtype=torch.int64)
    state.b_ssm_buffer_idx = torch.tensor([0], dtype=torch.int64)
    layer_weight = SimpleNamespace(
        linear_A_log=SimpleNamespace(weight=None),
        linear_dt_bias=SimpleNamespace(weight=None),
        linear_conv1d=SimpleNamespace(mm_param=SimpleNamespace(weight=None), bias=None),
    )

    state._gdn_prefill_kernel(
        qkv,
        torch.zeros((1, 3), dtype=cache_dtype),
        ssm_states,
        torch.zeros((1, 1), dtype=cache_dtype),
        torch.zeros((1, 1), dtype=cache_dtype),
        state.infer_state,
        layer_weight,
    )

    assert ssm_states.dtype == cache_dtype
    assert torch.equal(ssm_states, final_state.to(cache_dtype))


def test_flashqla_backend_falls_back_for_unsupported_call(monkeypatch):
    monkeypatch.setattr(
        gdn_prefill_backend,
        "_fla_chunk_gated_delta_rule",
        lambda **kwargs: ("fla", kwargs),
    )
    backend = gdn_prefill_backend.FlashQlaGdnPrefillBackend(lambda **kwargs: ("flashqla", kwargs))
    q = torch.zeros((1, 1, 1, 128), dtype=torch.float32)

    result, _ = backend(q=q, k=q, v=q)

    assert result == "fla"
    assert not backend.initialized


def test_flashqla_backend_disables_after_initialization_failure(monkeypatch):
    calls = {"flashqla": 0, "fla": 0}

    def flashqla(**kwargs):
        calls["flashqla"] += 1
        raise RuntimeError("compile failed")

    def fla(**kwargs):
        calls["fla"] += 1
        return "fla", None

    monkeypatch.setattr(gdn_prefill_backend, "_fla_chunk_gated_delta_rule", fla)
    backend = gdn_prefill_backend.FlashQlaGdnPrefillBackend(flashqla)
    monkeypatch.setattr(backend, "supports", lambda *args: True)
    q = torch.zeros((1, 1, 1, 128), dtype=torch.bfloat16)

    assert backend(q=q, k=q, v=q)[0] == "fla"
    assert backend(q=q, k=q, v=q)[0] == "fla"
    assert calls == {"flashqla": 1, "fla": 2}


def test_flashqla_backend_respects_disable_env(monkeypatch):
    monkeypatch.setenv("FLA_FLASH_QLA", "0")

    available, reason = gdn_prefill_backend.FlashQlaGdnPrefillBackend.is_available(128, 128)

    assert not available
    assert reason == "disabled by FLA_FLASH_QLA=0"
