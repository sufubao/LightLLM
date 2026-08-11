import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.attention.linear.gdn as gdn
import lightllm.common.basemodel.attention.linear.create_utils as linear_create_utils


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
        _chunk_gated_delta_rule=lambda *args, **kwargs: (None, final_state),
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


@pytest.fixture
def linear_model(monkeypatch):
    monkeypatch.setattr(
        gdn,
        "get_env_start_args",
        lambda: SimpleNamespace(linear_att_ssm_data_type="float32"),
    )
    return SimpleNamespace(
        config={
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        },
        tp_world_size_=2,
        data_type=torch.bfloat16,
    )


def test_gdn_prefill_backend_uses_validated_flashqla(monkeypatch, linear_model):
    backend_args = (1, 2, 128, 128, torch.bfloat16, torch.float32)
    validate_calls = []
    monkeypatch.setenv("FLA_FLASH_QLA", "1")
    flashqla = ModuleType("flash_qla")
    flashqla.chunk_gated_delta_rule = lambda **kwargs: ("flashqla", kwargs)
    monkeypatch.setitem(sys.modules, "flash_qla", flashqla)
    monkeypatch.setattr(
        linear_create_utils,
        "validate",
        lambda name, *args: validate_calls.append((name, args)) or True,
    )

    backend_class = linear_create_utils.get_linear_att_backend_class(linear_model)

    assert backend_class is gdn.FlashQlaLinearAttBackend
    assert backend_class._get_chunk_gated_delta_rule()(q="q")[0] == "flashqla"
    assert validate_calls == [("flashqla", backend_args)]


def test_gdn_prefill_backend_falls_back_when_flashqla_validation_fails(monkeypatch, linear_model):
    monkeypatch.setenv("FLA_FLASH_QLA", "1")
    monkeypatch.setattr(linear_create_utils, "validate", lambda *args: False)

    backend_class = linear_create_utils.get_linear_att_backend_class(linear_model)

    assert backend_class is gdn.FlaLinearAttBackend


def test_flashqla_backend_respects_disable_env(monkeypatch, linear_model):
    monkeypatch.setenv("FLA_FLASH_QLA", "0")
    monkeypatch.setattr(
        linear_create_utils,
        "validate",
        lambda *args: pytest.fail("disabled FlashQLA must not be validated"),
    )

    backend_class = linear_create_utils.get_linear_att_backend_class(linear_model)

    assert backend_class is gdn.FlaLinearAttBackend


def test_gdn_prefill_backend_tries_candidates_in_order(monkeypatch, linear_model):
    validate_calls = []
    monkeypatch.setenv("FLA_FLASH_QLA", "1")
    flashqla2_backend = type("FlashQla2LinearAttBackend", (), {})
    flashqla3_backend = type("FlashQla3LinearAttBackend", (), {})
    monkeypatch.setattr(
        linear_create_utils,
        "linear_att_backend_classes",
        {"flashqla2": flashqla2_backend, "flashqla3": flashqla3_backend},
    )
    monkeypatch.setattr(
        linear_create_utils,
        "validate",
        lambda name, *args: validate_calls.append(name) or name == "flashqla3",
    )

    backend_class = linear_create_utils.get_linear_att_backend_class(
        linear_model, priority_list=("flashqla2", "flashqla3")
    )

    assert backend_class is flashqla3_backend
    assert validate_calls == ["flashqla2", "flashqla3"]
