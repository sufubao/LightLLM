from types import SimpleNamespace

import pytest
import torch

from lightllm.v2.execution.formats.providers import state_providers
from lightllm.v2.models import get_model_adapter
from lightllm.v2.models.qwen38_flash_next import (
    Qwen38FlashNextModelAdapter,
    build_config,
    default_formats,
    normalize_model_config,
    requested_phases,
)


@pytest.fixture
def config():
    return {
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "indexed_attention"],
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "indexer_n_heads": 2,
        "indexer_head_dim": 16,
        "indexer_budget": 8,
        "partial_rotary_factor": 0.5,
        "ple_layer_ids": [2],
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "hc_count": 2,
        "eos_token_id": 63,
        "n_routed_experts": 2,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 32,
        "norm_topk_prob": True,
    }


def test_normalize_model_config_accepts_indexed_attention(config):
    normalized = normalize_model_config({"text_config": config})
    assert normalized["layer_types"] == ["linear_attention"] * 3 + ["full_attention"]
    assert normalized["head_dim"] == 8


def test_v2_model_registry_accepts_qwen38_aliases(config):
    assert get_model_adapter(dict(config, model_type="qwen4_exp")).config["layer_types"][-1] == "full_attention"
    assert get_model_adapter(dict(config, model_type="qwen3_8_flash_next_text")).layer_ids == (0, 1, 2, 3)


def test_adapter_declares_periodic_qsa_gdn_and_lexical_resources(config):
    adapter = Qwen38FlashNextModelAdapter(config, dtype=torch.bfloat16, tp_world_size=1, request_capacity=4)
    specs = adapter.attention_specs()
    states = {state.resource: state for state in adapter.state_specs()}
    assert specs[0].__class__.__name__ == "GatedDeltaNetSpec"
    assert specs[3].__class__.__name__ == "SparseMLAAttentionSpec"
    assert states["gdn.0"].retention == "recurrent"
    assert states["qsa.3"].retention == "history"
    assert states["qsa_positions.3"].retention == "positions"
    assert states["lexical"].retention == "bounded"
    assert adapter.attention_states()[3].auxiliary[0][0] == "positions"
    assert set(adapter.moe_specs()) == {0, 1, 2, 3}


def test_adapter_rejects_non_periodic_layout(config):
    invalid = dict(config, layer_types=["full_attention"] * 4)
    with pytest.raises(ValueError, match="periodic attention layout"):
        Qwen38FlashNextModelAdapter(invalid)


def test_default_formats_cover_every_declared_state(config):
    adapter = Qwen38FlashNextModelAdapter(config)
    formats = default_formats(config)
    assert set(formats) == {"history", "positions", "recurrent", "bounded"}
    assert set(formats) >= {state.retention for state in adapter.state_specs()}
    assert requested_phases(config)["history"] == {"prefill": "triton", "decode": "triton"}


def test_lexical_state_is_real_bounded_runtime(config):
    adapter = Qwen38FlashNextModelAdapter(config, request_capacity=3)
    state = adapter.lexical_state(device="cpu")
    assert state.get_layer_view(1)[0].shape == (3, 2)
    assert state.get_layer_view(1)[1].shape == (3, 9, 64)
    state.get_layer_view(1)[0][0].fill_(7)
    state.get_layer_view(1)[1][0].fill_(3)
    state.initialize_page_pools(1, device="cpu", pin_memory=False)
    state.save_bounded_att_runtime_to_pages([0], [8], "big", [0])
    state.reset_bounded_att_state([0])
    state.restore_bounded_att_runtime_from_pages([0], [8], "big", [0])
    assert int(state.get_layer_view(1)[0][0, 0]) == 7
    assert int(state.get_layer_view(1)[1][0, 0, 0]) == 3


def test_state_providers_include_qwen_lexical_storage():
    providers = state_providers()
    assert any(provider.layout == "request_lexical" and provider.retention == "bounded" for provider in providers)


def test_sparse_params_use_device_allocator(config):
    adapter = Qwen38FlashNextModelAdapter(config)
    allocated = []
    device = SimpleNamespace(allocate_tensor=lambda shape, dtype: allocated.append((shape, dtype)))
    linear = SimpleNamespace(
        conv1d=SimpleNamespace(weight=torch.empty(1), bias=None),
        A_log=torch.empty(1),
        dt_bias=torch.empty(1),
    )
    weights = {layer: SimpleNamespace(linear_attn=linear) for layer in adapter.attention_specs()}
    params = adapter.attention_params(weights, device)
    assert params[3].spec == adapter.attention_specs()[3]
    assert allocated == []


def test_build_config_wires_adapter_formats_and_static_params(config):
    adapter = Qwen38FlashNextModelAdapter(config)
    linear = SimpleNamespace(
        conv1d=SimpleNamespace(weight=torch.empty(1), bias=None),
        A_log=torch.empty(1),
        dt_bias=torch.empty(1),
    )
    weights = {layer: SimpleNamespace(linear_attn=linear) for layer in adapter.attention_specs()}
    device = SimpleNamespace(allocate_tensor=lambda shape, dtype: None)
    limits = SimpleNamespace()
    built = build_config(adapter, device, limits, weights)
    assert built.formats["history"].layout == "token_packed_mla"
    assert built.formats["positions"].layout == "token_head_dim"
    assert set(built.static_params) == set(adapter.attention_specs())
