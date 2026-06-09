from types import SimpleNamespace

import pytest
import torch


def _make_start_args(**overrides):
    base = dict(
        model_dir="/tmp/qwen3_5",
        tp=1,
        dp=1,
        data_type="bfloat16",
        linear_att_ssm_data_type="bfloat16",
        mtp_mode=None,
        mtp_step=0,
        linear_att_page_block_num=2,
        linear_att_hash_page_size=4,
        cpu_cache_token_page_size=8,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_model_cfg():
    return {
        "model_type": "qwen3_5",
        "num_hidden_layers": 64,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
    }


def _patch_linear_config_args(monkeypatch, args):
    import lightllm.common.linear_att_cache_manager.config_objs as config_objs

    monkeypatch.setattr(config_objs, "get_env_start_args", lambda: args)


def _make_config(draft_full_att_layer_num=0):
    from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

    return LinearAttCacheConfig(
        tp_world_size=1,
        full_att_all_num_kv_heads=16,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=16,
        full_att_head_dim=128,
        num_linear_k_heads=16,
        num_linear_v_heads=48,
        head_linear_k_dim=128,
        head_linear_v_dim=128,
        conv_kernel_size=4,
        linear_layer_num=48,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.bfloat16,
        full_attention_interval=4,
        all_layer_num=64,
        draft_full_att_layer_num=draft_full_att_layer_num,
    )


def test_load_from_args_includes_mtp_draft_full_att_layers(monkeypatch):
    from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
    from transformers.configuration_utils import PretrainedConfig

    args = _make_start_args(mtp_mode="vanilla_with_att", mtp_step=3)
    _patch_linear_config_args(monkeypatch, args)
    monkeypatch.setattr(PretrainedConfig, "get_config_dict", lambda _model_path: (_make_model_cfg(), None))

    cfg = LinearAttCacheConfig.load_from_args()

    assert cfg.get_main_full_att_layer_num() == 16
    assert cfg.draft_full_att_layer_num == 3
    assert cfg.get_persisted_full_att_layer_num() == 19


def test_cpu_cache_full_att_bytes_include_mtp_draft_layers(monkeypatch):
    args = _make_start_args()
    _patch_linear_config_args(monkeypatch, args)
    main_only = _make_config(draft_full_att_layer_num=0)
    with_draft = _make_config(draft_full_att_layer_num=2)

    bytes_per_full_att_layer = (
        args.cpu_cache_token_page_size
        * 2
        * main_only.full_att_all_num_kv_heads
        * main_only.full_att_head_dim
        * main_only.full_att_dtype.itemsize
    )

    assert main_only.get_main_full_att_layer_num() == 16
    assert with_draft.get_persisted_full_att_layer_num() == 18
    assert with_draft.get_cpu_cache_full_att_bytes() == (
        main_only.get_cpu_cache_full_att_bytes() + 2 * bytes_per_full_att_layer
    )


def test_linear_operator_persisted_full_att_slice_includes_draft_slots():
    from lightllm.common.kv_cache_mem_manager.operator.linear_att import LinearAttMemOperator

    class MtpMemManager:
        main_full_att_layer_num = 16
        draft_full_att_layers = 2
        kv_buffer = torch.empty((18, 1))

    class MainOnlyMemManager:
        main_full_att_layer_num = 16
        kv_buffer = torch.empty((18, 1))

    class PlainMemManager:
        kv_buffer = torch.empty((7, 1))

    assert LinearAttMemOperator._get_persisted_full_att_layer_num(MtpMemManager()) == 18
    assert LinearAttMemOperator._get_persisted_full_att_layer_num(MainOnlyMemManager()) == 16
    assert LinearAttMemOperator._get_persisted_full_att_layer_num(PlainMemManager()) == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_linear_cpu_cache_roundtrips_mtp_draft_full_att_slot(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
        copy_cpu_cache_to_kv_buffer,
        copy_kv_buffer_to_cpu_cache,
    )
    from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

    args = _make_start_args(
        linear_att_page_block_num=1,
        linear_att_hash_page_size=2,
        cpu_cache_token_page_size=2,
    )
    _patch_linear_config_args(monkeypatch, args)
    cfg = LinearAttCacheConfig(
        tp_world_size=1,
        full_att_all_num_kv_heads=2,
        full_att_dtype=torch.float32,
        full_att_num_kv_heads=2,
        full_att_head_dim=8,
        num_linear_k_heads=1,
        num_linear_v_heads=1,
        head_linear_k_dim=8,
        head_linear_v_dim=8,
        conv_kernel_size=2,
        linear_layer_num=1,
        conv_state_dtype=torch.float32,
        ssm_state_dtype=torch.float32,
        full_attention_interval=2,
        all_layer_num=2,
        draft_full_att_layer_num=1,
    )

    gpu_kv = torch.arange(2 * 2 * 4 * 8, dtype=torch.float32, device="cuda").reshape(2, 2, 4, 8)
    cpu_cache_tensor = torch.zeros(
        (1, 1, 1, 1, cfg.get_cpu_cache_big_page_bytes()),
        dtype=torch.uint8,
        device="cuda",
    )
    conv_state = torch.zeros(
        (1, cfg.linear_layer_num, cfg.get_conv_dim(), cfg.conv_kernel_size - 1),
        dtype=torch.float32,
        device="cuda",
    )
    ssm_state = torch.zeros(
        (
            1,
            cfg.linear_layer_num,
            cfg.num_linear_v_heads,
            cfg.head_linear_k_dim,
            cfg.head_linear_v_dim,
        ),
        dtype=torch.float32,
        device="cuda",
    )
    mem_indexes = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    page_indexes = torch.tensor([0], dtype=torch.int32, device="cuda")
    page_readies = torch.tensor([False], dtype=torch.bool, device="cuda")
    big_page_buffer_ids = torch.tensor([0], dtype=torch.int64, device="cuda")

    copy_kv_buffer_to_cpu_cache(
        mem_indexes=mem_indexes,
        page_indexes=page_indexes,
        page_readies=page_readies,
        big_page_buffer_ids=big_page_buffer_ids,
        gpu_kv_full_att_state=gpu_kv,
        cpu_kv_conv_state=conv_state,
        cpu_kv_ssm_state=ssm_state,
        cpu_cache_tensor=cpu_cache_tensor,
        tp_rank=0,
        tp_world_size=1,
        big_page_token_num=args.cpu_cache_token_page_size,
        linear_config=cfg,
        grid_num=1,
    )

    restored_gpu_kv = torch.full_like(gpu_kv, fill_value=-1)
    restored_conv = torch.empty_like(conv_state)
    restored_ssm = torch.empty_like(ssm_state)
    copy_cpu_cache_to_kv_buffer(
        mem_indexes=mem_indexes,
        big_page_buffer_ids=big_page_buffer_ids,
        page_indexes=page_indexes,
        gpu_full_att_kv_state=restored_gpu_kv,
        cpu_kv_conv_state=restored_conv,
        cpu_kv_ssm_state=restored_ssm,
        cpu_cache_tensor=cpu_cache_tensor,
        tp_rank=0,
        tp_world_size=1,
        big_page_token_num=args.cpu_cache_token_page_size,
        linear_config=cfg,
        grid_num=1,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(restored_gpu_kv, gpu_kv)
