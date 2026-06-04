import torch
import pytest


def _make_cfg(conv_kernel_size=4, mtp_step=0):
    from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

    return LinearAttCacheConfig(
        tp_world_size=1,
        full_att_all_num_kv_heads=16,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=16,
        full_att_head_dim=256,
        num_linear_k_heads=16,
        num_linear_v_heads=48,
        head_linear_k_dim=128,
        head_linear_v_dim=128,
        conv_kernel_size=conv_kernel_size,
        linear_layer_num=48,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.bfloat16,
        full_attention_interval=4,
        all_layer_num=64,
    )


@pytest.mark.parametrize("S", [0, 1, 2, 3])
def test_gpu_shape_widens_by_S_persisted_stays_narrow(S):
    cfg = _make_cfg(conv_kernel_size=4, mtp_step=S)
    conv_dim = cfg.get_conv_dim()
    assert cfg.get_persisted_conv_state_shape() == (conv_dim, 4 - 1)
    assert cfg.get_gpu_conv_state_shape(mtp_step=S) == (conv_dim, (4 - 1) + S)
    assert cfg.get_conv_state_bytes_per_layer() == conv_dim * (4 - 1) * cfg.conv_state_dtype.itemsize
