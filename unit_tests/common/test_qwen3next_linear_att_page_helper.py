from types import SimpleNamespace

import torch


class _Buf:
    def __init__(self, tensor):
        self.buffer = tensor


def _make_config():
    return SimpleNamespace(
        tp_world_size=1,
        linear_layer_num=1,
        conv_kernel_size=4,
        global_linear_k_heads=1,
        global_linear_v_heads=1,
        num_linear_k_heads=1,
        num_linear_v_heads=1,
        head_linear_k_dim=2,
        head_linear_v_dim=3,
    )


def _make_mem(mtp_step=2, req_slots=4):
    config = _make_config()
    conv_dim = (
        2 * config.num_linear_k_heads * config.head_linear_k_dim
        + config.num_linear_v_heads * config.head_linear_v_dim
    )
    narrow_w = config.conv_kernel_size - 1
    conv = torch.full(
        (config.linear_layer_num, req_slots, conv_dim, narrow_w + mtp_step),
        -9.0,
        dtype=torch.float32,
    )
    ssm = torch.full(
        (
            config.linear_layer_num,
            req_slots * (mtp_step + 1),
            config.num_linear_v_heads,
            config.head_linear_k_dim,
            config.head_linear_v_dim,
        ),
        -11.0,
        dtype=torch.float32,
    )
    return SimpleNamespace(
        linear_config=config,
        req_to_conv_state=_Buf(conv),
        req_to_ssm_state=_Buf(ssm),
        kv_move_buffer=torch.zeros((1, 4096), dtype=torch.uint8),
    )


def test_page_helper_writes_req_conv_slot_and_narrow_width(monkeypatch):
    import lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager as qwen3next_mem_manager
    from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextLinearAttPageHelper

    mtp_step = 2
    req_idx = 2
    monkeypatch.setattr(qwen3next_mem_manager, "get_env_start_args", lambda: SimpleNamespace(mtp_step=mtp_step))

    mem = _make_mem(mtp_step=mtp_step)
    helper = Qwen3NextLinearAttPageHelper(mem)
    mem.kv_move_buffer = torch.zeros((1, helper.state_nbytes), dtype=torch.uint8)

    narrow_w = helper.conv_shape[-1]
    marker_conv = torch.arange(
        helper.conv_shape[0] * helper.conv_shape[1] * narrow_w,
        dtype=torch.float32,
    ).view(helper.conv_shape)
    marker_ssm = torch.arange(
        helper.ssm_shape[0] * helper.ssm_shape[1] * helper.ssm_shape[2] * helper.ssm_shape[3],
        dtype=torch.float32,
    ).view(helper.ssm_shape)

    mem.req_to_conv_state.buffer[:, req_idx, :, :narrow_w] = marker_conv
    mem.req_to_conv_state.buffer[:, req_idx, :, narrow_w:] = 999.0
    mem.req_to_ssm_state.buffer[:, req_idx * (mtp_step + 1), ...] = marker_ssm

    helper.write_req_to_page(page_index=0, req_idx=req_idx, dp_mems=[mem])

    conv_page, ssm_page = helper.view_page_to_linear_att_state(page_index=0)
    torch.testing.assert_close(conv_page, marker_conv)
    torch.testing.assert_close(ssm_page, marker_ssm)


def test_page_helper_restores_narrow_conv_to_req_slot(monkeypatch):
    import lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager as qwen3next_mem_manager
    from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextLinearAttPageHelper

    mtp_step = 2
    req_idx = 2
    monkeypatch.setattr(qwen3next_mem_manager, "get_env_start_args", lambda: SimpleNamespace(mtp_step=mtp_step))

    mem = _make_mem(mtp_step=mtp_step)
    helper = Qwen3NextLinearAttPageHelper(mem)
    mem.kv_move_buffer = torch.zeros((1, helper.state_nbytes), dtype=torch.uint8)
    conv_page, ssm_page = helper.view_page_to_linear_att_state(page_index=0)

    marker_conv = torch.arange(conv_page.numel(), dtype=torch.float32).view_as(conv_page)
    marker_ssm = torch.arange(ssm_page.numel(), dtype=torch.float32).view_as(ssm_page)
    conv_page.copy_(marker_conv)
    ssm_page.copy_(marker_ssm)

    helper.read_page_to_req(page_index=0, req_idx=req_idx, dp_mems=[mem])

    narrow_w = helper.conv_shape[-1]
    torch.testing.assert_close(mem.req_to_conv_state.buffer[:, req_idx, :, :narrow_w], marker_conv)
    assert torch.all(mem.req_to_conv_state.buffer[:, req_idx, :, narrow_w:] == -9.0)
    torch.testing.assert_close(mem.req_to_ssm_state.buffer[:, req_idx * (mtp_step + 1), ...], marker_ssm)
