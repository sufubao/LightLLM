import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att_copy import (
    copy_linear_att_state_to_kv_buffer,
)


def _args(gpu_conv, accept_len, mtp_step):
    layer_num = gpu_conv.shape[0]
    dim_conv = gpu_conv.shape[2]
    width_narrow = 3
    return dict(
        b_req_idx=torch.tensor([0], dtype=torch.int32),
        big_page_buffer_ids=torch.tensor([0], dtype=torch.int32),
        gpu_conv_state=gpu_conv,
        gpu_ssm_state=torch.zeros(layer_num, 1 * (mtp_step + 1), 8),
        cpu_kv_conv_state=torch.zeros(1, layer_num, dim_conv, width_narrow),
        cpu_kv_ssm_state=torch.zeros(1, layer_num, 8),
        mtp_step=mtp_step,
        b_num_accepted_tokens=torch.tensor([accept_len], dtype=torch.int32),
    )


def test_rejects_non_contiguous_width_axis():
    mtp_step = 2
    # widened slot allocated 2x, then strided ::2 along the width axis -> stride(3) == 2
    base = torch.zeros(2, 1, 32, (3 + mtp_step) * 2)
    gpu_conv = base[:, :, :, ::2]
    assert gpu_conv.stride(3) != 1
    with pytest.raises(AssertionError, match="width"):
        copy_linear_att_state_to_kv_buffer(**_args(gpu_conv, accept_len=1, mtp_step=mtp_step))


def test_rejects_out_of_range_accept_len():
    mtp_step = 2
    gpu_conv = torch.zeros(2, 1, 32, 3 + mtp_step)  # contiguous, passes the #6 guard
    with pytest.raises(AssertionError, match="b_num_accepted_tokens"):
        copy_linear_att_state_to_kv_buffer(**_args(gpu_conv, accept_len=mtp_step + 2, mtp_step=mtp_step))
