import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("S", [1, 2, 3])
@pytest.mark.parametrize("accept_len", [1, 2])
def test_snapshot_reads_committed_conv_and_ssm(S, accept_len):
    from lightllm.common.basemodel.triton_kernel.linear_att_copy import (
        copy_linear_att_state_to_kv_buffer,
    )

    layer_num, dim_conv = 2, 32
    width_narrow = 3
    gpu_conv = torch.zeros(layer_num, 1, dim_conv, width_narrow + S, device="cuda")
    off = accept_len - 1
    marker_conv = torch.arange(dim_conv * width_narrow, device="cuda").float().reshape(dim_conv, width_narrow)
    gpu_conv[:, 0, :, off : off + width_narrow] = marker_conv

    hv, k, v = 4, 8, 8
    gpu_ssm = torch.zeros(layer_num, 1 * (S + 1), hv, k, v, device="cuda")
    marker_ssm = torch.arange(hv * k * v, device="cuda").float().reshape(hv, k, v)
    gpu_ssm[:, off, ...] = marker_ssm  # block slot 0*(S+1)+off

    cpu_conv = torch.zeros(1, layer_num, dim_conv, width_narrow, device="cuda")
    cpu_ssm = torch.zeros(1, layer_num, hv, k, v, device="cuda")

    copy_linear_att_state_to_kv_buffer(
        b_req_idx=torch.tensor([0], dtype=torch.int32, device="cuda"),
        big_page_buffer_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
        gpu_conv_state=gpu_conv,
        gpu_ssm_state=gpu_ssm,
        cpu_kv_conv_state=cpu_conv,
        cpu_kv_ssm_state=cpu_ssm,
        mtp_step=S,
        b_num_accepted_tokens=torch.tensor([accept_len], dtype=torch.int32, device="cuda"),
    )

    torch.testing.assert_close(cpu_conv[0], marker_conv.expand(layer_num, dim_conv, width_narrow))
    torch.testing.assert_close(cpu_ssm[0], marker_ssm.expand(layer_num, hv, k, v))
