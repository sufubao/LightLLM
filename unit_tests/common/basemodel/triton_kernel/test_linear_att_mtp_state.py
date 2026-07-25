import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att_mtp_state import (
    copy_canonical_to_mtp_workspace,
    copy_mtp_workspace_to_canonical,
)


if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def test_stage_and_commit_selected_linear_attention_state():
    layers = 2
    request_slots = 6
    active_capacity = 3
    mtp_size = 4
    mtp_step = mtp_size - 1
    conv_dim = 5
    conv_width = 3

    canonical_ssm = torch.arange(layers * request_slots * 2 * 3, dtype=torch.float32, device="cuda").reshape(
        layers, request_slots, 2, 3
    )
    ssm_state = torch.full(
        (layers, request_slots + (active_capacity + 1) * mtp_step, 2, 3),
        -1.0,
        dtype=torch.float32,
        device="cuda",
    )
    ssm_state[:, :request_slots].copy_(canonical_ssm)
    canonical_conv = torch.arange(
        layers * request_slots * conv_dim * conv_width,
        dtype=torch.float32,
        device="cuda",
    ).reshape(layers, request_slots, conv_dim, conv_width)
    original_ssm = canonical_ssm.clone()
    original_conv = canonical_conv.clone()

    workspace_conv = torch.full(
        (layers, active_capacity + 1, conv_dim, conv_width + mtp_size - 1),
        -1.0,
        dtype=torch.float32,
        device="cuda",
    )
    req_idx = torch.tensor([4, 1, 3], dtype=torch.int32, device="cuda")
    workspace_idx = torch.tensor([1, 0, 2], dtype=torch.int32, device="cuda")

    copy_canonical_to_mtp_workspace(
        canonical_conv=canonical_conv,
        workspace_conv=workspace_conv,
        req_idx=req_idx,
        workspace_idx=workspace_idx,
    )

    assert torch.equal(ssm_state[:, :request_slots], original_ssm)
    assert torch.all(ssm_state[:, request_slots:] == -1)
    assert torch.equal(workspace_conv[:, 1, :, :conv_width], original_conv[:, 4])
    assert torch.equal(workspace_conv[:, 0, :, :conv_width], original_conv[:, 1])
    assert torch.equal(workspace_conv[:, 2, :, :conv_width], original_conv[:, 3])

    for request, workspace_req in ((1, 0), (4, 1), (3, 2)):
        ssm_state[:, request].fill_(1000 + 100 * workspace_req)
        for mtp_index in range(mtp_size):
            if mtp_index > 0:
                compact_index = request_slots + workspace_req * mtp_step + mtp_index - 1
                ssm_state[:, compact_index].fill_(1000 + 100 * workspace_req + mtp_index)
        for column in range(workspace_conv.shape[-1]):
            workspace_conv[:, workspace_req, :, column].fill_(2000 + 100 * workspace_req + column)

    accepted_idx = torch.full((request_slots,), 3, dtype=torch.int32, device="cuda")
    accepted_idx[4] = 2
    accepted_idx[1] = 0
    accepted_idx[3] = 3
    copy_mtp_workspace_to_canonical(
        ssm_state=ssm_state,
        canonical_conv=canonical_conv,
        workspace_conv=workspace_conv,
        req_idx=req_idx,
        workspace_idx=workspace_idx,
        accepted_idx=accepted_idx,
        canonical_ssm_size=request_slots,
        mtp_step=mtp_step,
    )

    assert torch.all(ssm_state[:, 4] == 1102)
    assert torch.all(ssm_state[:, 1] == 1000)
    assert torch.all(ssm_state[:, 3] == 1203)
    assert torch.equal(canonical_conv[:, 4], workspace_conv[:, 1, :, 2 : 2 + conv_width])
    assert torch.equal(canonical_conv[:, 1], workspace_conv[:, 0, :, :conv_width])
    assert torch.equal(canonical_conv[:, 3], workspace_conv[:, 2, :, 3 : 3 + conv_width])
    assert torch.equal(ssm_state[:, 0], original_ssm[:, 0])
    assert torch.equal(canonical_conv[:, 0], original_conv[:, 0])
    assert accepted_idx[4].item() == 0
    assert accepted_idx[1].item() == 0
    assert accepted_idx[3].item() == 0
    assert accepted_idx[0].item() == 3


def test_stage_and_commit_contiguous_linear_attention_state():
    layers = 2
    request_slots = 6
    active_capacity = 2
    mtp_step = 3
    mtp_size = mtp_step + 1
    conv_dim = 3
    conv_width = 2

    ssm_state = torch.full(
        (layers, request_slots + (active_capacity + 1) * mtp_size, 2, 3),
        -1.0,
        dtype=torch.float32,
        device="cuda",
    )
    canonical_conv = torch.arange(
        layers * request_slots * conv_dim * conv_width,
        dtype=torch.float32,
        device="cuda",
    ).reshape(layers, request_slots, conv_dim, conv_width)
    workspace_conv = torch.full(
        (layers, active_capacity + 1, conv_dim, conv_width + mtp_step),
        -1.0,
        dtype=torch.float32,
        device="cuda",
    )
    req_idx = torch.tensor([4, 1], dtype=torch.int32, device="cuda")
    workspace_idx = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    ssm_state[:, 4].fill_(40)
    ssm_state[:, 1].fill_(10)

    copy_canonical_to_mtp_workspace(
        canonical_conv=canonical_conv,
        workspace_conv=workspace_conv,
        req_idx=req_idx,
        workspace_idx=workspace_idx,
        ssm_state=ssm_state,
        canonical_ssm_size=request_slots,
        mtp_step=mtp_step,
        use_contiguous_ssm_workspace=True,
    )

    assert torch.all(ssm_state[:, request_slots + mtp_size] == 40)
    assert torch.all(ssm_state[:, request_slots] == 10)

    for workspace_request in range(active_capacity):
        for mtp_index in range(mtp_size):
            ssm_state[
                :,
                request_slots + workspace_request * mtp_size + mtp_index,
            ].fill_(100 * workspace_request + mtp_index)
        for column in range(workspace_conv.shape[-1]):
            workspace_conv[:, workspace_request, :, column].fill_(200 + 100 * workspace_request + column)

    accepted_idx = torch.zeros((request_slots,), dtype=torch.int32, device="cuda")
    accepted_idx[4] = 2
    accepted_idx[1] = 0
    copy_mtp_workspace_to_canonical(
        ssm_state=ssm_state,
        canonical_conv=canonical_conv,
        workspace_conv=workspace_conv,
        req_idx=req_idx,
        workspace_idx=workspace_idx,
        accepted_idx=accepted_idx,
        canonical_ssm_size=request_slots,
        mtp_step=mtp_step,
        use_contiguous_ssm_workspace=True,
    )

    assert torch.all(ssm_state[:, 4] == 102)
    assert torch.all(ssm_state[:, 1] == 0)
    assert torch.equal(canonical_conv[:, 4], workspace_conv[:, 1, :, 2 : 2 + conv_width])
    assert torch.equal(canonical_conv[:, 1], workspace_conv[:, 0, :, :conv_width])
    assert accepted_idx[4].item() == 0
    assert accepted_idx[1].item() == 0


def test_commit_tp2_qwen35_compact_state_shape():
    layers = 48
    request_slots = 129
    workspace_slots = 1
    mtp_size = 4
    mtp_step = mtp_size - 1
    ssm_shape = (24, 128, 128)

    ssm_state = torch.empty(
        (layers, request_slots + workspace_slots * mtp_step, *ssm_shape),
        dtype=torch.uint8,
        device="cuda",
    )
    canonical_conv = torch.zeros((layers, request_slots, 1, 1), dtype=torch.uint8, device="cuda")
    workspace_conv = torch.zeros((layers, workspace_slots, 1, mtp_size), dtype=torch.uint8, device="cuda")
    ssm_state[:, 128].fill_(3)
    ssm_state[:, request_slots].fill_(7)
    canonical_conv[:, 128].fill_(5)
    workspace_conv[:, 0, :, 1].fill_(9)
    req_idx = torch.tensor([128], dtype=torch.int32, device="cuda")
    workspace_idx = torch.tensor([0], dtype=torch.int32, device="cuda")
    accepted_idx = torch.zeros((request_slots,), dtype=torch.int32, device="cuda")
    accepted_idx[128] = 1

    copy_mtp_workspace_to_canonical(
        ssm_state=ssm_state,
        canonical_conv=canonical_conv,
        workspace_conv=workspace_conv,
        req_idx=req_idx,
        workspace_idx=workspace_idx,
        accepted_idx=accepted_idx,
        canonical_ssm_size=request_slots,
        mtp_step=mtp_step,
    )
    torch.cuda.synchronize()

    assert torch.all(ssm_state[:, 128] == 7)
    assert torch.all(canonical_conv[:, 128] == 9)
    assert accepted_idx[128].item() == 0
