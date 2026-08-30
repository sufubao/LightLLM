from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.norm_weight import (
    GatedRMSNormWeight,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sigmoid_gated_rmsnorm_is_packed_batch_invariant(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr(
        "lightllm.common.basemodel.layer_weights.meta_weights.platform_op.get_env_start_args",
        lambda: SimpleNamespace(
            hardware_platform="cuda",
            enable_torch_fallback=False,
            enable_triton_fallback=False,
        ),
    )
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    row_count = 192
    packed_row_count = 768
    hidden_size = 128

    norm = GatedRMSNormWeight(
        dim=hidden_size,
        weight_name="weight",
        data_type=dtype,
        gate_activation="sigmoid",
    )
    norm.weight.copy_(torch.randn(hidden_size, device=device, dtype=dtype))
    x = torch.randn(row_count, hidden_size, device=device, dtype=dtype)
    gate = torch.randn_like(x)
    packed_x = torch.cat(
        (
            x,
            torch.randn(
                packed_row_count - row_count,
                hidden_size,
                device=device,
                dtype=dtype,
            ),
        )
    )
    packed_gate = torch.cat(
        (
            gate,
            torch.randn(
                packed_row_count - row_count,
                hidden_size,
                device=device,
                dtype=dtype,
            ),
        )
    )

    standalone = norm._triton_forward(x, gate, eps=1e-6)
    packed = norm._triton_forward(packed_x, packed_gate, eps=1e-6)

    torch.testing.assert_close(standalone, packed[:row_count], rtol=0, atol=0)
