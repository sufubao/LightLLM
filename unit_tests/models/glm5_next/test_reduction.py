from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_infer import base_layer_infer
from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tp_world_size", [1, 4])
def test_tp_reduction_accumulates_fp32(monkeypatch, dtype, tp_world_size):
    layer = object.__new__(Glm5NextTransformerLayerInfer)
    layer.tp_world_size_ = tp_world_size
    group = object()
    state = SimpleNamespace(dist_group=group)
    monkeypatch.setattr(base_layer_infer, "get_env_start_args", lambda: SimpleNamespace(enable_tpsp_mix_mode=False))
    large = 256 if dtype == torch.bfloat16 else 2048
    input_tensor = torch.full((2, 8), large, dtype=dtype)
    calls = []

    def reduce(input_, op, group, async_op):
        assert group is state.dist_group and async_op is False
        calls.append(input_.dtype)
        for partial in (1, -large, 0):
            input_.add_(partial)

    monkeypatch.setattr(base_layer_infer, "all_reduce", reduce)
    output = layer._tpsp_reduce(input_tensor, state)
    assert output.dtype == dtype and output.device == input_tensor.device
    if tp_world_size == 1:
        assert output is input_tensor and calls == []
    else:
        torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
        assert calls == [torch.float32]
        torch.testing.assert_close(input_tensor, torch.full_like(input_tensor, large), rtol=0, atol=0)
