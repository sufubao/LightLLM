from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager
from lightllm.models.qwen3_5_dspark.model import Qwen3_5DSparkModel
from lightllm.models.qwen3_5_dflash.model import Qwen3_5DFlashModel


def manager(device='cpu', dtype=torch.bfloat16, tp=4):
    mem = Qwen3NextMemManager.__new__(Qwen3NextMemManager)
    mem.head_num = 4 // tp
    mem.head_dim = 256
    mem.kv_buffer = torch.zeros((21, 7, 2 * mem.head_num, 256), dtype=dtype, device=device)
    mem.linear_config = SimpleNamespace(
        full_att_all_num_kv_heads=4,
        get_full_att_kv_layer_index=lambda layer: 16 + layer - 64,
    )
    mem.operator = mem.operator_class(mem)
    return mem


@pytest.mark.parametrize('tp', [1, 2, 4])
@pytest.mark.parametrize('model_class', [Qwen3_5DSparkModel, Qwen3_5DFlashModel])
def test_draft_head_view_preserves_target_storage_and_layer_mapping(tp, model_class):
    target = manager(tp=tp)
    model = model_class.__new__(model_class)
    model.main_model = SimpleNamespace(mem_manager=target)
    model.config = {'num_key_value_heads': 8, 'head_dim': 128}
    model._init_mem_manager()
    view = model.mem_manager
    k, v = view.get_att_input_params(64)
    k.fill_(3)
    v.fill_(5)
    target_k, target_v = target.get_att_input_params(64)
    assert torch.equal(target_k, torch.full_like(target_k, 3))
    assert torch.equal(target_v, torch.full_like(target_v, 5))
    assert not target.kv_buffer[:16].any()
    assert target.head_num == 4 // tp and target.head_dim == 256
    assert view.head_num == 8 // tp and view.head_dim == 128
    assert view.kv_buffer.data_ptr() == target.kv_buffer.data_ptr()
    assert view.operator.mem_manager is view
    assert target.operator.mem_manager is target


def test_static_head_scales_follow_split_heads_without_mutating_target():
    target = manager(dtype=torch.uint8)
    target.scales = torch.tensor([[2.0, 3.0]]).repeat(21, 1)
    target.q_scales = torch.tensor([[7.0]]).repeat(21, 1)
    view = target.create_kv_head_view(128)
    assert view.scales[16].tolist() == [2.0, 2.0, 3.0, 3.0]
    assert view.q_scales[16].tolist() == [7.0, 7.0]
    assert target.scales.shape == (21, 2)
    assert target.create_kv_head_view(256) is target
    with pytest.raises(ValueError):
        target.create_kv_head_view(192)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires GPU scatter kernels')
@pytest.mark.parametrize('tp', [2, 4])
def test_fp8_draft_write_roundtrip_in_target_owned_cache(tp):
    from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import FP8StaticPerHeadQuantQwen3NextMemManager

    target = manager(device='cuda', dtype=torch.uint8, tp=tp)
    target.__class__ = FP8StaticPerHeadQuantQwen3NextMemManager
    target.scales = torch.full((21, 2 * target.head_num), 0.02, device='cuda')
    target.q_scales = torch.full((21, target.head_num), 0.02, device='cuda')
    target.operator = target.operator_class(target)
    view = target.create_kv_head_view(128)
    indices = torch.tensor([1, 4], device='cuda', dtype=torch.int32)
    kv = torch.randn((2, 2 * view.head_num, 128), device='cuda', dtype=torch.bfloat16)
    view.operator.copy_kv_to_mem_manager(64, indices, kv)
    expected = (kv.float() / 0.02).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(view.kv_buffer[16, indices], expected)
    assert torch.equal(target.kv_buffer[16, indices].reshape_as(expected), expected)
    assert not target.kv_buffer[:16].any()
