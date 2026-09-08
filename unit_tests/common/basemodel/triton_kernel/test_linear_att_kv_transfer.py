"""Correctness of KV-only transport with a partial physical tail page."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.kv_cache_offload import offload_gpu_kv_to_cpu, load_cpu_kv_to_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("length", [1, 8, 11, 16])
def test_partial_page_transfer_does_not_touch_padding_or_hold_slot(length):
    page_size = 8
    page_count = (length + page_size - 1) // page_size
    source = torch.arange(2 * 40 * 4 * 8, device="cuda", dtype=torch.float32).reshape(2, 40, 4, 8)
    cpu = torch.full((page_count, 2, page_size, 4, 8), -123.0, dtype=torch.float32, pin_memory=True)
    indexes = torch.full((page_count * page_size,), -1, device="cuda", dtype=torch.int32)
    indexes[:length] = torch.arange(3, 3 + length, device="cuda", dtype=torch.int32)
    pages = torch.arange(page_count, dtype=torch.int32, device="cuda")
    ready = torch.zeros(page_count, dtype=torch.bool, device="cuda")
    offload_gpu_kv_to_cpu(indexes, source, None, cpu, None, pages, ready, 0, 1, 1)
    torch.cuda.synchronize()
    actual = cpu.transpose(0, 1).reshape(2, -1, 4, 8)
    assert torch.equal(actual[:, :length], source[:, 3 : 3 + length].cpu())
    assert torch.count_nonzero(actual[:, length:]) == 0
    destination = torch.full_like(source, -77.0)
    load_cpu_kv_to_gpu(indexes, destination, None, cpu, None, pages, 0, 1, 1)
    torch.cuda.synchronize()
    assert torch.equal(destination[:, 3 : 3 + length], source[:, 3 : 3 + length])
    assert torch.all(destination[:, :3] == -77.0)
    assert torch.all(destination[:, 3 + length :] == -77.0)
