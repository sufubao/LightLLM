import importlib.util
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


@pytest.fixture(scope="module")
def manager_modules():
    """Import GPU modules without leaking local macOS compatibility stubs."""
    lightllm_modules_before = {name for name in sys.modules if name.startswith("lightllm")}
    module_patch = nullcontext()
    if importlib.util.find_spec("triton") is None:
        triton = types.ModuleType("triton")
        triton.jit = lambda fn: fn
        triton.cdiv = lambda value, divisor: (value + divisor - 1) // divisor
        triton_language = types.ModuleType("triton.language")
        triton_language.constexpr = object()
        triton.language = triton_language
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForCausalLM = object()
        module_patch = patch.dict(
            sys.modules,
            {
                "transformers": transformers,
                "triton": triton,
                "triton.language": triton_language,
            },
        )

    try:
        with module_patch:
            from lightllm.common.kv_cache_mem_manager import Deepseek2MemoryManager, MemoryManager
            from lightllm.common.kv_cache_mem_manager import mem_manager as mem_manager_module

            yield SimpleNamespace(
                deepseek_class=Deepseek2MemoryManager,
                memory_manager_class=MemoryManager,
                module=mem_manager_module,
            )
    finally:
        for module_name in list(sys.modules):
            if module_name.startswith("lightllm") and module_name not in lightllm_modules_before:
                sys.modules.pop(module_name, None)


def _profile_manager(monkeypatch, manager_modules, manager, *, run_mode, mem_fraction=0.8, page_size=4096):
    mem_manager_module = manager_modules.module
    monkeypatch.setattr(mem_manager_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(mem_manager_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(mem_manager_module, "get_available_gpu_memory", lambda world_size: 10.0)
    monkeypatch.setattr(mem_manager_module, "get_total_gpu_memory", lambda: 10.0)
    monkeypatch.setattr(
        mem_manager_module,
        "get_env_start_args",
        lambda: SimpleNamespace(
            run_mode=run_mode,
            model_dir="unused",
            pd_kv_page_num=16,
            pd_kv_page_size=page_size,
        ),
    )

    manager.profile_size(mem_fraction)


@pytest.mark.parametrize("run_mode", ["prefill", "decode"])
@pytest.mark.parametrize("page_size, expected_token_capacity", [(1024, 427911), (4096, 231303)])
def test_pd_profile_reserves_qwen35_transfer_page_buffer(
    monkeypatch, manager_modules, run_mode, page_size, expected_token_capacity
):
    manager = object.__new__(manager_modules.memory_manager_class)
    manager.size = None
    manager.dtype = torch.bfloat16
    manager.head_num = 1
    manager.head_dim = 256
    manager.layer_num = 17
    monkeypatch.setattr(manager_modules.module, "get_num_key_value_heads", lambda model_dir: 4)

    _profile_manager(monkeypatch, manager_modules, manager, run_mode=run_mode, page_size=page_size)

    # At page_size=4096, mem_fraction leaves 8 GiB and the Qwen3.5 PD page buffer
    # occupies 4.25 GiB: 16 * 4096 * 17 layers * 2 K/V * 4 global heads * 256 * 2 bytes.
    assert manager.size == expected_token_capacity


@pytest.mark.parametrize("run_mode", ["normal", "pd_master"])
def test_non_pd_worker_profile_does_not_reserve_pd_transfer_buffer(monkeypatch, manager_modules, run_mode):
    manager = object.__new__(manager_modules.memory_manager_class)
    manager.size = None
    manager.dtype = torch.bfloat16
    manager.head_num = 1
    manager.head_dim = 256
    manager.layer_num = 17
    monkeypatch.setattr(manager_modules.module, "get_num_key_value_heads", lambda model_dir: 4)

    _profile_manager(monkeypatch, manager_modules, manager, run_mode=run_mode)

    assert manager.size == 493447


def test_pd_profile_uses_deepseek_transfer_buffer_layout(monkeypatch, manager_modules):
    manager = object.__new__(manager_modules.deepseek_class)
    manager.size = None
    manager.dtype = torch.bfloat16
    manager.head_num = 1
    manager.head_dim = 576
    manager.layer_num = 61

    _profile_manager(monkeypatch, manager_modules, manager, run_mode="decode")

    # MLA pages store one compressed KV tensor per local head rather than K/V
    # tensors for every global KV head.
    assert manager.size == 56702


def test_explicit_token_capacity_skips_pd_memory_profile(monkeypatch, manager_modules):
    manager = object.__new__(manager_modules.memory_manager_class)
    manager.size = 12345

    monkeypatch.setattr(
        manager_modules.module,
        "get_env_start_args",
        lambda: pytest.fail("explicit capacity must not inspect PD profile arguments"),
    )

    manager.profile_size(mem_fraction=0.8)

    assert manager.size == 12345


@pytest.mark.parametrize("mem_fraction", [0.4, 0.425001])
def test_pd_profile_fails_when_transfer_buffer_leaves_no_token_capacity(monkeypatch, manager_modules, mem_fraction):
    manager = object.__new__(manager_modules.memory_manager_class)
    manager.size = None
    manager.dtype = torch.bfloat16
    manager.head_num = 1
    manager.head_dim = 256
    manager.layer_num = 17
    monkeypatch.setattr(manager_modules.module, "get_num_key_value_heads", lambda model_dir: 4)

    with pytest.raises(
        RuntimeError,
        match=r"reduce --pd_kv_page_size or --pd_kv_page_num, or increase --mem_fraction",
    ):
        _profile_manager(monkeypatch, manager_modules, manager, run_mode="decode", mem_fraction=mem_fraction)


@pytest.mark.parametrize(
    "manager_name, head_num, head_dim, layer_num, global_kv_heads, expected_shape",
    [
        ("base", 1, 256, 17, 4, (2, 32, 17, 8, 256)),
        ("deepseek", 1, 576, 61, 4, (2, 32, 61, 1, 576)),
    ],
)
def test_pd_transfer_allocation_uses_profiled_buffer_shape(
    monkeypatch,
    manager_modules,
    manager_name,
    head_num,
    head_dim,
    layer_num,
    global_kv_heads,
    expected_shape,
):
    manager_class = manager_modules.memory_manager_class if manager_name == "base" else manager_modules.deepseek_class
    manager = object.__new__(manager_class)
    manager.dtype = torch.bfloat16
    manager.head_num = head_num
    manager.head_dim = head_dim
    manager.layer_num = layer_num
    monkeypatch.setattr(
        manager_modules.module,
        "get_env_start_args",
        lambda: SimpleNamespace(model_dir="unused"),
    )
    monkeypatch.setattr(
        manager_modules.module,
        "get_num_key_value_heads",
        lambda model_dir: global_kv_heads,
    )
    real_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        kwargs.pop("pin_memory", None)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)

    move_buffer = manager.alloc_paged_kv_move_buffer(page_num=2, page_size=32)

    assert tuple(move_buffer.shape) == expected_shape
