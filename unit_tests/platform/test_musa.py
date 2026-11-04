import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.core.compute.attention.dense.musa import MusaAttentionBackend
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.core.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.core.contracts.state import FormatSpec
from lightllm.platform.musa import validate_config
from lightllm.server.core.objs.start_args_type import StartArgs


def config(**kwargs):
    return StartArgs(
        hardware_platform="musa",
        disable_symm_mem_allreduce=True,
        disable_flashinfer_allreduce=True,
        **kwargs,
    )


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_musa_device_and_format_selection(phase):
    backend = MusaAttentionBackend(phase)
    spec = DenseAttentionSpec(4, 2, 64, dtype=torch.float16)
    fmt = FormatSpec("plain", torch.float16, "token_head_dim", ("k", "v"), 1)
    limits = ExecutionLimits(1024 * 1024, 2, 64, graph_phases=("decode",))
    caps = DeviceCaps("musa", 0, platform="musa")
    assert backend.probe(spec, fmt, caps, limits).offers
    for invalid in (DeviceCaps("cuda", 0, (9, 0)), replace(caps, platform="nvidia")):
        assert not backend.probe(spec, fmt, invalid, limits).offers
    assert not backend.probe(spec, replace(fmt, encoding="int8"), caps, limits).offers
    for change in (
        dict(dynamic_verify=True),
        dict(max_verify_tokens=3),
        dict(graph_phases=("prefill",)),
        dict(transfer_paths=("cpu_cache",)),
        dict(microbatch_count=2),
        dict(expert_parallel=True),
    ):
        result = backend.probe(spec, fmt, caps, replace(limits, **change))
        assert not result.offers and result.reason


@pytest.mark.parametrize(
    "option",
    [
        "enable_cpu_cache",
        "enable_ep_moe",
        "enable_torch_memory_saver",
        "enable_prefill_cudagraph",
        "enable_decode_microbatch_overlap",
        "enable_multimodal",
    ],
)
def test_unsupported_service_modes_fail_before_loading_musa(option):
    with pytest.raises(ValueError, match=option):
        validate_config(config(**{option: True}))


def test_factory_uses_registered_providers_without_cuda(monkeypatch):
    from lightllm.core.device import musa
    from lightllm.platform.musa import create_context
    from lightllm.utils import device_utils

    runtime = SimpleNamespace(caps=DeviceCaps("musa", 0, platform="musa"))
    constructor = Mock(return_value=runtime)
    monkeypatch.setattr(musa, "MusaRuntime", constructor)
    context = create_context(index=3, config=config())
    constructor.assert_called_once_with(3)
    assert context.device is runtime
    assert {b.name for b in context.providers.attention_backends(("prefill", "decode"))} == {
        "triton.prefill",
        "triton.decode",
    }
    monkeypatch.setattr(device_utils, "is_nvidia", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: pytest.fail("NVIDIA query on MUSA"))
    device_utils.is_tesla.cache_clear()
    assert not device_utils.is_tesla()
    device_utils.is_tesla.cache_clear()


def test_runtime_routes_graph_streams_and_events_to_musa(monkeypatch):
    from lightllm.core.device import musa

    api = Mock()
    api.get_device_name.return_value = "MTGPU"
    api.mem_get_info.return_value = (800, 1000)
    device = SimpleNamespace(type="musa", index=2)
    monkeypatch.setitem(sys.modules, "torch_musa", SimpleNamespace())
    monkeypatch.setattr(musa, "torch", SimpleNamespace(musa=api, device=lambda *args: device))
    runtime = musa.MusaRuntime(2)
    assert runtime.caps.platform == "musa" and not runtime.caps.is_nvidia
    assert runtime.distributed_backend == "mccl"
    assert runtime.workspace_budget(0.9) == 700
    stream, event, graph = runtime.current_stream(), runtime.record_event(), runtime.create_graph()
    event.record.assert_called_once_with(stream)
    runtime.use_stream(stream)
    api.stream.assert_called_once_with(stream)
    pool = runtime.graph_pool()
    runtime.capture_graph(graph, pool=pool)
    api.graph.assert_called_once_with(graph, pool=pool)
