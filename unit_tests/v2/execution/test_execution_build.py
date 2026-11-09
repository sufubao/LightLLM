import importlib.abc
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lightllm.v2.execution import BuildConfig, ProviderRegistry, build, prepare
from lightllm.v2.execution.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.v2.execution.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.v2.execution.contracts.provider import AttentionOffer, ProbeResult
from lightllm.v2.execution.contracts.state import FormatSpec, StateSpec, StateView
from lightllm.v2.execution.formats.weights.view import WeightView
from lightllm.v2.execution.planner.weights import bind_compute
from lightllm.v2.execution.registry import AttentionProvider, StateProvider


@dataclass(frozen=True)
class CounterState:
    size: int = 1

    def validate(self):
        if self.size <= 0:
            raise ValueError("counter size must be positive")


@dataclass(frozen=True)
class CounterAttention:
    def validate_state(self, request):
        assert isinstance(request.state.params, CounterState)


def _build_with_external_provider(retention):
    events = []
    declaration = StateSpec("offline", "counter", 3, (3,), CounterState(), retention=retention)
    format_spec = FormatSpec("custom", torch.float32, "counter", ("value",), 1)
    adapter = SimpleNamespace(
        attention_specs=lambda: {3: CounterAttention()},
        state_specs=lambda: (declaration,),
        attention_states=lambda: {3: SimpleNamespace(primary=declaration, auxiliary=())},
    )

    class Store:
        def __init__(self, declarations, format, manager, device, specs):
            assert declarations == (declaration,) and specs == {3: CounterAttention()}
            self.value = None
            events.append("declare")

        def views(self):
            assert self.value is not None
            return {declaration.key: StateView(declaration, format_spec, (self.value,))}

    class Backend:
        supports_auxiliary_states = False
        implementation = "external"

        def __init__(self, phase):
            self.phase = phase
            self.name = f"external.{phase}"

        def probe(self, spec, format, device, limits, *, auxiliary_formats=None):
            assert spec == CounterAttention() and device.device_type == "cpu"
            return ProbeResult((AttentionOffer(self.name, self.phase, format, 16, execution_limits=limits),))

        def bind(self, offer, view, params, workspace, *, auxiliary_views=None):
            events.append("bind")
            assert len(workspace) == 16
            return SimpleNamespace(
                metadata_key=self.name,
                phase=self.phase,
                view=view,
                run=lambda value: value + view.fields[0],
                create_metadata=lambda batch: batch,
            )

    def allocate_workspace(size):
        events.append("workspace")
        return bytearray(size)

    providers = ProviderRegistry(
        attention=(AttentionProvider(Backend),),
        storage=(StateProvider("counter", retention, Store, lambda view: view),),
    )
    prepared = prepare(
        model_adapter=adapter,
        providers=providers,
        device=SimpleNamespace(caps=DeviceCaps("cpu")),
        config=BuildConfig(
            limits=ExecutionLimits(32),
            formats={retention: format_spec},
            requested={retention: {"prefill": "external", "decode": "external"}},
            static_params={3: SimpleNamespace(spec=CounterAttention())},
        ),
    )
    assert events == []
    resources = prepared.allocate(manager=None, device=SimpleNamespace(allocate_workspace=allocate_workspace))
    assert events == ["workspace", "workspace", "declare"]
    resources.stores[retention].value = torch.tensor([4.0])
    engine = resources.bind()
    execution = engine.plan.bindings[1].execution
    assert execution.view.fields[0] is resources.stores[retention].value
    torch.testing.assert_close(execution.run(torch.tensor([2.0])), torch.tensor([6.0]))
    assert events == ["workspace", "workspace", "declare", "bind", "bind"]
    with pytest.raises(RuntimeError, match="cannot be replaced"):
        resources.bind()


def test_build_with_external_provider_and_native_state():
    _build_with_external_provider("history")


def test_recurrent_only_build_does_not_require_history_provider():
    _build_with_external_provider("recurrent")


def test_v2_declarations_import_without_server_or_backend_dependencies():
    script = """
import importlib.abc
import sys
import torch

class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        roots = ("lightllm.models", "lightllm.server", "triton", "flashinfer", "sgl_kernel", "flash_attn_interface")
        if any(fullname == root or fullname.startswith(root + ".") for root in roots):
            raise AssertionError(fullname)

sys.meta_path.insert(0, Blocked())
from lightllm.v2.execution import BuildConfig, ProviderRegistry
from lightllm.v2.execution.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.v2.execution.compute.attention.gdn.spec import GatedDeltaNetSpec
from lightllm.v2.execution.compute.attention.kda.spec import KimiDeltaAttentionSpec
from lightllm.v2.execution.compute.attention.mla.spec import MLAAttentionSpec
from lightllm.v2.execution.compute.attention.pooled_nope.spec import PooledNoPEAttentionSpec
from lightllm.v2.execution.compute.attention.sparse_mla.spec import SparseMLAAttentionSpec
assert DenseAttentionSpec(4, 2, 64).head_dim == 64
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_v2_provider_registry_rejects_ambiguous_storage():
    from dataclasses import replace

    fmt = FormatSpec("custom", torch.float32, "token_head_dim", ("k", "v"), 1)
    custom = StateProvider(
        "token_head_dim",
        "history",
        object,
        supports=lambda value: value.encoding == "custom" and value.dtype == torch.float32 and value.page_size == 1,
    )
    providers = ProviderRegistry(storage=(custom,))
    assert providers.state_provider(fmt, "history") is custom
    with pytest.raises(ValueError, match="not registered"):
        providers.state_provider(replace(fmt, page_size=2), "history")
    with pytest.raises(ValueError, match="ambiguous"):
        ProviderRegistry(storage=(custom, custom)).state_provider(fmt, "history")


def test_v2_weight_binding_uses_registered_compute():
    from lightllm.v2.execution.compute.linear.spec import LinearSpec
    from lightllm.v2.execution.contracts.provider import ProbeResult

    tensor = torch.eye(2)
    view = WeightView(FormatSpec("plain", torch.float32, "output_input", ("weight",), 1), {"weight": tensor})
    calls = []

    class ExternalLinear:
        name = "external.linear"

        def probe(self, spec, format, device):
            calls.append((spec, format, device))
            return ProbeResult((SimpleNamespace(backend=self.name),))

        def bind(self, offer, weight):
            assert weight is tensor
            return SimpleNamespace(run=lambda x: x @ weight)

    providers = ProviderRegistry(linear=(ExternalLinear,))
    spec, device = LinearSpec(2, 2, torch.float32), DeviceCaps("cpu")
    bound = bind_compute(spec, view, device, providers.linear, requested="external.linear")
    torch.testing.assert_close(bound.run(torch.tensor([[1.0, 2.0]])), torch.tensor([[1.0, 2.0]]))
    assert calls == [(spec, view.format, device)]
