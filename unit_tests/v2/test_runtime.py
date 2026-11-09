from types import SimpleNamespace

import torch

from lightllm.v2.attention_state import AttentionStateManager
from lightllm.v2.execution.contracts.device import DeviceCaps
from lightllm.v2.execution.contracts.state import FormatSpec, StateSpec
from lightllm.v2.execution.formats.kv.spec import DenseKVStateSpec
from lightllm.v2.execution.registry import ProviderRegistry
from lightllm.v2.execution.storage.attention import DenseStateStore
from lightllm.v2.runtime import V2ExecutionSession


class EmptyAdapter:
    attention_phases = ()

    def moe_specs(self):
        return {}

    def attention_specs(self):
        return {}

    def state_specs(self):
        return ()

    def attention_states(self):
        return {}


def test_v2_session_binds_empty_execution_without_touching_legacy_manager():
    runtime = SimpleNamespace(
        caps=DeviceCaps("cpu"),
        device=torch.device("cpu"),
        allocate_workspace=lambda size: torch.empty(size, dtype=torch.uint8),
    )
    device = SimpleNamespace(device=runtime, providers=ProviderRegistry())
    from lightllm.v2.execution.build import BuildConfig
    from lightllm.v2.execution.contracts.device import ExecutionLimits

    session = V2ExecutionSession.create(
        model_adapter=EmptyAdapter(),
        config=BuildConfig(ExecutionLimits(0), {}, {}, {}),
        device=device,
        request_capacity=1,
    )
    assert session.execution.plan.bindings == ()
    assert session.manager.__class__.__module__ == "lightllm.v2.attention_state.manager"
    session.close()
    session.close()


def test_v2_dense_store_allocates_native_state_inside_v2_manager(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "v2-runtime-test")
    declaration = StateSpec("main", "kv.0", 0, (0,), DenseKVStateSpec(kv_heads=2, head_dim=4))
    format_spec = FormatSpec("plain", torch.float32, "token_head_dim", ("k", "v"), 2)
    manager = AttentionStateManager(request_capacity=1)
    store = DenseStateStore((declaration,), format_spec, manager)
    manager.finish_state_registration()
    manager.allocate_storage(token_capacity=4, page_size=2, big_page_tokens=1, device="cpu")

    view = store.views()[declaration.key]
    assert view.state == declaration
    assert tuple(field.shape for field in view.fields) == ((6, 2, 4), (6, 2, 4))
    assert view.fields[0].device.type == "cpu"
    assert manager.get_full_attention_states()[0].__class__.__module__.startswith("lightllm.v2.attention_state")
