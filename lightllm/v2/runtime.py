"""High-level lifecycle for the isolated v2 execution runtime.

The legacy model/server path is intentionally untouched.  Callers opt into v2
by constructing a session with a v2 ModelAdapter, DeviceRuntime and provider
registry.  The session makes the ordering explicit:

    prepare -> allocate stores -> finalize native State storage -> bind runner
"""

from __future__ import annotations

from dataclasses import dataclass

from lightllm.v2.attention_state import AttentionStateManager
from lightllm.v2.execution.build import AttentionExecution, AttentionResources, BuildConfig, prepare
from lightllm.v2.execution.contracts.model import ModelAdapter
from lightllm.v2.execution.device.context import ExecutionContext
from lightllm.v2.execution.registry import ProviderRegistry


@dataclass
class V2ExecutionSession:
    """Own a fully bound v2 execution and its native State manager."""

    execution: AttentionExecution
    resources: AttentionResources
    manager: AttentionStateManager
    _closed: bool = False

    @classmethod
    def create(
        cls,
        *,
        model_adapter: ModelAdapter,
        config: BuildConfig,
        device: ExecutionContext,
        request_capacity: int,
        token_capacity: int | None = None,
        page_size: int = 1,
        big_page_tokens: int = 1,
        small_page_capacity: int = 0,
        manager: AttentionStateManager | None = None,
        providers: ProviderRegistry | None = None,
    ) -> "V2ExecutionSession":
        """Build a session without mutating any legacy LightLLM object.

        ``token_capacity`` is required when native attention storage is
        declared.  External providers may omit it because they own their own
        storage and only need the planner/runner binding.
        """

        if request_capacity <= 0:
            raise ValueError("request_capacity must be positive")
        manager = manager or AttentionStateManager(request_capacity=request_capacity)
        provider_registry = providers or device.providers
        prepared = prepare(
            model_adapter=model_adapter,
            providers=provider_registry,
            device=device.device,
            config=config,
        )
        resources = prepared.allocate(manager=manager, device=device.device)

        if not manager._state_registration_finished:
            manager.finish_state_registration()
        if token_capacity is not None:
            if token_capacity <= 0:
                raise ValueError("token_capacity must be positive")
            if manager.get_full_attention_states():
                manager.allocate_storage(
                    token_capacity=token_capacity,
                    page_size=page_size,
                    big_page_tokens=big_page_tokens,
                    small_page_capacity=small_page_capacity,
                    device=device.device.device,
                )
        execution = resources.bind()
        return cls(execution=execution, resources=resources, manager=manager)

    def close(self) -> None:
        """Close the session once callers have drained all async users."""

        if self._closed:
            return
        close_store = getattr(self.manager, "close_execution_stores", None)
        if close_store is not None:
            close_store()
        self._closed = True

    def __enter__(self) -> "V2ExecutionSession":
        if self._closed:
            raise RuntimeError("cannot enter a closed v2 execution session")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
