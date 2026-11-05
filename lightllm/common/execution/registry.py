"""Registration and capability matching for execution implementations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from lightllm.common.execution.compute.attention.backend import AttentionBackend
from lightllm.common.execution.compute.attention.backend import AttentionSpec, AttentionParams
from lightllm.common.execution.contracts.device import DeviceCaps, DeviceRuntime
from lightllm.common.execution.contracts.provider import Phase
from lightllm.common.execution.contracts.state import FormatSpec, StateSpec, StateStore, StateView, StateWriter
from lightllm.common.execution.contracts.transfer import TransferRequest

if TYPE_CHECKING:
    from lightllm.common.execution.planner.plan import AttentionRequest


AttentionFactory: TypeAlias = Callable[[Phase], AttentionBackend]
StateStoreFactory: TypeAlias = Callable[
    [tuple[StateSpec, ...], FormatSpec, object | None, DeviceRuntime, Mapping[int, AttentionSpec]], StateStore
]
StateWriterFactory: TypeAlias = Callable[[StateView], StateWriter]
StateFormatSupport: TypeAlias = Callable[[FormatSpec], bool]
OperationFactory: TypeAlias = Callable[[object, object, DeviceRuntime, Iterable[object]], object]
TransferSupport: TypeAlias = Callable[[TransferRequest, DeviceCaps], bool]
TransferFactory: TypeAlias = Callable[[TransferRequest, DeviceRuntime], object]
ProviderFactory: TypeAlias = Callable[[], object]

__all__ = [
    "AttentionProvider",
    "StateProvider",
    "TransferProvider",
    "ProviderRegistry",
    "default_providers",
]


@dataclass(frozen=True)
class AttentionProvider:
    """Factory for one attention implementation across selected phases."""

    create: AttentionFactory
    phases: tuple[Phase, ...] = ("prefill", "decode", "verify")


@dataclass(frozen=True)
class StateProvider:
    """Storage and append-writer implementation for one format family."""

    layout: str
    retention: str
    create_store: StateStoreFactory
    create_writer: StateWriterFactory | None = None
    supports: StateFormatSupport | None = None


@dataclass(frozen=True)
class TransferProvider:
    """Capability predicate and binder for one transfer path."""

    supports: TransferSupport
    bind: TransferFactory


class ProviderRegistry:
    """Provider snapshot used during planning and binding."""

    def __init__(
        self,
        attention: Iterable[AttentionProvider] = (),
        storage: Iterable[StateProvider] = (),
        *,
        linear: Iterable[ProviderFactory] = (),
        moe: Iterable[ProviderFactory] = (),
        expert: Iterable[ProviderFactory] = (),
        weight_finalizers: Iterable[Callable[..., object]] = (),
        operations: Mapping[str, OperationFactory] | Iterable[tuple[str, OperationFactory]] = (),
        transfers: Iterable[TransferProvider] = (),
    ):
        self.transfers = tuple(transfers)
        self.operations = dict(operations)
        self.weight_finalizers = tuple(weight_finalizers)
        self.attention = tuple(attention)
        self.storage = tuple(storage)
        self.linear, self.moe, self.expert = tuple(linear), tuple(moe), tuple(expert)

    def bind_operation(
        self,
        name: str,
        spec: object,
        resources: object,
        device: DeviceRuntime,
        communication: Iterable[object] = (),
    ) -> object | None:
        bind = self.operations.get(name)
        return None if bind is None else bind(spec, resources, device, communication)

    def transfer_driver(self, request: TransferRequest, runtime: DeviceRuntime) -> object:
        matches = [provider for provider in self.transfers if provider.supports(request, runtime.caps)]
        if not matches:
            raise ValueError(f"unsupported transfer path: {request}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous transfer providers: {request}")
        return matches[0].bind(request, runtime)

    @staticmethod
    def validate_attention_requests(requests: Iterable[AttentionRequest], backends: Iterable[AttentionBackend]) -> None:
        for request in requests:
            if request.requested == "auto":
                continue
            if not any(
                backend.phase == request.phase and request.requested in (backend.name, backend.implementation)
                for backend in backends
            ):
                raise ValueError(f"{request.phase} attention backend is not registered: {request.requested}")

    def attention_backends(self, phases: Iterable[Phase]) -> tuple[AttentionBackend, ...]:
        return tuple(
            provider.create(phase) for provider in self.attention for phase in phases if phase in provider.phases
        )

    def state_provider(self, format_spec: FormatSpec, retention: str) -> StateProvider:
        matches = [
            provider
            for provider in self.storage
            if provider.layout == format_spec.layout
            and provider.retention == retention
            and (provider.supports is None or provider.supports(format_spec))
        ]
        identity = format_spec.layout, retention, format_spec.encoding
        if not matches:
            raise ValueError(f"state provider is not registered: {identity}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous state providers: {identity}")
        return matches[0]


def default_providers() -> ProviderRegistry:
    from lightllm.common.execution.compute.attention.dense.fa3 import FA3AttentionBackend
    from lightllm.common.execution.compute.attention.dense.triton import TritonAttentionBackend
    from lightllm.common.execution.compute.attention.dense.group_quantized import GroupQuantizedAttentionBackend
    from lightllm.common.execution.compute.attention.dense.static_fp8 import StaticFP8AttentionBackend
    from lightllm.common.execution.compute.attention.dense.flashinfer import FlashInferAttentionBackend
    from lightllm.common.execution.compute.attention.gdn.backend import GatedDeltaNetBackend
    from lightllm.common.execution.compute.attention.kda.backend import KimiDeltaAttentionBackend
    from lightllm.common.execution.compute.attention.mla.backend import MLAAttentionBackend
    from lightllm.common.execution.compute.attention.mla.flashinfer import FlashInferMLABackend
    from lightllm.common.execution.compute.attention.sparse_mla.backend import SparseMLAAttentionBackend
    from lightllm.common.execution.compute.attention.pooled_nope.backend import PooledNoPEAttentionBackend
    from lightllm.common.execution.formats.providers import state_providers

    from lightllm.common.execution.compute.linear.backend import LinearBackend
    from lightllm.common.execution.compute.moe.backend import MoeBackend
    from lightllm.common.execution.compute.moe.expert import ExpertComputeBackend
    from functools import partial

    return ProviderRegistry(
        (
            AttentionProvider(FA3AttentionBackend),
            AttentionProvider(TritonAttentionBackend),
            AttentionProvider(GroupQuantizedAttentionBackend),
            AttentionProvider(StaticFP8AttentionBackend),
            AttentionProvider(FlashInferAttentionBackend),
            *(
                AttentionProvider(lambda phase, name=name: GatedDeltaNetBackend(name, phase))
                for name in ("flashqla", "triton")
            ),
            *(
                AttentionProvider(lambda phase, name=name: MLAAttentionBackend(name, phase))
                for name in ("fa3", "triton")
            ),
            AttentionProvider(FlashInferMLABackend),
            *(
                AttentionProvider(
                    lambda phase, name=name: SparseMLAAttentionBackend(name, phase), ("prefill", "decode")
                )
                for name in ("flashmla_sparse", "triton")
            ),
            AttentionProvider(KimiDeltaAttentionBackend),
            *(
                AttentionProvider(lambda phase, name=name: PooledNoPEAttentionBackend(name, phase))
                for name in ("flashmla_sparse", "triton")
            ),
        ),
        state_providers(),
        linear=(partial(LinearBackend, name) for name in LinearBackend.implementations),
        moe=(MoeBackend,),
        expert=(ExpertComputeBackend,),
    )
