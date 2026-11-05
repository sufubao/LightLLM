from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from lightllm.common.execution.contracts.state import StateSpec, LayerStates
from lightllm.common.execution.compute.attention.backend import AttentionSpec, AttentionParams
from lightllm.common.execution.contracts.device import DeviceRuntime
from lightllm.common.execution.compute.moe.spec import MoeSpec
from lightllm.common.execution.contracts.provider import Phase


PreT = TypeVar("PreT", covariant=True)
PostT = TypeVar("PostT", covariant=True)
LayerT = TypeVar("LayerT", covariant=True)


@dataclass(frozen=True)
class ModelRecipe(Generic[PreT, PostT, LayerT]):
    """Constructed model objects, with layers in forward order.

    Constructors and semantic validation have completed. Weights, runtime
    operations and attention resources must be bound before execution.
    """

    pre: PreT
    post: PostT
    layers: tuple[LayerT, ...]


@dataclass(frozen=True)
class ModelWeights(Generic[PreT, LayerT]):
    pre_post: PreT
    layers: tuple[LayerT, ...]


class ModelAdapter(Protocol):
    """Model-owned semantic declarations consumed by the execution builder.

    The adapter owns model-specific checkpoint and layer objects.  The core
    only consumes these typed declarations and never inspects a model config
    or weight object's private fields.
    """

    attention_phases: tuple[Phase, ...]

    def moe_specs(self) -> Mapping[int, MoeSpec]:
        ...

    def attention_specs(self) -> Mapping[int, AttentionSpec]:
        ...

    def state_specs(self) -> tuple[StateSpec, ...]:
        ...

    def attention_states(self) -> Mapping[int, LayerStates]:
        ...

    def build_recipe(self) -> ModelRecipe[object, object, object]:
        ...

    def build_weights(self, quant_config: object) -> ModelWeights[object, object]:
        ...

    def load_weights(
        self,
        weights: ModelWeights[object, object],
        directory: str,
        weight_dict: Mapping[str, object] | None = None,
    ) -> None:
        ...

    def attention_params(self, weights: Mapping[int, object], device: DeviceRuntime) -> Mapping[int, AttentionParams]:
        ...
