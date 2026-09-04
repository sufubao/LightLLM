"""Model support registry.

The registry resolves a complete bundle of model-specific behavior. Legacy
``@ModelRegistry`` registrations are adapted into ``ModelSupport`` objects so
models can be migrated one at a time.
"""

import collections
import importlib
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

T = TypeVar("T")


class UnsupportedModelError(ValueError):
    pass


class ModelSupportConflictError(ValueError):
    pass


@dataclass(frozen=True)
class ModelContext:
    """Normalized, read-only views over a Hugging Face model config."""

    raw_config: Mapping[str, Any]
    model_dir: Optional[str] = None

    @classmethod
    def from_config(cls, model_cfg: Mapping[str, Any], model_dir: Optional[str] = None) -> "ModelContext":
        return cls(raw_config=MappingProxyType(dict(model_cfg)), model_dir=model_dir)

    @property
    def model_type(self) -> str:
        return str(self.raw_config.get("model_type", ""))

    @property
    def architectures(self) -> Tuple[str, ...]:
        architectures = self.raw_config.get("architectures", ())
        if isinstance(architectures, str):
            return (architectures,)
        return tuple(architectures or ())

    @property
    def thinker_config(self) -> Mapping[str, Any]:
        return self._mapping(self.raw_config.get("thinker_config"))

    @property
    def text_config(self) -> Mapping[str, Any]:
        thinker_config = self.thinker_config
        if thinker_config:
            nested = self._mapping(thinker_config.get("text_config"))
            if nested:
                return nested
        for key in ("llm_config", "text_config"):
            nested = self._mapping(self.raw_config.get(key))
            if nested:
                return nested
        return self.raw_config

    @property
    def vision_config(self) -> Mapping[str, Any]:
        thinker_config = self.thinker_config
        if thinker_config:
            nested = self._mapping(thinker_config.get("vision_config"))
            if nested:
                return nested
        return self._mapping(self.raw_config.get("vision_config"))

    @property
    def audio_config(self) -> Mapping[str, Any]:
        thinker_config = self.thinker_config
        if thinker_config:
            nested = self._mapping(thinker_config.get("audio_config"))
            if nested:
                return nested
        return self._mapping(self.raw_config.get("audio_config"))

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any]:
        return MappingProxyType(dict(value)) if isinstance(value, Mapping) else MappingProxyType({})


@dataclass(frozen=True)
class TokenizerBuildContext:
    model: ModelContext
    tokenizer_name: str
    tokenizer: Any
    trust_remote_code: bool
    args: Tuple[Any, ...]
    kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class VisionBuildContext:
    model: ModelContext
    kvargs: Mapping[str, Any]


ModelCondition = Callable[[ModelContext], bool]
TokenizerFactory = Callable[[TokenizerBuildContext], Any]
VisionFactory = Callable[[VisionBuildContext], Any]


@dataclass(frozen=True)
class ModelSupport:
    """All model-specific components needed by framework entry points."""

    name: str
    model_types: Tuple[str, ...]
    text_model: Type
    condition: Optional[ModelCondition] = None
    tokenizer_factory: Optional[TokenizerFactory] = None
    vision_factory: Optional[VisionFactory] = None
    audio_factory: Optional[Callable[..., Any]] = None
    multimodal: bool = False
    priority: int = 100

    def matches(self, context: ModelContext) -> bool:
        return context.model_type in self.model_types and (self.condition is None or self.condition(context))

    @property
    def is_multimodal(self) -> bool:
        return self.multimodal or self.vision_factory is not None or self.audio_factory is not None

    def create_text_model(self, model_kvargs: dict) -> Any:
        return self.text_model(model_kvargs)

    def create_tokenizer(self, context: TokenizerBuildContext) -> Any:
        if self.tokenizer_factory is None:
            return context.tokenizer
        return self.tokenizer_factory(context)

    def create_vision_model(self, context: VisionBuildContext) -> Any:
        if self.vision_factory is None:
            raise UnsupportedModelError(f"Model support {self.name!r} has no vision component")
        return self.vision_factory(context)


@dataclass(frozen=True)
class _LegacyModelRegistration:
    model_class: Type
    condition: Optional[Callable[[dict], bool]] = None
    is_multimodal: bool = False

    def to_support(self, model_type: str) -> ModelSupport:
        condition = None
        if self.condition is not None:
            legacy_condition = self.condition
            condition = lambda context: legacy_condition(dict(context.raw_config))
        return ModelSupport(
            name=f"{self.model_class.__module__}:{self.model_class.__name__}",
            model_types=(model_type,),
            text_model=self.model_class,
            condition=condition,
            multimodal=self.is_multimodal,
            priority=10 if condition is not None else 0,
        )


class ModelRegistryStore:
    def __init__(self) -> None:
        self._legacy_registry: Dict[str, List[_LegacyModelRegistration]] = collections.defaultdict(list)
        self._support_registry: Dict[str, List[ModelSupport]] = collections.defaultdict(list)
        self._lazy_modules: Dict[str, List[str]] = collections.defaultdict(list)
        self._loaded_modules = set()
        self._load_lock = threading.RLock()

    def __call__(
        self,
        model_type: Union[str, Sequence[str]],
        is_multimodal: bool = False,
        condition: Optional[Callable[[dict], bool]] = None,
    ) -> Callable[[T], T]:
        """Compatibility decorator for model classes not migrated yet."""

        def decorator(model_class: T) -> T:
            model_types = (model_type,) if isinstance(model_type, str) else tuple(model_type)
            registration = _LegacyModelRegistration(
                model_class=model_class,
                condition=condition,
                is_multimodal=is_multimodal,
            )
            for current_model_type in model_types:
                self._legacy_registry[current_model_type].append(registration)
            return model_class

        return decorator

    def register_support(self, support: ModelSupport) -> ModelSupport:
        if not support.model_types:
            raise ValueError(f"Model support {support.name!r} must declare at least one model type")
        for model_type in support.model_types:
            if any(item.name == support.name for item in self._support_registry[model_type]):
                raise ValueError(f"Duplicate model support registration: {support.name!r} for {model_type!r}")
            self._support_registry[model_type].append(support)
        return support

    def register_lazy(self, model_type: str, module_name: str) -> None:
        if module_name not in self._lazy_modules[model_type]:
            self._lazy_modules[model_type].append(module_name)

    def resolve(self, model_cfg: Mapping[str, Any], model_dir: Optional[str] = None) -> ModelSupport:
        context = ModelContext.from_config(model_cfg, model_dir=model_dir)
        self._load_modules(context.model_type)

        candidates = [
            support for support in self._support_registry.get(context.model_type, ()) if support.matches(context)
        ]
        candidates.extend(
            registration.to_support(context.model_type)
            for registration in self._legacy_registry.get(context.model_type, ())
            if registration.condition is None or registration.condition(dict(context.raw_config))
        )

        if not candidates:
            raise UnsupportedModelError(f"Model type {context.model_type!r} is not supported")

        highest_priority = max(candidate.priority for candidate in candidates)
        matches = [candidate for candidate in candidates if candidate.priority == highest_priority]
        if len(matches) != 1:
            names = ", ".join(sorted(candidate.name for candidate in matches))
            raise ModelSupportConflictError(
                f"Model type {context.model_type!r} matches multiple supports at priority "
                f"{highest_priority}: {names}"
            )
        return matches[0]

    def get_model(self, model_cfg: Mapping[str, Any], model_kvargs: dict) -> tuple:
        support = self.resolve(model_cfg, model_dir=model_kvargs.get("weight_dir"))
        return support.create_text_model(model_kvargs), support.is_multimodal

    def get_model_class(self, model_cfg: Mapping[str, Any]) -> Type:
        return self.resolve(model_cfg).text_model

    def _load_modules(self, model_type: str) -> None:
        with self._load_lock:
            for module_name in self._lazy_modules.get(model_type, ()):
                if module_name in self._loaded_modules:
                    continue
                importlib.import_module(module_name)
                self._loaded_modules.add(module_name)


ModelRegistry = ModelRegistryStore()
ModelSupportRegistry = ModelRegistry


def _register_builtin_modules() -> None:
    from lightllm.models.builtin_registry import BUILTIN_MODEL_MODULES

    for model_type, module_names in BUILTIN_MODEL_MODULES.items():
        for module_name in module_names:
            ModelRegistry.register_lazy(model_type, module_name)


_register_builtin_modules()


def get_model_support(model_cfg: Mapping[str, Any], model_dir: Optional[str] = None) -> ModelSupport:
    return ModelRegistry.resolve(model_cfg, model_dir=model_dir)


def get_model(model_cfg: Mapping[str, Any], model_kvargs: dict) -> tuple:
    return ModelRegistry.get_model(model_cfg, model_kvargs)


def get_model_class(model_cfg: Mapping[str, Any]) -> Type:
    return ModelRegistry.get_model_class(model_cfg)


def is_reward_model() -> Callable[[Dict[str, Any]], bool]:
    return lambda model_cfg: "RewardModel" in (model_cfg.get("architectures") or [""])[0]


def llm_model_type_is(name: Union[str, Sequence[str]]) -> Callable[[Dict[str, Any]], bool]:
    names = (name,) if isinstance(name, str) else tuple(name)
    return lambda model_cfg: (
        model_cfg.get("llm_config", {}).get("model_type", "") in names
        or model_cfg.get("text_config", {}).get("model_type", "") in names
    )
