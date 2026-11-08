from dataclasses import dataclass
from operator import index
from typing import Any, Iterable, Protocol


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _tuple(values: Iterable[Any], name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    try:
        return tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc


class StateParams(Protocol):
    def validate(self) -> None:
        ...


@dataclass(frozen=True)
class StateSpec:
    """Immutable identity and ownership declaration for one execution state."""

    role: str
    resource: str
    producer: int
    consumers: tuple[int, ...]
    params: StateParams
    retention: str = "history"

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not isinstance(self.resource, str) or not self.role or not self.resource:
            raise ValueError("state identity and consumers are required")
        if not isinstance(self.retention, str) or not self.retention:
            raise ValueError("state retention must be a non-empty string")
        producer = _integer(self.producer, "state producer")
        consumers = tuple(_integer(value, "state consumer") for value in _tuple(self.consumers, "state consumers"))
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "consumers", consumers)
        if not consumers:
            raise ValueError("state identity and consumers are required")
        if min(consumers) < 0:
            raise ValueError("state needs nonnegative layer IDs")
        if producer not in consumers or len(set(consumers)) != len(consumers):
            raise ValueError("state consumers must be unique and include the producer")
        validate = getattr(self.params, "validate", None)
        if not callable(validate):
            raise TypeError("state params must provide validate()")
        validate()

    @property
    def key(self) -> tuple[str, str]:
        return self.role, self.resource


@dataclass(frozen=True)
class FormatSpec:
    """Physical representation of a state exposed to an execution provider."""

    encoding: str
    dtype: Any
    layout: str
    fields: tuple[str, ...]
    page_size: int
    shard: tuple[int, int] = (0, 1)
    version: int = 1
    calibration: str | None = None
    group_size: int | None = None
    scale_dtype: Any = None
    field_dtypes: tuple[Any, ...] = ()
    candidate_slots: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.encoding, str)
            or not isinstance(self.layout, str)
            or not self.encoding
            or not self.layout
        ):
            raise ValueError("format needs an encoding, layout and unique fields")
        fields = tuple(self.fields)
        object.__setattr__(self, "fields", fields)
        if (
            not fields
            or any(not isinstance(field, str) or not field for field in fields)
            or len(set(fields)) != len(fields)
        ):
            raise ValueError("format needs an encoding, layout and unique fields")
        page_size = _integer(self.page_size, "format page size")
        version = _integer(self.version, "format version")
        rank = _integer(self.shard[0], "format shard rank")
        size = _integer(self.shard[1], "format shard size")
        candidate_slots = _integer(self.candidate_slots, "format candidate slots")
        object.__setattr__(self, "page_size", page_size)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "shard", (rank, size))
        object.__setattr__(self, "candidate_slots", candidate_slots)
        field_dtypes = tuple(self.field_dtypes)
        object.__setattr__(self, "field_dtypes", field_dtypes)
        if page_size <= 0 or version <= 0 or size <= 0 or not 0 <= rank < size:
            raise ValueError("invalid page size, format version or shard")
        if self.group_size is not None and (self.group_size < 2 or self.group_size & (self.group_size - 1)):
            raise ValueError("quantization group size must be a power of two >= 2")
        if field_dtypes and len(field_dtypes) != len(fields):
            raise ValueError("field dtypes must describe every field")
        if candidate_slots <= 0:
            raise ValueError("state candidate slots must be positive")


@dataclass(frozen=True)
class StateView:
    """Read-only view binding a logical state to its physical fields."""

    state: StateSpec
    format: FormatSpec
    fields: tuple[Any, ...]

    def __post_init__(self) -> None:
        fields = _tuple(self.fields, "view fields")
        object.__setattr__(self, "fields", fields)
        if len(fields) != len(self.format.fields):
            raise ValueError("view fields must match its format")


class StateStore(Protocol):
    """Lifecycle owned by the execution layer for one logical state family.

    The store is deliberately smaller than the native attention-state API. It
    owns allocation and release, while the native state remains responsible
    for kernels, transfers, and layout-specific operations.
    """

    def allocate_storage(self, token_capacity: int) -> None:
        """Create the physical storage described by the bound state."""
        ...

    def view_for(self, logical_layer_id: int) -> StateView:
        """Return a view only after storage has been allocated."""
        ...

    def close(self) -> None:
        """Release owned storage after all users have stopped reading it."""
        ...


@dataclass(frozen=True)
class LayerStates:
    primary: StateSpec
    auxiliary: tuple[tuple[str, StateSpec], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.primary, StateSpec):
            raise TypeError("primary state must be a StateSpec")
        auxiliary = _tuple(self.auxiliary, "auxiliary states")
        names, keys = set(), set()
        normalized = []
        for item in auxiliary:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("auxiliary states must be (name, state) pairs")
            name, state = item
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("auxiliary state names must be non-empty and unique")
            if not isinstance(state, StateSpec) or state.key in keys:
                raise ValueError("auxiliary state resources must be unique StateSpec values")
            names.add(name)
            keys.add(state.key)
            normalized.append((name, state))
        object.__setattr__(self, "auxiliary", tuple(normalized))
