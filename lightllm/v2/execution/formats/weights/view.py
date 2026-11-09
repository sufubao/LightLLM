from dataclasses import dataclass
from typing import Any

from lightllm.v2.execution.contracts.state import FormatSpec


@dataclass(frozen=True)
class WeightView:
    format: FormatSpec
    fields: dict[str, Any]
    finalized: bool = True
    owner: Any = None
