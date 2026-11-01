"""注意力 State 的命名和传输视图工具。"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import WRAPPER_ASSIGNMENTS, wraps
from typing import ParamSpec, TypeVar

import torch

_P = ParamSpec("_P")
_R = TypeVar("_R")

MAX_DRAFT_MODELS = 36
MODEL_ROLES = ("main", *(f"draft{i}" for i in range(MAX_DRAFT_MODELS)))


# ========================================================================
# 一、State 命名与返回值副本
# ========================================================================


def dtype_name(dtype: torch.dtype):
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]


def deepcopy_result(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """每次返回结果的独立深拷贝，放在 lru_cache 外层以保护缓存原对象。

    保留被装饰函数的缓存管理接口，调用方仍可使用 cache_clear/cache_info/cache_parameters。
    """

    @wraps(func, assigned=WRAPPER_ASSIGNMENTS + ("cache_clear", "cache_info", "cache_parameters"))
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return deepcopy(func(*args, **kwargs))

    return wrapped
