"""Independent device-runtime and operator-backend boundaries.

CUDA remains the default. Importing these APIs does not import CUDA kernels or
probe accelerator availability. Select providers before importing model code.
"""

from typing import TYPE_CHECKING

from ._registry import BackendRegistry

if TYPE_CHECKING:
    from .runtime import DeviceRuntime
    from .operators import OperatorBackend


def _cuda_runtime():
    from .cuda.runtime import CudaRuntime

    return CudaRuntime()


def _cuda_operators():
    from .cuda.operators import CudaOperatorBackend

    return CudaOperatorBackend()


_runtimes = BackendRegistry("device runtime", _cuda_runtime)
_operators = BackendRegistry("operator", _cuda_operators)


def register_device_runtime(name, factory):
    _runtimes.register(name, factory)


def select_device_runtime(name):
    _runtimes.select(name)


def get_device_runtime() -> "DeviceRuntime":
    return _runtimes.get()


def register_operator_backend(name, factory):
    _operators.register(name, factory)


def select_operator_backend(name):
    _operators.select(name)


def get_operator_backend() -> "OperatorBackend":
    return _operators.get()
