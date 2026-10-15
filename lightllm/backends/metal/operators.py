"""Metal kernels; model execution and cache ownership stay in LightLLM."""
from importlib import import_module
from functools import lru_cache
from lightllm.backends.operators import OperatorBackend
from .bindings import KERNEL_BINDINGS


class MetalOperatorBackend(OperatorBackend):
    name = "metal"

    def resolve(self, name):
        from . import kernels

        # Logical names retain the existing signatures; never load the CUDA
        # modules themselves. Unsupported operations fail when selected.
        attr = KERNEL_BINDINGS.get(name)
        if attr is None:

            def unsupported(*args, **kwargs):
                raise NotImplementedError(f"Metal operator {name!r} is unavailable")

            return unsupported
        return getattr(kernels, attr)

    @lru_cache(maxsize=None)
    def selection_module(self, family):
        if family not in ("attention", "linear_attention"):
            raise NotImplementedError(f"Metal does not implement {family}")
        return import_module(f"{__package__}.{family}")

    def select_weight_forward(self, op):
        # Norm signatures, including gated RMS, are shared by the device
        # implementations and dispatch to this backend's native Torch kernels.
        from lightllm.common.basemodel.layer_weights.meta_weights.norm_weight import RMSNormWeight, QKRMSNORMWeight

        if isinstance(op, (RMSNormWeight, QKRMSNORMWeight)):
            return op._triton_forward

        def forward(*args, **kwargs):
            kwargs.pop("alloc_func", None)
            return op._native_forward(*args, **kwargs)

        return forward

    def initialize_linear_attention(self):
        # Torch owns all intermediate allocations on Metal.
        return None
