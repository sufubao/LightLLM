import functools
import importlib.util
import os

import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import (
    chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class FlaGdnPrefillBackend:
    def __call__(self, q, k, v, **kwargs):
        return _fla_chunk_gated_delta_rule(q=q, k=k, v=v, **kwargs)


class FlashQlaGdnPrefillBackend:
    _SUPPORTED_CAPABILITIES = {(9, 0), (10, 0), (12, 0)}

    def __init__(self, impl):
        self.impl = impl
        self.fallback = FlaGdnPrefillBackend()
        self.initialized = False
        self.disabled = False

    @classmethod
    def is_available(cls, head_k_dim, head_v_dim):
        if os.environ.get("FLA_FLASH_QLA", "1") == "0":
            return False, "disabled by FLA_FLASH_QLA=0"
        if head_k_dim != 128 or head_v_dim != 128:
            return False, f"requires K=V=128, got K={head_k_dim}, V={head_v_dim}"
        if not torch.cuda.is_available():
            return False, "CUDA is unavailable"
        capability = torch.cuda.get_device_capability()
        if capability not in cls._SUPPORTED_CAPABILITIES:
            return False, f"requires SM90, SM100, or SM120, got SM{capability[0]}{capability[1]}"
        if importlib.util.find_spec("flash_qla") is None:
            return False, "flash_qla is not installed"
        return True, None

    @staticmethod
    def supports(q, k, v):
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            return False
        if q.dtype not in (torch.float16, torch.bfloat16) or not (q.dtype == k.dtype == v.dtype):
            return False
        if q.shape[-1] != 128 or v.shape[-1] != 128:
            return False
        if v.shape[2] % k.shape[2] != 0:
            return False
        if torch.cuda.get_device_capability(q.device) == (12, 0) and q.dtype == torch.float16:
            return False
        return True

    def __call__(self, q, k, v, **kwargs):
        if self.disabled or not self.supports(q, k, v):
            return self.fallback(q=q, k=k, v=v, **kwargs)

        try:
            result = self.impl(q=q, k=k, v=v, **kwargs)
        except Exception as exc:
            if self.initialized:
                raise
            self.disabled = True
            logger.warning(f"FlashQLA initialization failed ({exc!r}); falling back to the FLA Triton kernel.")
            return self.fallback(q=q, k=k, v=v, **kwargs)

        self.initialized = True
        return result


@functools.lru_cache(maxsize=None)
def get_gdn_prefill_backend(head_k_dim, head_v_dim):
    available, reason = FlashQlaGdnPrefillBackend.is_available(head_k_dim, head_v_dim)
    if not available:
        logger.info(f"GDN chunked-prefill backend: FLA ({reason}).")
        return FlaGdnPrefillBackend()

    try:
        from flash_qla import chunk_gated_delta_rule
    except Exception as exc:
        logger.warning(f"Importing FlashQLA failed ({exc!r}); falling back to the FLA Triton kernel.")
        return FlaGdnPrefillBackend()

    logger.info("GDN chunked-prefill backend: FlashQLA.")
    return FlashQlaGdnPrefillBackend(chunk_gated_delta_rule)
