import functools
import os

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import (
    chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
)
from lightllm.utils.backend_validator import validate
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def _load_flashqla():
    from flash_qla import chunk_gated_delta_rule

    return chunk_gated_delta_rule


_GDN_PREFILL_BACKENDS = (("flashqla", _load_flashqla),)


@functools.lru_cache(maxsize=None)
def get_gdn_prefill_backend(num_k_heads, num_v_heads, head_k_dim, head_v_dim, qkv_dtype, state_dtype):
    if os.environ.get("FLA_FLASH_QLA", "1") != "0":
        backend_args = (num_k_heads, num_v_heads, head_k_dim, head_v_dim, qkv_dtype, state_dtype)
        for backend_name, load_backend in _GDN_PREFILL_BACKENDS:
            if validate(backend_name, *backend_args):
                logger.info(f"GDN chunked-prefill backend: {backend_name} (validated).")
                return load_backend()

    logger.info("GDN chunked-prefill backend: FLA.")
    return _fla_chunk_gated_delta_rule
