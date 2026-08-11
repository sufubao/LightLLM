import functools
import os

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import (
    chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
)
from lightllm.utils.backend_validator import validate
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@functools.lru_cache(maxsize=None)
def get_gdn_prefill_backend(num_k_heads, num_v_heads, head_k_dim, head_v_dim, qkv_dtype, state_dtype):
    if os.environ.get("FLA_FLASH_QLA", "1") != "0" and validate(
        "flashqla", num_k_heads, num_v_heads, head_k_dim, head_v_dim, qkv_dtype, state_dtype
    ):
        from flash_qla import chunk_gated_delta_rule

        logger.info("GDN chunked-prefill backend: FlashQLA (validated).")
        return chunk_gated_delta_rule

    logger.info("GDN chunked-prefill backend: FLA.")
    return _fla_chunk_gated_delta_rule
