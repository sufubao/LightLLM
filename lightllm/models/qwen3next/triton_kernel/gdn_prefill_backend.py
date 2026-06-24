import functools
import torch

from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models.qwen3next.triton_kernel.fla.ops import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule

logger = init_logger(__name__)


@functools.lru_cache(maxsize=1)
def get_gdn_prefill_chunk_fn():
    """Resolve the GDN chunked-prefill kernel once per process.

    Returns the vendored flash-linear-attention triton kernel by default, or the TileLang
    FlashQLA kernel when ``--gdn_prefill_backend flashqla`` is set AND the hardware is Hopper
    (SM90+) AND the ``flash_qla`` package imports. Falls back to FLA with a warning otherwise.

    FlashQLA's high-level ``chunk_gated_delta_rule`` is signature- and result-compatible with the
    vendored one (same args incl. ``use_qk_l2norm_in_kernel`` / ``cu_seqlens`` / ``head_first`` and
    same default ``scale = 1/sqrt(K)``), so the call site needs no other change.
    """
    backend = getattr(get_env_start_args(), "gdn_prefill_backend", "fla")
    if backend != "flashqla":
        return _fla_chunk_gated_delta_rule

    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        logger.warning(
            f"gdn_prefill_backend=flashqla requires Hopper (SM90+), got SM{cap[0]}{cap[1]}; "
            "falling back to the FLA triton kernel."
        )
        return _fla_chunk_gated_delta_rule

    try:
        from flash_qla import chunk_gated_delta_rule as _flashqla_chunk_gated_delta_rule
    except Exception as e:
        logger.warning(
            f"gdn_prefill_backend=flashqla but importing flash_qla failed ({e!r}); "
            "falling back to the FLA triton kernel. Install FlashQLA (https://github.com/QwenLM/FlashQLA)."
        )
        return _fla_chunk_gated_delta_rule

    def _flashqla_chunk(q, k, v, **kwargs):
        # FlashQLA's l2norm / gemm kernels require the last dim to be contiguous (stride==1),
        # but q/k/v coming from _rearrange_mixed_qkv are not. The vendored FLA kernel tolerates
        # the strided layout; FlashQLA asserts. Make them contiguous (a no-op when already so).
        return _flashqla_chunk_gated_delta_rule(q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), **kwargs)

    logger.info("GDN chunked-prefill backend: FlashQLA (TileLang, Hopper).")
    return _flashqla_chunk
