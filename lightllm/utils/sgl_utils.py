from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    import sgl_kernel

    sgl_ops = sgl_kernel
    sgl_allreduce_ops = sgl_ops.allreduce
    HAS_SGL_KERNEL = True
except:
    sgl_ops = None
    sgl_allreduce_ops = None
    HAS_SGL_KERNEL = False
    logger.warning(
        "sgl_kernel is not installed, you can't use the api of it. \
                   You can solve it by running `pip install sgl_kernel`."
    )

try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    flash_attn_varlen_func = flash_attn_varlen_func
    flash_attn_with_kvcache = flash_attn_with_kvcache
    merge_state_v2 = sgl_ops.merge_state_v2
except:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    merge_state_v2 = None
    logger.warning(
        "sgl_kernel is not installed, or the installed version did not support fa3. \
        Try to upgrade it."
    )

# Imported separately: older sgl_kernel builds ship fa3 (flash_attn_varlen_func) without
# get_scheduler_metadata. Keeping it on the import line above would null out the whole fa3
# backend on those builds; fa3/fp.py already guards against get_scheduler_metadata being None.
try:
    from sgl_kernel.flash_attn import get_scheduler_metadata

    get_scheduler_metadata = get_scheduler_metadata
except:
    get_scheduler_metadata = None
