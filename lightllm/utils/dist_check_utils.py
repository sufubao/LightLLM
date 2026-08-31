"""Cheap startup checks for optional all-reduce fast paths."""

from typing import TYPE_CHECKING

from lightllm.utils.device_utils import has_cuda_compiler
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from lightllm.server.core.objs.start_args_type import StartArgs

logger = init_logger(__name__)


def auto_configure_allreduce_flags_from_args(args: "StartArgs") -> None:
    """Keep correctness fallbacks local to the real process group instead of probing two GPUs."""
    if args.hardware_platform != "cuda":
        return

    if not args.disable_flashinfer_allreduce and not has_cuda_compiler():
        args.disable_flashinfer_allreduce = True
        logger.info("Auto-set disable_flashinfer_allreduce=True because the CUDA compiler is unavailable.")
