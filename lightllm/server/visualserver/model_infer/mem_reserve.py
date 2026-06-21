import math
from typing import List, Tuple
import torch
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_vit_reserved_shm_name(device_id: int, global_rank: int) -> str:
    return f"{get_unique_server_name()}_vit_reserved_mem_d{device_id}_r{global_rank}"


def publish_vit_reserved_mem(device_id: int, global_rank: int, reserved_bytes: int) -> None:
    """Visual rank writes its held worst-case reservation (bytes) for cross-process discovery."""
    shm = SharedInt(get_vit_reserved_shm_name(device_id, global_rank))
    shm.set_value(int(reserved_bytes))


def read_vit_reserved_mem_for_device(args, device_id: int) -> int:
    """Router side: sum reservations of all visual ranks placed on `device_id`. Diagnostic only."""
    if getattr(args, "disable_vision", False) or not getattr(args, "enable_multimodal", False):
        return 0
    gpu_ids = getattr(args, "visual_gpu_ids", None) or []
    total = 0
    for global_rank, dev in enumerate(gpu_ids):
        if dev == device_id:
            total += int(SharedInt(get_vit_reserved_shm_name(dev, global_rank)).get_value())
    return total


def reserve_guard_tensor(device_id: int, reserved_gb: float) -> Tuple[torch.Tensor, int]:
    """Allocate and HOLD a guard tensor of `reserved_gb` GB so the allocator high-water mark persists.
    Returns (tensor, nbytes). The caller MUST keep a reference to the tensor."""
    nbytes = int(reserved_gb * 1024**3)
    guard = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{device_id}")
    return guard, nbytes


def compute_qwen_worst_case_grid(
    batch_size: int,
    max_image_pixels: int,
    max_image_token_count: int,
    patch_size: int,
    temporal_patch_size: int,
    in_channels: int,
    spatial_merge_size: int,
) -> Tuple[Tuple[int, int], List[List[int]]]:
    """Pure shape math for the Qwen-VL worst case.

    Returns ((total_patches, row_width), grid_thw) where pixel_values has shape
    (total_patches, row_width) and grid_thw is one [t, h, w] triple per dummy image.
    Bounds each image by BOTH the per-image token cap and pixel cap (whichever is tighter),
    using a near-square grid whose sides are multiples of spatial_merge_size.
    """
    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    patches_by_tokens = max_image_token_count * spatial_merge_unit
    patches_by_pixels = max_image_pixels // (patch_size * patch_size)
    max_patches = max(1, min(patches_by_tokens, patches_by_pixels))

    side = int(math.isqrt(max_patches))
    side -= side % spatial_merge_size
    side = max(side, spatial_merge_size)

    grid_h = grid_w = side
    row_width = in_channels * temporal_patch_size * patch_size * patch_size
    total_patches = grid_h * grid_w * batch_size
    grid_thw = [[1, grid_h, grid_w] for _ in range(batch_size)]
    return (total_patches, row_width), grid_thw
