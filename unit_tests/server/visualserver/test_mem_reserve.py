import math
import types
from lightllm.server.visualserver.model_infer.mem_reserve import (
    compute_qwen_worst_case_grid,
    publish_vit_reserved_mem,
    read_vit_reserved_mem_for_device,
)


def test_qwen_worst_case_grid_is_bounded_and_square():
    (total_patches, row_width), grid_thw = compute_qwen_worst_case_grid(
        batch_size=2,
        max_image_pixels=8294400,
        max_image_token_count=8192,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
    )
    assert row_width == 3 * 2 * 14 * 14
    assert len(grid_thw) == 2
    for t, h, w in grid_thw:
        assert t == 1
        assert h % 2 == 0 and w % 2 == 0
    side = grid_thw[0][1]
    assert side == 180  # isqrt(32768)=181 -> floor to even -> 180
    assert total_patches == side * side * 2


def test_qwen_worst_case_respects_pixel_cap_when_tighter():
    (_, _), grid_thw = compute_qwen_worst_case_grid(
        batch_size=1,
        max_image_pixels=200704,  # 448*448 -> 1024 patches at patch_size 14
        max_image_token_count=8192,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
    )
    side = grid_thw[0][1]
    assert side * side <= 200704 // (14 * 14)


def test_shared_int_publish_and_read_sums_per_device():
    publish_vit_reserved_mem(device_id=0, global_rank=0, reserved_bytes=1 * 1024**3)
    publish_vit_reserved_mem(device_id=0, global_rank=1, reserved_bytes=2 * 1024**3)
    publish_vit_reserved_mem(device_id=1, global_rank=2, reserved_bytes=5 * 1024**3)

    args = types.SimpleNamespace(
        disable_vision=False,
        enable_multimodal=True,
        visual_gpu_ids=[0, 0, 1],
    )
    assert read_vit_reserved_mem_for_device(args, device_id=0) == 3 * 1024**3
    assert read_vit_reserved_mem_for_device(args, device_id=1) == 5 * 1024**3


def test_read_returns_zero_when_vision_disabled():
    args = types.SimpleNamespace(disable_vision=True, enable_multimodal=False, visual_gpu_ids=[0])
    assert read_vit_reserved_mem_for_device(args, device_id=0) == 0
