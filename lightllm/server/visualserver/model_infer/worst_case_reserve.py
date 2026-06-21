import torch
from lightllm.server.visualserver.model_infer.mem_reserve import compute_qwen_worst_case_grid
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_RESERVE_OOM_HINT = (
    "ViT worst-case activation reservation hit OOM. Lower --visual_infer_batch_size, "
    "--max_image_pixels, or --max_image_token_count, or place the ViT on a separate GPU "
    "with --visual_gpu_ids."
)


class WorstCaseReserveMixin:
    """Adds a reserve-and-HOLD worst-case activation probe to a visual model.

    Subclasses MUST implement build_worst_case_input(...). The reservation is held by
    deliberately NOT calling torch.cuda.empty_cache() — the retained allocator high-water
    mark is what the LLM router observes via mem_get_info during KV-pool profiling.
    """

    def build_worst_case_input(self, batch_size, max_image_pixels, max_image_token_count) -> dict:
        raise NotImplementedError

    def run_worst_case_forward(self, dummy: dict):
        return self.forward(**dummy)

    @torch.no_grad()
    def reserve_worst_case_activation(
        self, device_id: int, batch_size: int, max_image_pixels: int, max_image_token_count: int
    ) -> int:
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats(device_id)
        try:
            dummy = self.build_worst_case_input(batch_size, max_image_pixels, max_image_token_count)
            out = self.run_worst_case_forward(dummy)
            del out, dummy
        except (RuntimeError, torch.OutOfMemoryError) as e:
            logger.exception(str(e))
            raise Exception(_RESERVE_OOM_HINT)
        # NB: intentionally NO torch.cuda.empty_cache() here — holding the high-water mark IS the mechanism.
        return int(torch.cuda.max_memory_reserved(device_id))


class QwenVLWorstCaseMixin(WorstCaseReserveMixin):
    """Worst-case builder for Qwen2/2.5/3-VL visual towers (shared forward(hidden_states, grid_thw))."""

    def build_worst_case_input(self, batch_size, max_image_pixels, max_image_token_count) -> dict:
        (total_patches, row_width), grid_thw = compute_qwen_worst_case_grid(
            batch_size=batch_size,
            max_image_pixels=max_image_pixels,
            max_image_token_count=max_image_token_count,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=self.in_channels,
            spatial_merge_size=self.spatial_merge_size,
        )
        # Derive dtype from the loaded weights rather than self.data_type — the latter is not
        # guaranteed to be a torch.dtype on every Qwen visual class; parameters() always is.
        dtype = next(self.parameters()).dtype
        hidden_states = torch.randn((total_patches, row_width), dtype=dtype, device="cuda")
        grid_thw_t = torch.tensor(grid_thw, dtype=torch.long, device="cuda")
        return {"hidden_states": hidden_states, "grid_thw": grid_thw_t}
