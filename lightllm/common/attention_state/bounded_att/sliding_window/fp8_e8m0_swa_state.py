from functools import partial
import torch
from lightllm.common.attention_state.state_ops.sliding_window import SlidingWindowRuntime
from lightllm.common.attention_state.bounded_att.sliding_window.formats import (
    FP8E8M0SlidingWindowConfig,
    sliding_window_spec,
)
from ..state import WindowBoundedState
from lightllm.common.attention_state.storage.page_pool import WindowPagesPool


class FP8E8M0SlidingWindowState(WindowBoundedState):
    def __init__(
        self,
        config: FP8E8M0SlidingWindowConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
    ):
        pool = partial(
            WindowPagesPool,
            snapshot_shape=(len(config.logical_layer_ids), config.window_size, config.kv_bytes),
            dtype=torch.uint8,
        )
        super().__init__(
            config=config,
            spec=sliding_window_spec(config, request_capacity, mtp_step),
            runtime=SlidingWindowRuntime(config),
            request_capacity=request_capacity,
            mtp_step=mtp_step,
            device=device,
            page_pool_factory=pool,
        )
        self.swa_capacity_tokens = self.swa_buffer.shape[2]
        self.HOLD_REQUEST_ID = request_capacity - 1
        self.HOLD_SWA_SLOT = self.HOLD_REQUEST_ID * self.swa_capacity_tokens

    @property
    def swa_buffer(self):
        return self.resource.buffers.get("swa_buffer")

    @swa_buffer.setter
    def swa_buffer(self, value):
        self.resource.buffers["swa_buffer"] = value


def create_sliding_window_state(
    config: FP8E8M0SlidingWindowConfig,
    request_capacity: int,
    mtp_step: int = 0,
    device: str | torch.device = "cuda",
) -> FP8E8M0SlidingWindowState:
    return FP8E8M0SlidingWindowState(
        config=config,
        request_capacity=request_capacity,
        mtp_step=mtp_step,
        device=device,
    )
