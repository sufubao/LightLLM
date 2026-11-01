import torch
from lightllm.common.attention_state.state_ops.gqa import StaticFP8KVWriter
from .formats import StaticFP8AttentionConfig, static_fp8_spec
from .state import GQAStorageState


class StaticFP8FullAttentionState(GQAStorageState):
    def __init__(
        self, layout_config: StaticFP8AttentionConfig, page_size: int = 1, device: str | torch.device = "cuda"
    ):
        super().__init__(
            layout_config, static_fp8_spec(layout_config), StaticFP8KVWriter(layout_config), page_size, device
        )

    def bind_calibration(self, calibration, storage_layer_start: int) -> None:
        self.writer.bind_calibration(calibration, storage_layer_start)

    @property
    def scales(self):
        return self.writer.scales

    @property
    def q_scales(self):
        return self.writer.q_scales

    @property
    def cpu_scales(self):
        return self.writer.cpu_scales
