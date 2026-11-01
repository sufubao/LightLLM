from lightllm.common.attention_state.full_att.base_state import FullAttentionState
import torch
from lightllm.common.attention_state.state_ops.mla import MLAKVWriter
from .formats import MLAConfig, mla_spec
from ..state import SingleTensorAttentionState


class MLAFullAttentionState(SingleTensorAttentionState):
    def __init__(self, layout_config: MLAConfig, page_size: int = 1, device: str | torch.device = "cuda"):
        super().__init__(
            config=layout_config,
            spec=mla_spec(layout_config),
            writer=MLAKVWriter(layout_config),
            page_size=page_size,
            device=device,
        )


def create_mla_state(
    config: MLAConfig,
    page_size: int = 1,
    device: str | torch.device = "cuda",
) -> FullAttentionState:
    """Create a normal latent MLA State without indexer fields."""
    assert type(config) is MLAConfig, "normal MLA creation requires MLAConfig"
    return MLAFullAttentionState(layout_config=config, page_size=page_size, device=device)
