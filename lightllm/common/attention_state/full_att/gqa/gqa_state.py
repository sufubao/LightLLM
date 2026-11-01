import torch
from lightllm.common.attention_state.state_ops.gqa import PlainKVWriter
from .formats import GQAAttentionConfig, plain_spec
from .state import GQAStorageState


class GQAFullAttentionState(GQAStorageState):
    def __init__(self, layout_config: GQAAttentionConfig, page_size: int = 1, device: str | torch.device = "cuda"):
        super().__init__(layout_config, plain_spec(layout_config), PlainKVWriter(layout_config), page_size, device)
