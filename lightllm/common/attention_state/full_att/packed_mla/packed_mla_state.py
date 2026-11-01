import torch
from lightllm.common.attention_state.state_ops.mla import PackedMLAKVWriter
from lightllm.common.attention_state.full_att.packed_mla.formats import (
    PackedMLAAttentionConfig,
    PackedMLALogicalConfig as PackedMLALogicalConfig,
    packed_mla_spec,
)
from lightllm.common.attention_state.types import PackedMLALayerView as PackedMLALayerView
from ..state import SingleTensorAttentionState


class PackedMLAFullAttentionState(SingleTensorAttentionState):
    def __init__(
        self, layout_config: PackedMLAAttentionConfig, page_size: int = 1, device: str | torch.device = "cuda"
    ):
        super().__init__(
            config=layout_config,
            spec=packed_mla_spec(layout_config),
            writer=PackedMLAKVWriter(layout_config),
            page_size=page_size,
            device=device,
        )
