import torch
from lightllm.common.attention_state.state_ops.mla import CompressedKVWriter
from lightllm.common.attention_state.full_att.compressed_att.formats import CompressedAttentionConfig, compressed_spec
from lightllm.common.attention_state.types import CompressedAttentionLayerView as CompressedAttentionLayerView
from ..state import SingleTensorAttentionState
from lightllm.common.attention_state.transfer.token import BlockTokenTransfer


class CompressedAttentionState(SingleTensorAttentionState):
    def __init__(
        self, layout_config: CompressedAttentionConfig, page_size: int = 1, device: str | torch.device = "cuda"
    ):
        super().__init__(
            config=layout_config,
            spec=compressed_spec(layout_config),
            writer=CompressedKVWriter(layout_config),
            page_size=page_size,
            device=device,
        )
        self.transfer = BlockTokenTransfer(self.resource)


def create_compressed_att_state(
    config: CompressedAttentionConfig,
    page_size: int = 256,
    device: str | torch.device = "cuda",
) -> CompressedAttentionState:
    """根据压缩比例创建历史 KV 定义；不分配存储，也不创建 SWA/compressor 运行态。"""
    return CompressedAttentionState(layout_config=config, page_size=page_size, device=device)
