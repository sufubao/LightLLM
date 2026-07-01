from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import (
    Qwen35MOETransformerLayerWeight,
)
from lightllm.models.qwen3_5_mtp.layer_weights.transformer_layer_weight import (
    ATTN_NORM_NAME_ATTRS,
    MTP_LAYER_PREFIX,
    init_mtp_qkv_weight,
    retarget_mtp_layer_attrs,
)


class Qwen3_5MoeMTPTransformerLayerWeight(Qwen35MOETransformerLayerWeight):
    def _init_qkv(self):
        init_mtp_qkv_weight(self)

    def _fused_expert_layer_prefix(self):
        return f"{MTP_LAYER_PREFIX}{self.layer_num_}."

    def _moe_layer_prefix(self):
        return f"{MTP_LAYER_PREFIX}{self.layer_num_}.mlp"

    def _init_weight_names(self):
        super()._init_weight_names()
        retarget_mtp_layer_attrs(self, ATTN_NORM_NAME_ATTRS)
