from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import (
    Qwen35MOETransformerLayerWeight,
)
from lightllm.models.qwen3_5_mtp.layer_weights.transformer_layer_weight import (
    Qwen3_5MTPTransformerLayerWeightMixin,
)


class Qwen3_5MoeMTPTransformerLayerWeight(Qwen3_5MTPTransformerLayerWeightMixin, Qwen35MOETransformerLayerWeight):
    def _fused_expert_layer_prefix(self):
        return f"{self._MTP_PREFIX}{self.layer_num_}."

    def _moe_layer_prefix(self):
        return f"{self._MTP_PREFIX}{self.layer_num_}.mlp"

    def _init_weight_names(self):
        super()._init_weight_names()
        self._retarget_attn_norm_names()
