from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
)
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import (
    rename_weight_keys,
)

from .hyperconnection import Qwen4ExpGatedResidualWeight


class Qwen4ExpPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]
        self.wte_weight_ = EmbeddingWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="model.embed_tokens.weight",
            data_type=data_type,
        )
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="lm_head.weight",
            data_type=data_type,
            embedding_weight=self.wte_weight_
            if network_config.get("tie_word_embeddings", False)
            else None,
        )
        self.final_mixer = Qwen4ExpGatedResidualWeight(
            prefix="model.hyper_connection_mixer",
            hidden_size=hidden_size,
            hc_count=network_config["hc_count"],
            hc_lowrank=network_config["hc_lowrank"],
            data_type=data_type,
            use_combine=False,
        )

    def load_hf_weights(self, weights):
        rename_weight_keys(weights)
        super().load_hf_weights(weights)
