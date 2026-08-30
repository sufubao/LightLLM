import torch

from lightllm.common.basemodel.layer_weights.meta_weights import (
    QKGEMMANormWeight,
    RMSNormWeight,
    ROWMMWeight,
)
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import (
    BaseWeightTpl,
)
from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import (
    Qwen35MOETransformerLayerWeight,
)

from .hyperconnection import Qwen4ExpGatedResidualWeight
from .ple import Qwen4ExpPLEWeight


class Qwen4ExpQSAIndexerWeight(BaseWeightTpl):
    def __init__(self, *, prefix: str, network_config: dict, data_type) -> None:
        super().__init__(data_type=data_type)
        hidden_size = network_config["hidden_size"]
        index_n_heads = network_config["indexer_n_heads"]
        index_kv_heads = network_config["indexer_kv_heads"]
        index_head_dim = network_config["indexer_head_dim"]
        self.fused_weight_name = f"{prefix}.index_qk_proj.weight"
        self.q_weight_name = f"{prefix}.index_q_proj.weight"
        self.k_weight_name = f"{prefix}.index_k_proj.weight"
        self.q_proj = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[index_n_heads * index_head_dim],
            weight_names=self.q_weight_name,
            data_type=data_type,
        )
        self.k_proj = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[index_kv_heads * index_head_dim],
            weight_names=self.k_weight_name,
            data_type=data_type,
            tp_rank=0,
            tp_world_size=1,
        )
        # Keep the checkpoint's Gemma-style delta weights unchanged here.
        # The indexer applies ``1 + weight`` in FP32 at runtime so the scale is
        # not rounded to BF16 during model loading.
        self.q_norm = RMSNormWeight(
            index_head_dim, f"{prefix}.q_layernorm.weight", data_type
        )
        self.k_norm = RMSNormWeight(
            index_head_dim, f"{prefix}.k_layernorm.weight", data_type
        )
        self.q_rows = index_n_heads * index_head_dim

    def _create_weight(self):
        return

    def load_hf_weights(self, weights):
        if self.fused_weight_name in weights:
            fused = weights.pop(self.fused_weight_name)
            weights[self.q_weight_name], weights[self.k_weight_name] = torch.split(
                fused, [self.q_rows, fused.shape[0] - self.q_rows], dim=0
            )
        self.q_proj.load_hf_weights(weights)
        self.k_proj.load_hf_weights(weights)
        self.q_norm.load_hf_weights(weights)
        self.k_norm.load_hf_weights(weights)

    def verify_load(self) -> bool:
        return all(
            child.verify_load()
            for child in (self.q_proj, self.k_proj, self.q_norm, self.k_norm)
        )


class Qwen4ExpTransformerLayerWeight(Qwen35MOETransformerLayerWeight):
    def _init_norm(self):
        config = self.network_config_
        self.attn_hyper_connection = Qwen4ExpGatedResidualWeight(
            prefix=f"model.layers.{self.layer_num_}.attn_hyper_connection",
            hidden_size=config["hidden_size"],
            hc_count=config["hc_count"],
            hc_lowrank=config["hc_lowrank"],
            data_type=self.data_type_,
        )
        self.mlp_hyper_connection = Qwen4ExpGatedResidualWeight(
            prefix=f"model.layers.{self.layer_num_}.mlp_hyper_connection",
            hidden_size=config["hidden_size"],
            hc_count=config["hc_count"],
            hc_lowrank=config["hc_lowrank"],
            data_type=self.data_type_,
        )
        if not self.is_linear_attention_layer:
            self.qk_norm_weight_ = QKGEMMANormWeight(
                dim=self.head_dim,
                q_weight_name=self._q_norm_name,
                k_weight_name=self._k_norm_name,
                data_type=self.data_type_,
            )
            self.qsa_indexer = Qwen4ExpQSAIndexerWeight(
                prefix=f"model.layers.{self.layer_num_}.self_attn.indexer",
                network_config=config,
                data_type=self.data_type_,
            )
        else:
            self.qsa_indexer = None

        self.ple = None
        if self.layer_num_ + 1 in config.get("ple_layer_ids", []):
            self.ple = Qwen4ExpPLEWeight(
                prefix=f"model.layers.{self.layer_num_}.ple",
                network_config=config,
                data_type=self.data_type_,
            )

    def _init_gdn_weight(self):
        super()._init_gdn_weight()
        self.linear_norm.gate_activation = (
            self.network_config_.get("output_gate_type", "silu") or "silu"
        )
