from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    RMSNormWeight,
    ROWMMWeight,
)
from lightllm.common.quantization import Quantcfg
from lightllm.models.qwen4_exp.layer_weights.hyperconnection import (
    Qwen4ExpGatedResidualWeight,
)


class Qwen4ExpMTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    """Standalone view of the Qwen4 MTP weights embedded in the target checkpoint."""

    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        hc_count = network_config["hc_count"]

        # The checkpoint projections are ColumnParallelLinear(...,
        # gather_output=True) in the reference implementation. Replication is
        # equivalent and avoids two extra all-gathers for these small H x H
        # matrices.
        self.fc_embedding = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[hidden_size],
            weight_names="mtp.fc_embedding.weight",
            data_type=data_type,
            tp_rank=0,
            tp_world_size=1,
        )
        self.fc_hidden = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[hidden_size],
            weight_names="mtp.fc_hidden.weight",
            data_type=data_type,
            tp_rank=0,
            tp_world_size=1,
        )
        # Preserve the raw Gemma delta weights. grouped_gemma_rmsnorm applies
        # 1 + weight in FP32 at runtime, matching the Qwen4 reference path.
        self.pre_fc_norm_embedding = RMSNormWeight(
            dim=hidden_size,
            weight_name="mtp.pre_fc_norm_embedding.weight",
            data_type=data_type,
        )
        self.pre_fc_norm_hidden = RMSNormWeight(
            dim=hidden_size * hc_count,
            weight_name="mtp.pre_fc_norm_hidden.weight",
            data_type=data_type,
        )
        self.final_mixer = Qwen4ExpGatedResidualWeight(
            prefix="mtp.hyper_connection_mixer",
            hidden_size=hidden_size,
            hc_count=hc_count,
            hc_lowrank=network_config["hc_lowrank"],
            data_type=data_type,
            use_combine=False,
        )

        # Qwen4 sets mtp_use_dedicated_embeddings=false. The model injects the
        # target embedding and LM-head objects after constructing this layer.
        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
