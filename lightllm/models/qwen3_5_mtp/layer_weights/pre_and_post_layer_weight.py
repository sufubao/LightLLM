from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    NoTpGEMMANormWeight,
    ROWMMWeight,
)
from lightllm.common.quantization import Quantcfg


class Qwen3_5MTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    """Pre/post weights for the Qwen3.5 MTP draft block.

    All weights live under the dedicated ``mtp.*`` namespace of the checkpoint:
        - ``mtp.fc.weight``                  -> eh_proj fusion (in=hidden*2, out=hidden)
        - ``mtp.pre_fc_norm_embedding.weight`` -> enorm (applied to the token embedding)
        - ``mtp.pre_fc_norm_hidden.weight``    -> hnorm (applied to the draft input hidden)
        - ``mtp.norm.weight``                -> shared-head final norm

    ``wte_weight_`` / ``lm_head_weight_`` stay ``None`` here; the model injects the
    main model's shared embedding / lm_head (config ``mtp_use_dedicated_embeddings: false``).

    NOTE: Qwen3.5 uses the Gemma-style ``(1 + weight)`` RMSNorm (NoTpGEMMANormWeight)
    for all of its norms, so the MTP norms must use the same class -- NOT the plain
    RMSNormWeight that the deepseek_mtp template uses.
    """

    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        self.quant_cfg: Quantcfg = quant_cfg
        hidden_size = network_config["hidden_size"]

        self.eh_proj_weight_ = ROWMMWeight(
            in_dim=hidden_size * 2,
            out_dims=[hidden_size],
            weight_names="mtp.fc.weight",
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(0, "eh_proj"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.enorm_weight_ = NoTpGEMMANormWeight(
            dim=hidden_size,
            weight_name="mtp.pre_fc_norm_embedding.weight",
            data_type=self.data_type_,
        )
        self.hnorm_weight_ = NoTpGEMMANormWeight(
            dim=hidden_size,
            weight_name="mtp.pre_fc_norm_hidden.weight",
            data_type=self.data_type_,
        )
        self.final_norm_weight_ = NoTpGEMMANormWeight(
            dim=hidden_size,
            weight_name="mtp.norm.weight",
            data_type=self.data_type_,
        )

        # Shared with the main Qwen3.5 model, injected by the model class (not loaded here).
        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
        return
