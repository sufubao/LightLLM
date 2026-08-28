# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from lightllm.common.basemodel.layer_weights.transformer_layer_weight import (
    TransformerLayerWeight,
)
from lightllm.common.basemodel.layer_weights.meta_weights import (
    COLMMWeight,
    LayerNormWeight,
    ParameterWeight,
    RMSNormWeight,
    ROWMMWeight,
    TpParameterWeight,
)
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import (
    Deepseek2TransformerLayerWeight,
)
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import (
    Deepseek3_2TransformerLayerWeight,
)
from .pre_and_post_layer_weight import add_language_model_aliases


class Glm5NextTransformerLayerWeight(Deepseek3_2TransformerLayerWeight):
    def _parse_config(self):
        super()._parse_config()
        # The released sparse MLA keeps kv_b_proj in BF16 even though the
        # surrounding projections are native FP8.  Its compressed-context
        # shortcut assumes a quantized kv_b matrix, so GLM uses the BMM split.
        self.enable_cc_method = False
        self.is_mtp_layer = self.layer_num_ >= self.network_config_["num_hidden_layers"]
        self.is_linear_attention_layer = (
            not self.is_mtp_layer
            and self.network_config_["layer_types"][self.layer_num_]
            == "linear_attention"
        )
        linear = self.network_config_["linear_attn_config"]
        self.linear_num_heads = linear["num_heads"]
        self.linear_head_dim = linear["head_dim"]
        self.linear_projection_size = self.linear_num_heads * self.linear_head_dim
        self.linear_conv_kernel_size = linear["short_conv_kernel_size"]
        self.mhc_streams = self.network_config_.get("hc_mult", 4)

    def _init_weight(self):
        if self.is_linear_attention_layer:
            self._init_kda()
        else:
            Deepseek2TransformerLayerWeight._init_qkvo(self)
            self._init_indexer_weight()

        if self.is_moe:
            self._init_moe()
        else:
            self._init_ffn()
        self._init_glm_norms()
        if not self.is_mtp_layer:
            self._init_mhc()

    def _init_kda(self):
        prefix = f"model.layers.{self.layer_num_}.self_attn"
        projection = self.linear_projection_size
        head_count = self.linear_num_heads
        head_dim = self.linear_head_dim

        self.linear_qkvb_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[projection, projection, projection, head_count],
            weight_names=[
                f"{prefix}.q_proj.weight",
                f"{prefix}.k_proj.weight",
                f"{prefix}.v_proj.weight",
                f"{prefix}.b_proj.weight",
            ],
            data_type=self.data_type_,
            quant_method=None,
        )
        # f_a and g_a are replicated across TP ranks.
        self.linear_fg_a_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[head_dim, head_dim],
            weight_names=[f"{prefix}.f_a_proj.weight", f"{prefix}.g_a_proj.weight"],
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.linear_fg_b_proj = ROWMMWeight(
            in_dim=head_dim,
            out_dims=[projection, projection],
            weight_names=[f"{prefix}.f_b_proj.weight", f"{prefix}.g_b_proj.weight"],
            data_type=self.data_type_,
            quant_method=None,
        )
        self.linear_qkv_conv1d = ROWMMWeight(
            in_dim=self.linear_conv_kernel_size,
            out_dims=[projection, projection, projection],
            weight_names=[
                f"{prefix}.q_conv1d.weight",
                f"{prefix}.k_conv1d.weight",
                f"{prefix}.v_conv1d.weight",
            ],
            data_type=self.data_type_,
            quant_method=None,
        )
        self.linear_A_log = TpParameterWeight(
            weight_name=f"{prefix}.A_log",
            data_type=torch.float32,
            weight_shape=(head_count,),
        )
        self.linear_dt_bias = TpParameterWeight(
            weight_name=f"{prefix}.dt_bias",
            data_type=torch.float32,
            weight_shape=(projection,),
        )
        self.linear_o_norm = RMSNormWeight(
            dim=head_dim,
            weight_name=f"{prefix}.o_norm.weight",
            data_type=self.data_type_,
        )
        self.linear_o_proj = COLMMWeight(
            in_dim=projection,
            out_dims=[self.n_embed],
            weight_names=f"{prefix}.o_proj.weight",
            data_type=self.data_type_,
            quant_method=None,
        )

    def _init_indexer_weight(self):
        """Initialize GLM's NoPE, K-pool indexer parameters.

        The head-weight projection intentionally accumulates in fp32.  Both
        reference engines do this because bf16 head weights can change close
        K-pool rankings on difficult long-context prompts.
        """

        prefix = f"model.layers.{self.layer_num_}.self_attn.indexer"
        self.wq_b_proj_ = ROWMMWeight(
            in_dim=self.q_lora_rank,
            out_dims=[self.index_n_heads * self.index_head_dim],
            weight_names=f"{prefix}.wq_b.weight",
            data_type=self.data_type_,
            quant_method=None,
        )
        self.wk_proj_ = ROWMMWeight(
            in_dim=self.hidden_size,
            out_dims=[self.index_head_dim],
            weight_names=f"{prefix}.wk.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.k_norm_ = LayerNormWeight(
            dim=self.index_head_dim,
            weight_name=f"{prefix}.k_norm.weight",
            data_type=self.data_type_,
            bias_name=f"{prefix}.k_norm.bias",
        )
        self.weights_proj_ = ROWMMWeight(
            in_dim=self.hidden_size,
            out_dims=[self.index_n_heads],
            weight_names=f"{prefix}.weights_proj.weight",
            data_type=torch.float32,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.index_kpool_compress_gate = ROWMMWeight(
            in_dim=self.hidden_size,
            out_dims=[self.index_head_dim],
            weight_names=f"{prefix}.index_kpool_compress_gate",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.index_kpool_compress_ape = ParameterWeight(
            weight_name=f"{prefix}.index_kpool_compress_ape",
            data_type=torch.float32,
            weight_shape=(self.network_config_["index_kpool"], self.index_head_dim),
        )

    def _init_glm_norms(self):
        prefix = f"model.layers.{self.layer_num_}"
        self.att_norm_weight_ = RMSNormWeight(
            dim=self.n_embed,
            weight_name=f"{prefix}.input_layernorm.weight",
            data_type=self.data_type_,
        )
        self.ffn_norm_weight_ = RMSNormWeight(
            dim=self.n_embed,
            weight_name=f"{prefix}.post_attention_layernorm.weight",
            data_type=self.data_type_,
        )
        if not self.is_linear_attention_layer:
            self.kv_a_layernorm_ = RMSNormWeight(
                dim=self.kv_lora_rank,
                weight_name=f"{prefix}.self_attn.kv_a_layernorm.weight",
                data_type=self.data_type_,
            )
            self.q_a_layernorm_ = RMSNormWeight(
                dim=self.q_lora_rank,
                weight_name=f"{prefix}.self_attn.q_a_layernorm.weight",
                data_type=self.data_type_,
            )

    def _init_mhc(self):
        prefix = f"model.layers.{self.layer_num_}"
        streams = self.mhc_streams
        mix_size = (2 + streams) * streams
        flattened_hidden = streams * self.n_embed
        self.hc_attn_fn = ParameterWeight(
            weight_name=f"{prefix}.hc_attn_fn",
            data_type=torch.float32,
            weight_shape=(mix_size, flattened_hidden),
        )
        self.hc_attn_base = ParameterWeight(
            weight_name=f"{prefix}.hc_attn_base",
            data_type=torch.float32,
            weight_shape=(mix_size,),
        )
        self.hc_attn_scale = ParameterWeight(
            weight_name=f"{prefix}.hc_attn_scale",
            data_type=torch.float32,
            weight_shape=(3,),
        )
        self.hc_ffn_fn = ParameterWeight(
            weight_name=f"{prefix}.hc_ffn_fn",
            data_type=torch.float32,
            weight_shape=(mix_size, flattened_hidden),
        )
        self.hc_ffn_base = ParameterWeight(
            weight_name=f"{prefix}.hc_ffn_base",
            data_type=torch.float32,
            weight_shape=(mix_size,),
        )
        self.hc_ffn_scale = ParameterWeight(
            weight_name=f"{prefix}.hc_ffn_scale",
            data_type=torch.float32,
            weight_shape=(3,),
        )

    def get_merged_kda_conv_weight(self):
        return self.linear_qkv_conv1d.mm_param.weight

    def project_kda_fg_b(self, f_a: torch.Tensor, g_a: torch.Tensor):
        method = self.linear_fg_b_proj.quant_method
        f = method.apply(f_a, self.linear_fg_b_proj.mm_param_list[0])
        g = method.apply(g_a, self.linear_fg_b_proj.mm_param_list[1])
        return f, g

    def _preprocess_kda_weights(self, weights):
        prefix = f"model.layers.{self.layer_num_}.self_attn"
        for projection in ("q", "k", "v"):
            name = f"{prefix}.{projection}_conv1d.weight"
            if name in weights and weights[name].ndim == 3:
                weights[name] = weights[name].squeeze(1)

    def load_hf_weights(self, weights):
        add_language_model_aliases(weights)

        # GLM checkpoints nest the shared expert under
        # ``mlp.shared_experts``.  This class deliberately bypasses
        # Deepseek2TransformerLayerWeight.load_hf_weights below, so perform
        # the fused-shared remap here before the generic loader consumes the
        # expert tensors.
        if self.num_fused_shared_experts > 0:
            self._rename_shared_experts(
                weights,
                self.experts.quant_method.weight_scale_suffix,
            )

        if self.is_linear_attention_layer:
            self._preprocess_kda_weights(weights)
            return TransformerLayerWeight.load_hf_weights(self, weights)

        kv_b_name = f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight"
        if kv_b_name in weights:
            k_b_proj, v_b_proj = self._split_kv_b_proj(weights[kv_b_name])
            weights[f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight"] = k_b_proj
            weights[f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight"] = v_b_proj

        return TransformerLayerWeight.load_hf_weights(self, weights)
