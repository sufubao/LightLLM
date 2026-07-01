from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import (
    Qwen35TransformerLayerWeight,
)
from lightllm.models.qwen3next.layer_weights.qkv_gated_rowmm_weight import QKVGatedROWNMMWeight


MAIN_LAYER_PREFIX = "model.layers."
MTP_LAYER_PREFIX = "mtp.layers."

ATTN_NORM_NAME_ATTRS = (
    "_q_weight_name",
    "_q_norm_name",
    "_q_bias_name",
    "_k_weight_name",
    "_k_norm_name",
    "_k_bias_name",
    "_v_weight_name",
    "_v_bias_name",
    "_kv_weight_name",
    "_kv_bias_name",
    "_o_weight_name",
    "_o_bias_name",
    "_att_norm_weight_name",
    "_att_norm_bias_name",
    "_ffn_norm_weight_name",
    "_ffn_norm_bias_name",
)


def retarget_mtp_layer_name(name):
    if name is None:
        return None
    return name.replace(MAIN_LAYER_PREFIX, MTP_LAYER_PREFIX, 1)


def retarget_mtp_layer_attrs(layer_weight, attrs):
    for attr in attrs:
        setattr(layer_weight, attr, retarget_mtp_layer_name(getattr(layer_weight, attr)))


def init_mtp_qkv_weight(layer_weight):
    in_dim = layer_weight.n_embed
    layer_weight._o_gate_weight_name = f"{MTP_LAYER_PREFIX}{layer_weight.layer_num_}.self_attn.o_gate_proj.weight"
    qkv_quant = layer_weight.get_quant_method("qkv_proj")
    layer_weight.qkvo_gate_proj = QKVGatedROWNMMWeight(
        in_dim=in_dim,
        q_head_num=layer_weight.q_head_num_,
        kv_head_num=layer_weight.k_head_num_,
        head_dim=layer_weight.head_dim,
        weight_names=[
            layer_weight._q_weight_name,
            layer_weight._k_weight_name,
            layer_weight._v_weight_name,
            layer_weight._o_gate_weight_name,
        ],
        data_type=layer_weight.data_type_,
        bias_names=[layer_weight._q_bias_name, layer_weight._k_bias_name, layer_weight._v_bias_name, None],
        quant_method=qkv_quant,
    )


class Qwen3_5MTPTransformerLayerWeight(Qwen35TransformerLayerWeight):
    def _init_qkv(self):
        init_mtp_qkv_weight(self)

    def _init_weight_names(self):
        super()._init_weight_names()
        retarget_mtp_layer_attrs(self, ATTN_NORM_NAME_ATTRS)
        retarget_mtp_layer_attrs(
            self,
            (
                "_gate_weight_name",
                "_gate_bias_name",
                "_up_weight_name",
                "_up_bias_name",
                "_gate_up_weight_name",
                "_gate_up_bias_name",
                "_down_weight_name",
                "_down_bias_name",
            ),
        )
