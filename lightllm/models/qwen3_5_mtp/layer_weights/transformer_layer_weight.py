from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, QKVROWNMMWeight
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import (
    Qwen35TransformerLayerWeight,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen3_5MTPTransformerLayerWeight(Qwen35TransformerLayerWeight):
    """The single full-attention transformer layer of the Qwen3.5 MTP draft.

    Reuses the base Qwen3.5 full-attention layer weight (qkv/o with the attn output
    gate, q_norm/k_norm, gemma-style norms, dense gate/up/down MLP), but retargets
    every ``model.layers.{N}.*`` checkpoint key to the ``mtp.layers.{N}.*`` namespace.

    Real checkpoint keys (confirmed against /mtc/models/Qwen3.5-27B):
        mtp.layers.0.self_attn.{q,k,v,o}_proj.weight
        mtp.layers.0.self_attn.{q,k}_norm.weight
        mtp.layers.0.{input_layernorm, post_attention_layernorm}.weight
        mtp.layers.0.mlp.{gate,up,down}_proj.weight

    The draft layer is always full-attention (the model forces full_attention_interval=1),
    so the GDN / linear-attention weight paths are never taken and need no retargeting.
    """

    _MAIN_PREFIX = "model.layers."
    _MTP_PREFIX = "mtp.layers."

    def _retarget(self, name):
        if name is None:
            return None
        return name.replace(self._MAIN_PREFIX, self._MTP_PREFIX, 1)

    def _init_weight_names(self):
        super()._init_weight_names()
        # Retarget all main-model layer key names to the mtp.* namespace.
        self._q_weight_name = self._retarget(self._q_weight_name)
        self._q_norm_name = self._retarget(self._q_norm_name)
        self._q_bias_name = self._retarget(self._q_bias_name)
        self._k_weight_name = self._retarget(self._k_weight_name)
        self._k_norm_name = self._retarget(self._k_norm_name)
        self._k_bias_name = self._retarget(self._k_bias_name)
        self._v_weight_name = self._retarget(self._v_weight_name)
        self._v_bias_name = self._retarget(self._v_bias_name)
        self._kv_weight_name = self._retarget(self._kv_weight_name)
        self._kv_bias_name = self._retarget(self._kv_bias_name)
        self._o_weight_name = self._retarget(self._o_weight_name)
        self._o_bias_name = self._retarget(self._o_bias_name)
        self._att_norm_weight_name = self._retarget(self._att_norm_weight_name)
        self._att_norm_bias_name = self._retarget(self._att_norm_bias_name)
        self._ffn_norm_weight_name = self._retarget(self._ffn_norm_weight_name)
        self._ffn_norm_bias_name = self._retarget(self._ffn_norm_bias_name)
        # MLP (dense) projection names retargeted by Qwen35TransformerLayerWeight.
        self._gate_weight_name = self._retarget(self._gate_weight_name)
        self._gate_bias_name = self._retarget(self._gate_bias_name)
        self._up_weight_name = self._retarget(self._up_weight_name)
        self._up_bias_name = self._retarget(self._up_bias_name)
        self._gate_up_weight_name = self._retarget(self._gate_up_weight_name)
        self._gate_up_bias_name = self._retarget(self._gate_up_bias_name)
        self._down_weight_name = self._retarget(self._down_weight_name)
        self._down_bias_name = self._retarget(self._down_bias_name)

    def _init_qkv(self):
        # Mirror Qwen3Next._init_qkv but with the attn-output-gate key in the mtp.* namespace.
        # The base builds _o_gate_weight_name via an inline "model.layers." f-string, and
        # _split_q_with_gate (in load_hf_weights) derives the gate from q_proj and writes it
        # under this exact key -- so the ROWMMWeight must be constructed with the mtp.* name,
        # not retargeted afterwards (otherwise the constructed key and the written key differ).
        in_dim = self.n_embed
        q_out_dim = self.q_head_num_ * self.head_dim
        self.qkv_proj = QKVROWNMMWeight(
            in_dim=in_dim,
            q_head_num=self.q_head_num_,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[self._q_weight_name, self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._q_bias_name, self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("qkv_proj"),
        )
        self._o_gate_weight_name = f"{self._MTP_PREFIX}{self.layer_num_}.self_attn.o_gate_proj.weight"
        self._o_gate_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[q_out_dim],
            weight_names=[self._o_gate_weight_name],
            data_type=self.data_type_,
            bias_names=None,
            quant_method=self.get_quant_method("o_gate_proj"),
        )
