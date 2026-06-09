from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, QKVROWNMMWeight


class MTPRetargetMixin:
    """Shared MTP weight-name retargeting (model.layers.* -> mtp.layers.*) and qkv/o_gate wiring,
    used by both the dense and MoE Qwen3.5 MTP layer-weight classes (#11). The dense subclass adds
    its dense-MLP retargets on top; the MoE subclass must not (it uses fused experts)."""

    _MAIN_PREFIX = "model.layers."
    _MTP_PREFIX = "mtp.layers."

    _ATTN_NORM_NAME_ATTRS = (
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

    def _retarget(self, name):
        if name is None:
            return None
        return name.replace(self._MAIN_PREFIX, self._MTP_PREFIX, 1)

    def _retarget_attn_norm_names(self):
        for attr in self._ATTN_NORM_NAME_ATTRS:
            setattr(self, attr, self._retarget(getattr(self, attr)))

    def _init_qkv(self):
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
