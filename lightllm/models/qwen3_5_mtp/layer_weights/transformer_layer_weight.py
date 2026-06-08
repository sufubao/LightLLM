from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import (
    Qwen35TransformerLayerWeight,
)
from lightllm.models.qwen3_5_mtp.layer_weights.mtp_retarget_mixin import MTPRetargetMixin
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen3_5MTPTransformerLayerWeight(MTPRetargetMixin, Qwen35TransformerLayerWeight):
    def _init_weight_names(self):
        super()._init_weight_names()
        # Retarget all main-model layer key names to the mtp.* namespace.
        self._retarget_attn_norm_names()
        # MLP (dense) projection names retargeted by Qwen35TransformerLayerWeight.
        self._gate_weight_name = self._retarget(self._gate_weight_name)
        self._gate_bias_name = self._retarget(self._gate_bias_name)
        self._up_weight_name = self._retarget(self._up_weight_name)
        self._up_bias_name = self._retarget(self._up_bias_name)
        self._gate_up_weight_name = self._retarget(self._gate_up_weight_name)
        self._gate_up_bias_name = self._retarget(self._gate_up_bias_name)
        self._down_weight_name = self._retarget(self._down_weight_name)
        self._down_bias_name = self._retarget(self._down_bias_name)
