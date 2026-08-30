import torch

from typing import List

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.draft_registry import DraftModelRegistry
from lightllm.models.qwen4_exp.layer_infer import (
    Qwen4ExpTransformerLayerInfer,
)
from lightllm.models.qwen4_exp.model import Qwen4ExpTpPartModel
from lightllm.models.qwen4_exp_mtp.layer_infer import (
    Qwen4ExpMTPPostLayerInfer,
    Qwen4ExpMTPPreLayerInfer,
)
from lightllm.models.qwen4_exp_mtp.layer_weights import (
    Qwen4ExpMTPPreAndPostLayerWeight,
    Qwen4ExpMTPTransformerLayerWeight,
)


@DraftModelRegistry(
    model_type=("qwen4_exp", "qwen4_exp_text"),
    spec_modes=("vanilla_with_att", "eagle_with_att"),
)
class Qwen4ExpMTPModel(Qwen4ExpTpPartModel):
    """The recurrent, single-layer MTP decoder stored in a Qwen4 checkpoint."""

    pre_and_post_weight_class = Qwen4ExpMTPPreAndPostLayerWeight
    pre_layer_infer_class = Qwen4ExpMTPPreLayerInfer
    post_layer_infer_class = Qwen4ExpMTPPostLayerInfer
    transformer_weight_class = Qwen4ExpMTPTransformerLayerWeight
    transformer_layer_infer_class = Qwen4ExpTransformerLayerInfer
    is_mtp_draft_model = True

    def __init__(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop(
            "mtp_previous_draft_models"
        )
        super().__init__(kvargs)

    def _init_config(self):
        super()._init_config()
        # The embedded predictor owns one recurrent full-attention layer and
        # does not run PLE. Global runtime layer indices are assigned separately
        # in _init_infer_layer so each chained cache remains distinct.
        self.config["full_attention_interval"] = 1
        self.config["num_hidden_layers"] = 1
        self.config["n_layer"] = 1
        self.config["ple_layer_ids"] = []

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type,
            network_config=self.config,
            quant_cfg=self.quant_cfg,
        )
        # Qwen4 ships one MTP layer and reuses it recurrently. Vanilla mode may
        # instantiate several model objects, but every object loads layer 0.
        self.trans_layers_weight = [
            self.transformer_weight_class(
                0,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
        ]
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = (
            self.main_model.pre_post_weight.lm_head_weight_
        )

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager

    def _init_mem_manager(self):
        self.mem_manager = self.main_model.mem_manager

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        global_layer_index = len(self.main_model.layers_infer) + sum(
            len(model.layers_infer) for model in self.mtp_previous_draft_models
        )
        super()._init_infer_layer(start_layer_index=global_layer_index)

    def _init_some_value(self):
        super()._init_some_value()
        self.layers_num = 1

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached

    def _gen_special_model_input(self, token_num: int):
        # BaseModel's draft warmup assumes a single hidden stream. Qwen4's
        # scheme-A target/draft handoff keeps all HC streams before the final
        # mixer, so its synthetic warmup input must use hc_count * hidden_size.
        return {
            "mtp_draft_input_hiddens": torch.randn(
                token_num,
                self.config["hc_count"] * self.config["hidden_size"],
                dtype=self.data_type,
                device="cuda",
            )
        }
