"""Kimi K3/K3.1 model entry point.

The model class only normalizes the Hugging Face configuration and selects the
Kimi recipe. Attention state allocation and backend selection are delegated to
the core model adapter.
"""

import json
import os

from lightllm.common.build_utils import repair_config
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.kimi_linear.adapter import KimiLinearModelAdapter
from lightllm.models.kimi_linear.infer_struct import KimiLinearInferStateInfo
from lightllm.models.kimi_linear.layer_infer.transformer_layer_infer import KimiLinearTransformerLayerInfer
from lightllm.models.kimi_linear.layer_weights.transformer_layer_weight import KimiLinearTransformerLayerWeight
from lightllm.utils.envs_utils import get_env_start_args


class KimiLinearTpPartModel(Deepseek2TpPartModel):
    """Text decoder for the Kimi K3 family, including Kimi K3.1 Pro configs."""

    model_adapter_class = KimiLinearModelAdapter
    transformer_weight_class = KimiLinearTransformerLayerWeight
    transformer_layer_infer_class = KimiLinearTransformerLayerInfer
    infer_state_class = KimiLinearInferStateInfo

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            raw_config = json.load(json_file)

        model_type = raw_config.get("model_type")
        self.is_kimi_k31 = model_type == "kimi_k31"
        self.is_kimi_k3 = model_type in ("kimi_k3", "kimi_k31")
        self.weight_prefix = ""
        if isinstance(raw_config.get("text_config"), dict):
            raw_config = raw_config["text_config"].copy()
            self.weight_prefix = "language_model."
        else:
            raw_config = raw_config.copy()

        self.config = raw_config
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        self.config.setdefault("rope_scaling", None)
        # MLA stores a single latent KV stream; the inherited DeepSeek
        # validation only needs this compatibility value for TP checks.
        self.config.setdefault("num_key_value_heads", 1)
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size

        # DeepSeek's legacy layer classes use these aliases. Keep them in the
        # normalized model config so the recipe and compatibility path see the
        # same routing semantics.
        self.config["n_routed_experts"] = self.config.get("num_experts")
        self.config["n_shared_experts"] = self.config.get("num_shared_experts", 0)
        self.config["num_experts_per_tok"] = self.config.get("num_experts_per_token")
        self.config["norm_topk_prob"] = self.config.get("moe_renormalize", False)
        self.config["n_group"] = self.config.get("num_expert_group", 0)
        self.config["topk_group"] = self.config.get("topk_group", self.config.get("num_expert_group", 0))
        self.config["scoring_func"] = self.config.get("moe_router_activation_func", "softmax")
        self.config.setdefault("first_k_dense_replace", 0)
        self.config.setdefault("moe_layer_freq", 1)
        self.config.setdefault("q_lora_rank", None)
        self.config.setdefault("v_head_dim", self.config.get("qk_nope_head_dim"))
        # The adapter consumes this only while loading language-model weights;
        # it is not a model semantic field.
        self.config["_weight_prefix"] = self.weight_prefix

    def _verify_params(self):
        super()._verify_params()
        assert get_env_start_args().mtp_step == 0, "Kimi Linear does not provide MTP layers"

    def _init_model_adapter(self):
        super()._init_model_adapter()
        if self.model_adapter is not None:
            self.model_adapter.weight_prefix = self.weight_prefix

    def _init_custom(self):
        # DeepSeek's rotary setup and communication-group preparation are
        # reusable. Kimi does not add a second provider-specific group here.
        super()._init_custom()
