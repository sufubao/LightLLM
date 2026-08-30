import json
import os

import torch

from lightllm.common.build_utils import repair_config
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.models.qwen3_5.infer_struct import Qwen35InferStateInfo
from lightllm.models.qwen3_5_moe.model import Qwen3_5MOETpPartModel
from lightllm.models.registry import ModelRegistry
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args

from .layer_infer import (
    Qwen4ExpPostLayerInfer,
    Qwen4ExpPreLayerInfer,
    Qwen4ExpTransformerLayerInfer,
)
from .layer_weights.pre_and_post_layer_weight import Qwen4ExpPreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import Qwen4ExpTransformerLayerWeight
from .mem_manager import Qwen4ExpMemManager
from .ple import build_layer_multipliers, build_ngram_vocab_layout


def normalize_qwen4_text_config(all_config: dict) -> tuple[dict, dict | None]:
    config = dict(all_config.get("text_config", all_config))
    vision_config = all_config.get("vision_config")
    repair_config(config, same_names=["num_attention_heads", "n_head"])
    repair_config(config, same_names=["hidden_size", "n_embd", "n_embed"])
    repair_config(config, same_names=["num_hidden_layers", "n_layer"])

    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        if "rope_theta" in rope_parameters:
            config.setdefault("rope_theta", rope_parameters["rope_theta"])
        if "partial_rotary_factor" in rope_parameters:
            config.setdefault(
                "partial_rotary_factor", rope_parameters["partial_rotary_factor"]
            )
        config.setdefault("rope_scaling", rope_parameters)

    config.setdefault("norm_topk_prob", True)
    config.setdefault("seed", 1234)
    config.setdefault("split_ngram_parts", 128)
    config.setdefault("hc_count", 4)
    config.setdefault("hc_lowrank", 320)
    config.setdefault("ple_layer_ids", [])
    config.setdefault("ple_embed_dim", config["hidden_size"])

    if config["ple_layer_ids"]:
        _, _, padded_vocab_size = build_ngram_vocab_layout(
            ngram_size=config.get("ngram_size", 3),
            heads_per_ngram=config.get("heads_per_ngram", 8),
            ngram_vocab_size_base=config.get("ngram_vocab_size_base", 20_000_000),
            ple_layer_index=0,
            make_divisible_by=config.get("make_ngram_vocab_size_divisible_by", 128),
        )
        config["ple_padded_vocab_size"] = padded_vocab_size
    return config, vision_config


@ModelRegistry("qwen4_exp", is_multimodal=True)
class Qwen4ExpTpPartModel(Qwen3_5MOETpPartModel):
    pre_and_post_weight_class = Qwen4ExpPreAndPostLayerWeight
    transformer_weight_class = Qwen4ExpTransformerLayerWeight
    pre_layer_infer_class = Qwen4ExpPreLayerInfer
    post_layer_infer_class = Qwen4ExpPostLayerInfer
    transformer_layer_infer_class = Qwen4ExpTransformerLayerInfer
    infer_state_class = Qwen35InferStateInfo

    def _init_config(self):
        config_path = os.path.join(self.weight_dir_, "config.json")
        with open(config_path, "r") as config_file:
            all_config = json.load(config_file)
        self.config, self.vision_config = normalize_qwen4_text_config(all_config)
        self.config["_qsa_runtime_enabled"] = self.max_seq_length > self.config.get(
            "indexer_budget", 2048
        )
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        self.num_kv_heads = max(
            self.config["num_key_value_heads"] // self.tp_world_size_, 1
        )

    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        start_args: StartArgs = get_env_start_args()
        ssm_dtype_dict = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        draft_full_att_kv_layer_num = get_added_mtp_kv_layer_num()
        self.linear_config = LinearAttCacheConfig(
            tp_world_size=self.tp_world_size_,
            full_att_all_num_kv_heads=self.config["num_key_value_heads"],
            full_att_dtype=self.data_type,
            full_att_num_kv_heads=self.num_kv_heads,
            full_att_head_dim=self.config["head_dim"],
            global_linear_k_heads=self.config["linear_num_key_heads"],
            global_linear_v_heads=self.config["linear_num_value_heads"],
            num_linear_k_heads=max(
                1, self.config["linear_num_key_heads"] // self.tp_world_size_
            ),
            num_linear_v_heads=max(
                1, self.config["linear_num_value_heads"] // self.tp_world_size_
            ),
            head_linear_k_dim=self.config["linear_key_head_dim"],
            head_linear_v_dim=self.config["linear_value_head_dim"],
            conv_kernel_size=self.config["linear_conv_kernel_dim"],
            linear_layer_num=self.config["n_layer"]
            - (self.config["n_layer"] // self.config["full_attention_interval"]),
            conv_state_dtype=self.data_type,
            ssm_state_dtype=ssm_dtype_dict[start_args.linear_att_ssm_data_type],
            full_attention_interval=self.config["full_attention_interval"],
            all_layer_num=self.config["n_layer"],
            draft_full_att_kv_layer_num=draft_full_att_kv_layer_num,
        )
        rotary_dim = int(
            self.config["head_dim"]
            * self.config.get("partial_rotary_factor", 1.0)
        )
        self.mem_manager = Qwen4ExpMemManager(
            size=self.max_total_token_num,
            dtype=self.data_type,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.config["head_dim"],
            full_att_layer_num=self.linear_config.get_full_att_kv_layer_num_with_draft_model(),
            linear_config=self.linear_config,
            mem_fraction=self.mem_fraction,
            qsa_enabled=self.config["_qsa_runtime_enabled"],
            qsa_head_dim=self.config["indexer_head_dim"],
            qsa_rotary_half_dim=rotary_dim // 2,
        )

    def _init_req_manager(self):
        super()._init_req_manager()
        config = self.config
        if not config.get("ple_layer_ids"):
            return
        context_len = config.get("ngram_size", 3) - 1
        eos_token_id = config["eos_token_id"]
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        req_slots = self.max_req_num + 1
        mtp_state_width = get_env_start_args().mtp_step + 1
        self.req_manager.req_to_ple_token_context = torch.full(
            (req_slots, mtp_state_width, context_len),
            eos_token_id,
            dtype=torch.long,
            device="cuda",
        )
        conv_state_len = (config.get("ple_conv_kernel_size", 4) - 1) * config.get(
            "ngram_size", 3
        )
        self.req_manager.req_to_ple_conv_state = torch.zeros(
            (
                req_slots,
                mtp_state_width,
                config["hc_count"] * config["hidden_size"],
                conv_state_len,
            ),
            dtype=self.data_type,
            device="cuda",
        )
        self.req_manager.req_to_ple_state_index = torch.zeros(
            req_slots, dtype=torch.int32, device="cuda"
        )
        sizes, offsets, _ = build_ngram_vocab_layout(
            ngram_size=config.get("ngram_size", 3),
            heads_per_ngram=config.get("heads_per_ngram", 8),
            ngram_vocab_size_base=config.get("ngram_vocab_size_base", 20_000_000),
            ple_layer_index=0,
            make_divisible_by=config.get("make_ngram_vocab_size_divisible_by", 128),
        )
        self.req_manager.ple_layer_multipliers = build_layer_multipliers(
            config["vocab_size"],
            config.get("ngram_size", 3),
            0,
            config.get("seed", 1234),
        ).cuda()
        self.req_manager.ple_head_vocab_sizes = sizes.cuda()
        self.req_manager.ple_head_offsets = offsets.cuda()

    def _init_cudagraph(self):
        # Decode PLE has a fixed shape for each captured batch and indexes its
        # persistent request state through graph input tensors, so it is safe to
        # use LightLLM's normal decode graph path.
        super()._init_cudagraph()
