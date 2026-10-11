from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_dspark.model import Qwen3DSparkModel


class Qwen3_5DSparkModel(Qwen3DSparkModel):
    """Adapter for the current Qwen3 DSpark checkpoint with a Qwen3.5 target."""

    def _init_config(self):
        super()._init_config()
        self.config.update(self.config.get("dflash_config", {}))

        rope_parameters = self.config["rope_parameters"]
        if "rope_theta" in rope_parameters and "rope_theta" not in self.config:
            self.config["rope_theta"] = rope_parameters["rope_theta"]

        # The draft is Qwen3-style and owns a 1D rotary cache. Released
        # checkpoints rotate the full head, while target-shaped custom
        # checkpoints can retain Qwen3.5's partial rotary layout.
        self.config["rope_scaling"] = rope_parameters
        self.config["partial_rotary_factor"] = rope_parameters.get("partial_rotary_factor", 1.0)

    def _init_custom(self):
        # Draft and target use different rotary shapes, so the draft owns its rotary cache.
        LlamaTpPartModel._init_custom(self)
        self.block_size = self.config["block_size"]
        self.mask_token_id = self.config["mask_token_id"]

    def _init_mem_manager(self):
        target_mem_manager = self.main_model.mem_manager
        draft_kv_shape = (self.config["num_key_value_heads"], self.config["head_dim"])
        target_kv_shape = (
            target_mem_manager.linear_config.full_att_all_num_kv_heads,
            target_mem_manager.head_dim,
        )
        draft_heads, draft_dim = draft_kv_shape
        target_heads, target_dim = target_kv_shape
        assert draft_heads * draft_dim == target_heads * target_dim and draft_heads % target_heads == 0, (
            "Qwen3.5 parallel block drafter requires the same KV width per token, "
            f"got draft={draft_kv_shape}, target={target_kv_shape}."
        )
        self.mem_manager = target_mem_manager.create_kv_head_view(draft_dim)

    def _init_weights(self, start_layer_index=None):
        super()._init_weights(start_layer_index=start_layer_index)
        self.pre_post_weight.target_pre_post_weight = self.main_model.pre_post_weight
