"""Model declaration for Kimi Linear and Kimi K3.1 Pro.

The adapter keeps the hybrid attention and latent-MoE semantics visible to the
core planner.  It does not allocate state or choose a CUDA kernel.
"""

from copy import deepcopy

from lightllm.core.compute.attention.kda.spec import KimiDeltaAttentionSpec
from lightllm.core.compute.attention.mla.spec import MLAAttentionSpec
from lightllm.core.contracts.state import StateSpec
from lightllm.models.base.adapter import RecipeModelAdapter
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer

from .layer_infer.transformer_layer_infer import KimiLinearTransformerLayerInfer
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import KimiLinearTransformerLayerWeight


class KimiLinearModelAdapter(RecipeModelAdapter):
    """Declare KDA/MLA layers and the latent MoE shape used by Kimi K3.1."""

    pre_and_post_weight_class = LlamaPreAndPostLayerWeight
    transformer_weight_class = KimiLinearTransformerLayerWeight
    pre_layer_infer_class = LlamaPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = KimiLinearTransformerLayerInfer

    def __init__(self, config, *, role, dtype, tp_world_size=1, layer_ids=None):
        self.config = deepcopy(config)
        self.dtype = dtype
        self.weight_prefix = self.config.pop("_weight_prefix", "")
        depth = config["num_hidden_layers"]
        linear = config["linear_attn_config"]
        kda_layers = {layer - 1 for layer in linear["kda_layers"]}
        full_layers = {layer - 1 for layer in linear["full_attn_layers"]}
        if kda_layers & full_layers or len(kda_layers | full_layers) != depth:
            raise ValueError("Kimi linear_attn_config must partition every transformer layer")
        if linear["num_heads"] % tp_world_size or config["num_attention_heads"] % tp_world_size:
            raise ValueError("Kimi attention requires complete TP head shards")

        self.layer_ids = tuple(range(depth)) if layer_ids is None else tuple(layer_ids)
        if len(self.layer_ids) != depth or set(self.layer_ids) != set(range(depth)):
            raise ValueError("Kimi execution layer IDs must cover every transformer layer exactly once")

        self._attention = {}
        states = []
        for layer in self.layer_ids:
            if layer in kda_layers:
                spec = KimiDeltaAttentionSpec(
                    linear["num_heads"] // tp_world_size,
                    linear["head_dim"],
                    linear["short_conv_kernel_size"],
                    dtype,
                    linear.get("gate_lower_bound", -5.0),
                )
                shape = (
                    linear["num_heads"],
                    linear["num_heads"],
                    spec.head_dim,
                    spec.head_dim,
                    spec.conv_size - 1,
                )
                states.append(StateSpec(role, f"kda.{layer}", layer, (layer,), shape, "recurrent"))
            else:
                spec = MLAAttentionSpec(
                    config["num_attention_heads"] // tp_world_size,
                    config["kv_lora_rank"],
                    config["qk_nope_head_dim"],
                    config["qk_rope_head_dim"],
                    config.get("v_head_dim", config["qk_nope_head_dim"]),
                    dtype,
                )
                states.append(
                    StateSpec(
                        role,
                        f"mla.{layer}",
                        layer,
                        (layer,),
                        (spec.latent_dim, spec.rope_dim),
                    )
                )
            self._attention[layer] = spec
        self._states = tuple(states)

    def load_weights(self, weights, directory, weight_dict=None):
        from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights

        if weight_dict is not None and not isinstance(weight_dict, dict):
            raise TypeError("weight_dict must be a dict or None")
        pre_post, layers = weights
        layers = list({id(layer): layer for layer in layers}.values())
        load_hf_weights(
            data_type=self.dtype,
            weight_dir=directory,
            pre_post_layer=pre_post,
            transformer_layer_list=layers,
            weight_dict=weight_dict,
            weight_prefix=self.weight_prefix,
        )
        pre_post.verify_load()
        for layer in layers:
            layer.verify_load()

    def moe_specs(self):
        from lightllm.models.base.adapter import grouped_moe_specs

        # The routed experts consume the latent projection.  The layer recipe
        # performs the model-width down/up projections around the provider.
        config = dict(self.config)
        config.setdefault("n_routed_experts", config.get("num_experts"))
        config.setdefault("num_experts_per_tok", config.get("num_experts_per_token"))
        config.setdefault("n_group", config.get("num_expert_group", 0))
        config.setdefault("topk_group", config.get("num_expert_group", 0))
        config.setdefault("norm_topk_prob", config.get("moe_renormalize", False))
        config.setdefault("scoring_func", config.get("moe_router_activation_func", "softmax"))
        config.setdefault("first_k_dense_replace", 0)
        config.setdefault("moe_layer_freq", 1)
        config["hidden_size"] = config.get("routed_expert_hidden_size", config["hidden_size"])
        return grouped_moe_specs(
            config,
            self.dtype,
            layer_ids=self.layer_ids,
            group_score_topk=2,
            activation=config.get("hidden_act", "silu"),
            activation_situ_beta=config.get("activation_situ_beta"),
            activation_situ_linear_beta=config.get("activation_situ_linear_beta"),
        )

    def validate_layer(self, layer):
        spec = self._attention[layer.layer_num_]
        if layer.is_linear_attention_layer != isinstance(spec, KimiDeltaAttentionSpec):
            raise ValueError("Kimi recipe differs from the declared attention family")
        if isinstance(spec, KimiDeltaAttentionSpec):
            actual = (layer.kda_local_num_heads, layer.kda_head_dim)
            expected = (spec.heads, spec.head_dim)
        else:
            actual = (layer.tp_q_head_num_, layer.kv_lora_rank, layer.qk_nope_head_dim, layer.qk_rope_head_dim)
            expected = (spec.q_heads, spec.latent_dim, spec.nope_dim, spec.rope_dim)
        if actual != expected:
            raise ValueError(f"Kimi recipe dimensions differ from semantics: {actual} != {expected}")
