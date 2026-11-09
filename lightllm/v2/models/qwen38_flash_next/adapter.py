import torch

from lightllm.v2.execution.compute.attention.gdn.backend import GDNStaticParams
from lightllm.v2.execution.compute.attention.gdn.spec import GatedDeltaNetSpec
from lightllm.v2.execution.compute.attention.sparse_mla.backend import SparseMLAStaticParams
from lightllm.v2.execution.compute.attention.sparse_mla.spec import SparseMLAAttentionSpec
from lightllm.v2.execution.contracts.state import LayerStates, StateSpec
from lightllm.v2.execution.formats.kv.spec import DenseKVStateSpec, SparseMLAKVStateSpec
from lightllm.v2.execution.formats.lexical import LexicalStateSpec
from lightllm.v2.execution.formats.recurrent.spec import ConvSSMStateSpec
from lightllm.v2.models.base.adapter import RecipeModelAdapter, grouped_moe_specs

from .configuration import is_linear_attention_layer, normalize_model_config
from .lexical_state import LexicalStateConfig, Qwen38LexicalState


class Qwen38FlashNextModelAdapter(RecipeModelAdapter):
    attention_phases = ("prefill", "decode")

    def __init__(self, config, *, role="main", dtype=torch.bfloat16, tp_world_size=1, request_capacity=1):
        self.config = normalize_model_config(config)
        self.dtype = dtype
        self.role = role
        self.tp_world_size = tp_world_size
        self.request_capacity = request_capacity
        self.layer_ids = tuple(range(self.config["num_hidden_layers"]))
        self._attention = {}
        states = []
        for layer in self.layer_ids:
            if is_linear_attention_layer(self.config, layer):
                spec = self._gdn_spec()
                self._attention[layer] = spec
                states.append(
                    StateSpec(
                        role,
                        f"gdn.{layer}",
                        layer,
                        (layer,),
                        ConvSSMStateSpec(
                            key_heads=self.config["linear_num_key_heads"],
                            value_heads=self.config["linear_num_value_heads"],
                            key_dim=spec.key_dim,
                            value_dim=spec.value_dim,
                            conv_history_length=spec.conv_size - 1,
                        ),
                        "recurrent",
                    )
                )
                continue
            spec = self._sparse_mla_spec()
            self._attention[layer] = spec
            states.append(
                StateSpec(
                    role,
                    f"qsa.{layer}",
                    layer,
                    (layer,),
                    SparseMLAKVStateSpec(spec.latent_dim, spec.rope_dim, spec.index_dim),
                    "history",
                )
            )
            states.append(
                StateSpec(
                    role,
                    f"qsa_positions.{layer}",
                    layer,
                    (layer,),
                    DenseKVStateSpec(kv_heads=1, head_dim=spec.rope_dim),
                    "positions",
                )
            )
        self._states = tuple(states)
        self._lexical = self._build_lexical_state()
        if self._lexical is not None:
            self._states += (self._lexical[0],)

    def _gdn_spec(self):
        config = self.config
        if config["linear_num_key_heads"] % self.tp_world_size or config["linear_num_value_heads"] % self.tp_world_size:
            raise ValueError("GDN projections require whole TP head shards")
        return GatedDeltaNetSpec(
            config["linear_num_key_heads"] // self.tp_world_size,
            config["linear_num_value_heads"] // self.tp_world_size,
            config["linear_key_head_dim"],
            config["linear_value_head_dim"],
            config["linear_conv_kernel_dim"],
            self.dtype,
            config.get("hidden_act", "silu"),
        )

    def _sparse_mla_spec(self):
        config = self.config
        if config["num_attention_heads"] % self.tp_world_size:
            raise ValueError("QSA query heads require whole TP shards")
        q_heads = config["num_attention_heads"] // self.tp_world_size
        rope_dim = int(config["head_dim"] * config.get("partial_rotary_factor", 1.0))
        latent_dim = config["head_dim"] - rope_dim
        return SparseMLAAttentionSpec(
            q_heads=q_heads,
            latent_dim=latent_dim,
            nope_dim=latent_dim,
            rope_dim=rope_dim,
            value_dim=latent_dim,
            dtype=self.dtype,
            index_heads=config["indexer_n_heads"],
            index_dim=config["indexer_head_dim"],
            index_topk=config["indexer_budget"],
            index_quant_block=128,
        )

    def _build_lexical_state(self):
        config = self.config
        layers = tuple(layer - 1 for layer in config.get("ple_layer_ids", ()))
        if not layers:
            return None
        if any(layer < 0 or not is_linear_attention_layer(config, layer) for layer in layers):
            raise ValueError("Qwen3.8 PLE layers must refer to linear-attention layers")
        params = LexicalStateSpec(
            ngram_context=config["ngram_size"] - 1,
            conv_context=(config["ple_conv_kernel_size"] - 1) * config["ngram_size"],
            width=config["hc_count"] * config["hidden_size"],
            eos_token_id=self._eos_token_id(),
            dtype=self.dtype,
        )
        return (
            StateSpec(self.role, "lexical", layers[0], layers, params, "bounded"),
            LexicalStateConfig(
                layers,
                params.ngram_context,
                params.conv_context,
                params.width,
                params.eos_token_id,
                self.dtype,
            ),
        )

    def _eos_token_id(self):
        eos = self.config["eos_token_id"]
        if isinstance(eos, list):
            if not eos:
                raise ValueError("Qwen3.8 lexical state requires an EOS token ID")
            eos = eos[0]
        return eos

    def attention_specs(self):
        return self._attention.copy()

    def state_specs(self):
        return self._states

    def attention_states(self):
        by_key = {state.key: state for state in self._states}
        result = {}
        for layer, spec in self._attention.items():
            key = f"gdn.{layer}" if isinstance(spec, GatedDeltaNetSpec) else f"qsa.{layer}"
            auxiliary = ()
            if isinstance(spec, SparseMLAAttentionSpec):
                auxiliary = (("positions", by_key[(self.role, f"qsa_positions.{layer}")]),)
            result[layer] = LayerStates(by_key[(self.role, key)], auxiliary)
        return result

    def moe_specs(self):
        config = dict(self.config)
        if "n_routed_experts" not in config and "num_experts" in config:
            config["n_routed_experts"] = config["num_experts"]
        config.setdefault("first_k_dense_replace", 0)
        config.setdefault("moe_layer_freq", 1)
        return grouped_moe_specs(config, self.dtype, layer_ids=self.layer_ids)

    def attention_params(self, weights, device):
        params = {}
        for layer, spec in self._attention.items():
            if isinstance(spec, SparseMLAAttentionSpec):
                params[layer] = SparseMLAStaticParams(spec, device.allocate_tensor)
                continue
            weight = weights[layer]
            linear = getattr(weight, "linear_attn", weight)
            conv = getattr(linear, "conv1d", linear)
            params[layer] = GDNStaticParams(
                spec, conv.weight, getattr(conv, "bias", None), linear.A_log, linear.dt_bias
            )
        return params

    def lexical_state(self, device="meta"):
        if self._lexical is None:
            return None
        return Qwen38LexicalState(self._lexical[1], self.request_capacity, device=device)

    def validate_layer(self, layer):
        spec = self._attention[layer.layer_num_]
        if isinstance(spec, GatedDeltaNetSpec):
            if not getattr(layer, "is_linear_attention_layer", False):
                raise ValueError("Qwen3.8 recipe does not implement the declared recurrent layer")
            return
        if getattr(layer, "is_linear_attention_layer", False):
            raise ValueError("Qwen3.8 recipe does not implement the declared indexed layer")
        if layer.tp_q_head_num_ != spec.q_heads or layer.head_dim_ != self.config["head_dim"]:
            raise ValueError("Qwen3.8 recipe dimensions disagree with the declared QSA semantics")

    def build_weights(self, quant_config):
        from .runtime.layer_weights.pre_and_post_layer_weight import Qwen4ExpPreAndPostLayerWeight
        from .runtime.layer_weights.transformer_layer_weight import Qwen4ExpTransformerLayerWeight

        self.pre_and_post_weight_class = Qwen4ExpPreAndPostLayerWeight
        self.transformer_weight_class = Qwen4ExpTransformerLayerWeight
        return super().build_weights(quant_config)

    def build_recipe(self):
        from .runtime.layer_infer.post_layer_infer import Qwen4ExpPostLayerInfer
        from .runtime.layer_infer.pre_layer_infer import Qwen4ExpPreLayerInfer
        from .runtime.layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer

        self.pre_layer_infer_class = Qwen4ExpPreLayerInfer
        self.post_layer_infer_class = Qwen4ExpPostLayerInfer
        self.transformer_layer_infer_class = Qwen4ExpTransformerLayerInfer
        return super().build_recipe()

    def load_weights(self, weights, directory, weight_dict=None):
        return super().load_weights(weights, directory, weight_dict)
