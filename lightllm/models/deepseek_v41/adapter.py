from functools import partial
from types import SimpleNamespace
import torch
import torch.distributed as dist

from lightllm.core.contracts.state import StateSpec, LayerStates, FormatSpec
from lightllm.core.registry import ProviderRegistry, AttentionProvider, StateProvider
from lightllm.core.compute.attention.shared_compressed.spec import SharedCompressedAttentionSpec
from lightllm.core.compute.attention.shared_compressed.backend import (
    SharedCompressedBackend,
    SharedCompressedParams,
    rotary_frequencies,
    rms_norm,
)
from lightllm.core.compute.linear.backend import LinearBackend
from lightllm.core.compute.linear.mxfp import MXLinearBackend
from lightllm.core.compute.moe.weighted_activation import WeightedActivationMoeSpec, WeightedActivationMoeBackend
from .config import DeepseekV41Config
from .weights import WeightSet
from .recipe import DeepseekV41Layer


class DeepseekV41Adapter:
    attention_phases = ("prefill", "decode")

    def __init__(self, config, *, role, runtime, rank=0, world=1, request_capacity=1, max_seq_len=1024, group=None):
        self.config = config if isinstance(config, DeepseekV41Config) else DeepseekV41Config.from_dict(config)
        c = self.config
        c.validate_parallelism(world)
        self.role, self.runtime, self.rank, self.world = role, runtime, rank, world
        self.request_capacity, self.max_seq_len = request_capacity, max_seq_len
        self.group = group
        self.all_reduce = partial(dist.all_reduce, group=group) if world > 1 else None
        self._attention, self._states, self._resources = {}, [], {}
        windows, compressed, indices = {}, {}, {}
        for i in range(c.n_layers):
            windows[i] = StateSpec(role, f"window.{i}", i, (i,), (c.window_size, c.head_dim), "window")
            self._states.append(windows[i])
        for source in c.kv_source_layers:
            consumers = tuple(i for i in range(c.n_layers) if c.compress_ratios[i] and c.sources(i)[0] == source)
            compressed[source] = StateSpec(role, f"compressed.{source}", source, consumers, (c.head_dim,))
            indices[source] = StateSpec(role, f"index.{source}", source, consumers, (c.index_head_dim,))
            self._states.extend((compressed[source], indices[source]))
        for i, ratio in enumerate(c.compress_ratios):
            source, index = c.sources(i) if ratio else (None, None)
            self._attention[i] = SharedCompressedAttentionSpec(
                c.n_heads // world,
                c.head_dim,
                c.rope_head_dim,
                c.window_size,
                ratio,
                source,
                index,
                i,
                c.index_n_heads // world,
                c.index_n_heads,
                c.index_head_dim,
                c.index_topk,
                c.candidate_source_layer,
                c.candidate_topk_blocks,
                c.candidate_block_size,
                torch.bfloat16,
            )
            self._resources[i] = LayerStates(
                windows[i], (("compressed", compressed[source]), ("index", indices[source])) if ratio else ()
            )
        self.formats = {
            "history": FormatSpec(
                "fp4_roundtrip_bf16", torch.bfloat16, "token_mxfp_latent", ("latent",), 1, (rank, world)
            ),
            "window": FormatSpec(
                "fp8_roundtrip_bf16",
                torch.bfloat16,
                "request_mxfp_window",
                ("window", "partial", "gate"),
                1,
                (rank, world),
                field_dtypes=(torch.bfloat16, torch.float32, torch.float32),
            ),
        }
        self.hashes = None
        self.weights = None
        self.recipes = [DeepseekV41Layer(i, c, world, self.all_reduce) for i in range(c.n_layers)]
        self.frequencies = {}
        for ratio in set(c.compress_ratios):
            self.frequencies[ratio] = rotary_frequencies(
                c.rope_head_dim,
                max_seq_len,
                c.compress_rope_theta if ratio else c.rope_theta,
                c.original_seq_len if ratio else 0,
                c.rope_factor,
                c.beta_fast,
                c.beta_slow,
                runtime.device,
            )

    def attention_specs(self):
        return self._attention.copy()

    def state_specs(self):
        return tuple(self._states)

    def attention_states(self):
        return self._resources.copy()

    def moe_specs(self):
        c = self.config
        spec = WeightedActivationMoeSpec(
            c.dim,
            c.moe_inter_dim,
            c.n_routed_experts,
            c.n_activated_experts,
            self.rank,
            self.world,
            torch.bfloat16,
            c.gate_temp,
            c.norm_topk_prob,
            c.route_scale,
            c.swiglu_limit,
        )
        return {i: spec for i in range(c.n_layers)}

    def providers(self, base=None):
        from lightllm.core.storage.shared_compressed import CompressedHistoryStore, CompressedWindowStore

        base = base if base is not None else ProviderRegistry()
        storage = (
            StateProvider("token_mxfp_latent", "history", CompressedHistoryStore),
            StateProvider("request_mxfp_window", "window", CompressedWindowStore),
        )
        storage = tuple(
            provider
            for provider in storage
            if not any(
                item.layout == provider.layout
                and item.retention == provider.retention
                and (item.supports is None or item.supports(self.formats[provider.retention]))
                for item in base.storage
            )
        )
        return ProviderRegistry(
            base.attention + (AttentionProvider(SharedCompressedBackend, self.attention_phases),),
            base.storage + storage,
            parameters=((SharedCompressedAttentionSpec, self.static_params), *base.parameters.items()),
            static_params=base._static_params,
            linear=base.linear
            + (
                partial(MXLinearBackend, "fp8_block32_e8m0"),
                partial(MXLinearBackend, "fp4_group32_e8m0"),
                partial(LinearBackend, "torch.linear"),
            ),
            moe=base.moe + (WeightedActivationMoeBackend,),
            expert=base.expert,
            weight_finalizers=base.weight_finalizers,
            operations=base.operations.items(),
            transfers=base.transfers,
        )

    def build_weights(self, quant_config=None):
        self.weights = WeightSet(self.config, self.rank, self.world, self.runtime.device)
        for layer in self.weights.layers:
            layer.tp_rank = self.rank
        return self.weights.pre_post, self.weights.layers

    def build_recipe(self):
        return SimpleNamespace(), SimpleNamespace(), self.recipes

    def load_weights(self, weights, directory, weight_dict=None):
        if weights[0] is not self.weights.pre_post:
            raise ValueError("weights must belong to this adapter")
        self.weights.load(directory, weight_dict)
        if self.config.engram_layer_ids:
            from transformers import AutoTokenizer
            from .engram import EngramHashes

            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            self.hashes = EngramHashes(self.config, tokenizer, self.request_capacity, self.runtime.device)

    def static_params(self, spec, weight, *, device):
        compressor, indexer = None, None
        if spec.layer_id == spec.kv_source:
            compressor = SimpleNamespace(
                wkv=weight.attn_compressor_wkv,
                wgate=getattr(weight, "attn_compressor_wgate", None),
                norm=weight.attn_compressor_norm_weight.value,
            )
        if spec.layer_id == spec.index_source:
            indexer = SimpleNamespace(
                wq_b=weight.attn_indexer_wq_b,
                weights_proj=weight.attn_indexer_weights_proj,
                wk=getattr(weight, "attn_indexer_wk", None),
                norm=weight.attn_indexer_k_norm_weight.value if spec.layer_id == spec.kv_source else None,
            )
        return SharedCompressedParams(
            spec,
            self.frequencies[spec.ratio],
            weight.attn_attn_sink.value,
            self.config.norm_eps,
            compressor,
            indexer,
            self.all_reduce,
        )

    def forward(self, tokens, metadata, phase, *, all_logits=False):
        c, w = self.config, self.weights.pre_post
        part = c.vocab_size // self.world
        first = self.rank * part
        mask = (tokens < first) | (tokens >= first + part)
        hidden = torch.nn.functional.embedding((tokens - first).masked_fill(mask, 0), w.embed.value)
        hidden.masked_fill_(mask.unsqueeze(-1), 0)
        if self.all_reduce is not None:
            self.all_reduce(hidden)
        x = hidden.unsqueeze(1).repeat(1, c.hc_mult, 1)
        mix = torch.zeros(1, len(tokens), c.hc_mult, dtype=torch.float32, device=tokens.device)
        mix[..., 0] = 1
        hashes = self.hashes(tokens, metadata.requests) if self.hashes is not None else None
        for i, (recipe, weights) in enumerate(zip(self.recipes, self.weights.layers)):
            layer_hashes = hashes[:, c.engram_layer_ids.index(i)] if i in c.engram_layer_ids else None
            x, mix = recipe.forward(x, mix, metadata, weights, phase, layer_hashes)
        hidden = rms_norm(self.recipes[-1].collapse(x, mix), w.norm.value, c.norm_eps)
        if not all_logits:
            rows = [offset + count - 1 for _, _, count, offset in metadata.requests]
            hidden = hidden[rows]
        logits = torch.nn.functional.linear(hidden.float(), w.head.value)
        if self.world > 1:
            gathered = [torch.empty_like(logits) for _ in range(self.world)]
            dist.all_gather(gathered, logits, group=self.group)
            logits = torch.cat(gathered, -1)
        return logits
