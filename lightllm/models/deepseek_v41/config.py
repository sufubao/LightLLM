from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DeepseekV41Config:
    vocab_size: int
    dim: int
    moe_inter_dim: int
    n_layers: int
    n_heads: int
    n_routed_experts: int
    n_activated_experts: int
    q_lora_rank: int
    head_dim: int
    rope_head_dim: int
    o_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratios: tuple[int, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    n_shared_experts: int = 1
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    norm_eps: float = 1e-20
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_factor: float = 16.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 0
    dtype: str = "fp8"
    expert_dtype: str | None = "fp4"

    @classmethod
    def from_dict(cls, config):
        text = config.get("text_config", config)
        aliases = dict(
            dim="hidden_size",
            moe_inter_dim="moe_intermediate_size",
            n_layers="num_hidden_layers",
            n_heads="num_attention_heads",
            n_activated_experts="num_experts_per_tok",
            rope_head_dim="qk_rope_head_dim",
            window_size="sliding_window",
            norm_eps="rms_norm_eps",
            kv_source_layers="kv_source_layer_ids",
            index_source_layers="index_source_layer_ids",
            candidate_source_layer="candidate_source_layer_id",
            route_scale="routed_scaling_factor",
            score_func="scoring_func",
            engram_pad_id="engram_pad_token_id",
        )
        values = {}
        for field in fields(cls):
            key = aliases.get(field.name, field.name)
            if key in text:
                value = text[key]
                values[field.name] = tuple(value) if isinstance(value, list) else value
        values["compress_ratios"] = values["compress_ratios"][: values["n_layers"]]
        rope = text.get("rope_scaling") or {}
        for target, source in (
            ("rope_factor", "factor"),
            ("original_seq_len", "original_max_position_embeddings"),
            ("beta_fast", "beta_fast"),
            ("beta_slow", "beta_slow"),
        ):
            if source in rope:
                values[target] = rope[source]
        quant = config.get("quantization_config", {})
        if quant:
            if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [32, 32]:
                raise ValueError("DeepSeek-V4.1 requires FP8 block-32 checkpoint weights")
            if quant.get("scale_fmt") != "ue8m0" or quant.get("expert_dtype") != "fp4":
                raise ValueError("DeepSeek-V4.1 requires UE8M0 scales and FP4 experts")
            values.update(dtype="fp8", expert_dtype="fp4")
        else:
            values.update(dtype="bf16", expert_dtype=None)
        return cls(**values)

    def __post_init__(self):
        if self.score_func != "sqrtsoftplus" or self.gate_temp <= 0:
            raise ValueError("DeepSeek-V4.1 requires sqrtsoftplus routing and positive gate temperature")
        if self.dtype not in ("bf16", "fp8") or self.expert_dtype not in (None, "fp4"):
            raise ValueError("unsupported DeepSeek-V4.1 checkpoint encoding")
        if len(self.compress_ratios) != self.n_layers or any(r not in (0, 1, 2) for r in self.compress_ratios):
            raise ValueError("DeepSeek-V4.1 requires one ratio (0, 1, 2) per backbone layer")
        for sources in (self.kv_source_layers, self.index_source_layers, self.engram_layer_ids):
            if tuple(sorted(set(sources))) != sources or any(i < 0 or i >= self.n_layers for i in sources):
                raise ValueError("source layer identities must be unique, ordered backbone layers")
        if not set(self.kv_source_layers) <= set(self.index_source_layers):
            raise ValueError("each compressed KV source must produce index keys")
        for layer, ratio in enumerate(self.compress_ratios):
            if not ratio:
                if layer in self.kv_source_layers or layer in self.index_source_layers:
                    raise ValueError("a pure window layer cannot produce compressed state")
                continue
            kv, index = self.sources(layer)
            if kv is None or index is None or self.compress_ratios[kv] != ratio or self.compress_ratios[index] != ratio:
                raise ValueError(f"layer {layer} has no compatible shared KV/index source")
        if self.candidate_source_layer >= 0 and (
            self.candidate_source_layer not in self.index_source_layers
            or min(self.candidate_topk_blocks, self.candidate_block_size) <= 0
        ):
            raise ValueError("candidate filtering requires an index source and positive block geometry")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("each Engram layer needs its table capacity")
        if (
            self.engram_layer_ids
            and min(
                self.engram_vocab_size,
                self.engram_n_heads,
                self.engram_head_dim,
                self.engram_compressed_vocab_size,
                self.engram_max_ngram_size - 1,
            )
            <= 0
        ):
            raise ValueError("Engram hash and embedding geometry must be positive")
        if self.n_shared_experts != 1 or not 0 < self.n_activated_experts <= self.n_routed_experts:
            raise ValueError("DeepSeek-V4.1 requires one shared expert and a valid routed top-k")
        if self.n_heads % self.o_groups or not 0 < self.rope_head_dim <= min(self.head_dim, self.index_head_dim):
            raise ValueError("invalid grouped projection or rotary dimensions")

    def sources(self, layer):
        return tuple(
            next((i for i in reversed(sources) if i <= layer), None)
            for sources in (self.kv_source_layers, self.index_source_layers)
        )

    def validate_parallelism(self, world_size):
        if world_size <= 0 or any(
            value % world_size
            for value in (self.n_heads, self.o_groups, self.index_n_heads, self.n_routed_experts, self.vocab_size)
        ):
            raise ValueError("TP size must divide attention heads, output groups, index heads, experts and vocabulary")
