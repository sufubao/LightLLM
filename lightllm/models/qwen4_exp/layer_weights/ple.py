import threading

from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    ParameterWeight,
    RMSNormWeight,
    ROWMMWeight,
)
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import (
    BaseWeightTpl,
)

from lightllm.models.qwen4_exp.ple import compute_shard_overlap


class Qwen4ExpNGramEmbeddingWeight(EmbeddingWeight):
    """TP-sharded PLE table loaded from the checkpoint's 128 file shards."""

    def __init__(
        self,
        *,
        dim: int,
        vocab_size: int,
        weight_prefix: str,
        split_parts: int,
        data_type,
    ) -> None:
        self.weight_prefix = weight_prefix
        self.split_parts = split_parts
        self._loaded_shards = set()
        self._copied_rows = 0
        self._load_lock = threading.Lock()
        super().__init__(
            dim=dim,
            vocab_size=vocab_size,
            weight_name=f"{weight_prefix}.weight",
            data_type=data_type,
        )

    def load_hf_weights(self, weights):
        checkpoint_rows = (self.vocab_size + self.split_parts - 1) // self.split_parts
        for shard_index in range(self.split_parts):
            name = f"{self.weight_prefix}.shard_{shard_index}.weight"
            if name not in weights:
                continue
            loaded_weight = weights[name]
            checkpoint_start = shard_index * checkpoint_rows
            overlap = compute_shard_overlap(
                checkpoint_start=checkpoint_start,
                checkpoint_rows=loaded_weight.shape[0],
                tp_start=self.tp_vocab_start_id,
                tp_end=self.tp_vocab_end_id,
            )
            with self._load_lock:
                if shard_index in self._loaded_shards:
                    continue
                self._loaded_shards.add(shard_index)
                if overlap is None:
                    continue
                source_start, destination_start, row_count = overlap
                source = loaded_weight.narrow(0, source_start, row_count)
                target = self.weight.narrow(0, destination_start, row_count)
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                self._copied_rows += row_count
                if self._copied_rows == self.weight.shape[0]:
                    self.weight.load_ok = True


class Qwen4ExpPLEWeight(BaseWeightTpl):
    """All weights owned by one Qwen4 per-layer embedding injection."""

    def __init__(self, *, prefix: str, network_config: dict, data_type) -> None:
        super().__init__(data_type=data_type)
        hidden_size = network_config["hidden_size"]
        hc_count = network_config["hc_count"]
        hc_hidden_size = hidden_size * hc_count
        ple_embed_dim = network_config.get("ple_embed_dim", hidden_size)
        ngram_heads = (network_config.get("ngram_size", 3) - 1) * network_config.get(
            "heads_per_ngram", 8
        )
        if ple_embed_dim % ngram_heads != 0:
            raise ValueError(
                f"ple_embed_dim={ple_embed_dim} is not divisible by ngram_heads={ngram_heads}"
            )

        self.ngram_embedding = Qwen4ExpNGramEmbeddingWeight(
            dim=ple_embed_dim // ngram_heads,
            vocab_size=network_config["ple_padded_vocab_size"],
            weight_prefix=f"{prefix}.ple_embedding.ngram_embedding",
            split_parts=network_config.get("split_ngram_parts", 128),
            data_type=data_type,
        )
        self.key_proj = ROWMMWeight(
            in_dim=ple_embed_dim,
            out_dims=[hc_hidden_size],
            weight_names=f"{prefix}.key_proj.weight",
            data_type=data_type,
            tp_rank=0,
            tp_world_size=1,
        )
        self.value_proj = ROWMMWeight(
            in_dim=ple_embed_dim,
            out_dims=[hidden_size],
            weight_names=f"{prefix}.value_proj.weight",
            data_type=data_type,
            tp_rank=0,
            tp_world_size=1,
        )
        self.norm_key = RMSNormWeight(
            hc_hidden_size, f"{prefix}.norm_key.weight", data_type
        )
        self.norm_query = RMSNormWeight(
            hc_hidden_size, f"{prefix}.norm_query.weight", data_type
        )
        self.norm_conv = RMSNormWeight(
            hc_hidden_size, f"{prefix}.norm_conv.weight", data_type
        )
        conv_kernel_size = network_config.get("ple_conv_kernel_size", 4)
        self.conv1d = ParameterWeight(
            weight_name=f"{prefix}.conv1d.weight",
            data_type=data_type,
            weight_shape=(hc_hidden_size, 1, conv_kernel_size),
        )

    def _create_weight(self):
        return

    def load_hf_weights(self, weights):
        for child in (
            self.ngram_embedding,
            self.key_proj,
            self.value_proj,
            self.norm_key,
            self.norm_query,
            self.norm_conv,
            self.conv1d,
        ):
            child.load_hf_weights(weights)

    def verify_load(self) -> bool:
        return all(
            child.verify_load()
            for child in (
                self.ngram_embedding,
                self.key_proj,
                self.value_proj,
                self.norm_key,
                self.norm_query,
                self.norm_conv,
                self.conv1d,
            )
        )
