from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json

import torch

from lightllm.core.compute.linear.spec import LinearSpec
from lightllm.core.contracts.state import FormatSpec
from lightllm.core.formats.weights.view import WeightView


@dataclass
class WeightEntry:
    shape: tuple
    dtype: object
    axis: int | None = None
    padded: bool = False
    value: torch.Tensor | None = None


class NativeLinear:
    def __init__(self, weight, scale, dtype):
        self.weight, self.scale, self.dtype = weight, scale, dtype
        self.mm = None

    def native_linear_view(self):
        weight, scale = self.weight.value, None if self.scale is None else self.scale.value
        encoding = (
            "plain"
            if scale is None
            else ("fp4_group32_e8m0" if weight.dtype == torch.float4_e2m1fn_x2 else "fp8_block32_e8m0")
        )
        fmt = FormatSpec(
            encoding,
            weight.dtype,
            "output_input",
            ("weight",) if scale is None else ("weight", "scale"),
            1,
            group_size=32 if scale is not None else None,
            scale_dtype=None if scale is None else scale.dtype,
        )
        spec = LinearSpec(weight.shape[1] * (2 if encoding.startswith("fp4") else 1), weight.shape[0], self.dtype)
        return spec, WeightView(fmt, dict(weight=weight, scale=scale, bias=None)), "mm"

    def __call__(self, x):
        shape = (*x.shape[:-1], self.weight.value.shape[0])
        return self.mm(x.reshape(-1, x.shape[-1])).view(shape)


class NativeExperts:
    enable_ep_moe = False

    def __init__(self, c, rank, world, device, entries, prefix):
        n, mid, dim = c.n_routed_experts // world, c.moe_inter_dim, c.dim
        quantized = c.expert_dtype == "fp4"
        dtype, packing = (torch.float4_e2m1fn_x2, 2) if quantized else (torch.bfloat16, 1)
        self.gate_up = torch.empty(n, 2 * mid, dim // packing, dtype=dtype, device=device)
        self.down = torch.empty(n, dim, mid // packing, dtype=dtype, device=device)
        self.gate_up_scale = (
            torch.empty(n, 2 * mid, dim // 32, dtype=torch.float8_e8m0fnu, device=device) if quantized else None
        )
        self.down_scale = (
            torch.empty(n, dim, mid // 32, dtype=torch.float8_e8m0fnu, device=device) if quantized else None
        )
        for i in range(n):
            for name, target, scales in (
                ("w1", self.gate_up[i, :mid], None if not quantized else self.gate_up_scale[i, :mid]),
                ("w3", self.gate_up[i, mid:], None if not quantized else self.gate_up_scale[i, mid:]),
                ("w2", self.down[i], None if not quantized else self.down_scale[i]),
            ):
                key = f"{prefix}.experts.{rank * n + i}.{name}"
                entries[key + ".weight"] = WeightEntry(tuple(target.shape), dtype, value=target)
                if scales is not None:
                    entries[key + ".scale"] = WeightEntry(tuple(scales.shape), scales.dtype, value=scales)
        self.bias = entries[prefix + ".gate.bias"]
        self.format = FormatSpec(
            "fp4_group32_e8m0" if quantized else "plain",
            dtype,
            "local_expert_output_input",
            ("gate_up", "down", "gate_up_scale", "down_scale") if quantized else ("gate_up", "down"),
            1,
            (rank, world),
            group_size=32 if quantized else None,
            scale_dtype=torch.float8_e8m0fnu if quantized else None,
        )

    def native_moe_view(self):
        return WeightView(
            self.format,
            dict(
                gate_up=self.gate_up,
                down=self.down,
                gate_up_scale=self.gate_up_scale,
                down_scale=self.down_scale,
                correction_bias=self.bias.value,
            ),
        )


class WeightSet:
    def __init__(self, config, rank, world, device):
        self.config, self.rank, self.world, self.device = config, rank, world, device
        self.entries = {}
        self.pre_post, self.layers = SimpleNamespace(), []
        c = config
        self.pre_post.embed = self.add("embed.weight", (c.vocab_size, c.dim), torch.bfloat16, axis=0)
        self.pre_post.head = self.add("head.weight", (c.vocab_size, c.dim), torch.float32, axis=0)
        self.pre_post.norm = self.add("norm.weight", (c.dim,), torch.bfloat16)
        for i in range(c.n_layers):
            layer = SimpleNamespace(layer_num_=i)
            prefix = f"layers.{i}."

            def add(name, shape, dtype=torch.bfloat16, axis=None, padded=False):
                value = self.add(prefix + name, shape, dtype, axis, padded)
                setattr(layer, name.replace(".", "_"), value)
                return value

            def linear(name, k, n, axis=None, dtype=None):
                dtype = dtype or (torch.float8_e4m3fn if c.dtype == "fp8" else torch.bfloat16)
                weight = add(name + ".weight", (n, k), dtype, axis)
                scale = (
                    add(name + ".scale", (n // 32, k // 32), torch.float8_e8m0fnu, axis)
                    if dtype == torch.float8_e4m3fn
                    else None
                )
                value = NativeLinear(weight, scale, torch.bfloat16 if scale is not None else dtype)
                setattr(layer, name.replace(".", "_"), value)
                return value

            h = c.hc_mult
            for sub in ("attn", "ffn"):
                add(f"hc_{sub}_fn", ((2 + h) * h, h * c.dim), torch.float32)
                add(f"hc_{sub}_base", ((2 + h) * h,), torch.float32)
                add(f"hc_{sub}_scale", (3,), torch.float32)
                add(f"{sub}_norm.weight", (c.dim,))
            add("attn.attn_sink", (c.n_heads,), torch.float32, 0)
            linear("attn.wq_a", c.dim, c.q_lora_rank)
            linear("attn.wq_b", c.q_lora_rank, c.n_heads * c.head_dim, 0)
            linear("attn.wkv", c.dim, c.head_dim)
            linear("attn.wo_b", c.o_groups * c.o_lora_rank, c.dim, 1)

            linear("attn.wo_a", c.n_heads * c.head_dim // c.o_groups, c.o_groups * c.o_lora_rank, 0)
            add("attn.q_norm.weight", (c.q_lora_rank,))
            add("attn.kv_norm.weight", (c.head_dim,))
            if i in c.kv_source_layers:
                ratio = c.compress_ratios[i]
                linear("attn.compressor.wkv", c.dim, c.head_dim, dtype=torch.float32 if ratio > 1 else torch.bfloat16)
                if ratio > 1:
                    linear("attn.compressor.wgate", c.dim, c.head_dim, dtype=torch.float32)
                add("attn.compressor.norm.weight", (c.head_dim,))
                linear("attn.indexer.wk", c.head_dim, c.index_head_dim, dtype=torch.bfloat16)
                add("attn.indexer.k_norm.weight", (c.index_head_dim,))
            if i in c.index_source_layers:
                linear("attn.indexer.wq_b", c.q_lora_rank, c.index_n_heads * c.index_head_dim, 0)
                linear("attn.indexer.weights_proj", c.dim, c.index_n_heads, 0, torch.bfloat16)
            add("ffn.gate.weight", (c.n_routed_experts, c.dim))
            add("ffn.gate.bias", (c.n_routed_experts,), torch.float32)
            for name, k, n in (
                ("w1", c.dim, c.moe_inter_dim),
                ("w3", c.dim, c.moe_inter_dim),
                ("w2", c.moe_inter_dim, c.dim),
            ):
                linear("ffn.shared_experts." + name, k, n)
            layer.experts = NativeExperts(c, rank, world, device, self.entries, prefix + "ffn")
            if i in c.engram_layer_ids:
                rows = c.engram_num_embeddings[c.engram_layer_ids.index(i)]
                add("engram.embed.weight", (rows, c.engram_head_dim), torch.float8_e4m3fn, 0, True)
                add("engram.embed.scale", (rows, c.engram_head_dim // 32), torch.float8_e8m0fnu, 0, True)
                add("engram.q_weight", (h, c.dim))
                add("engram.k_weight", (h, c.dim))
                linear(
                    "engram.wkv", (c.engram_max_ngram_size - 1) * c.engram_n_heads * c.engram_head_dim, c.dim * (h + 1)
                )
            self.layers.append(layer)

    def add(self, name, shape, dtype, axis=None, padded=False):
        value = WeightEntry(shape, dtype, axis, padded)
        self.entries[name] = value
        return value

    def load(self, directory, weight_dict=None):
        from safetensors import safe_open
        from collections import defaultdict

        remaining = set(self.entries)

        def accept(name, tensor):
            entry = self.entries[name]
            shape = list(entry.shape)
            actual = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else tuple(tensor.get_shape())
            if actual != entry.shape:
                raise ValueError(f"checkpoint shape mismatch: {name}: {actual}, expected {entry.shape}")
            if entry.axis is not None:
                axis = entry.axis
                part = (shape[axis] + self.world - 1) // self.world if entry.padded else shape[axis] // self.world
                if not entry.padded and shape[axis] % self.world:
                    raise ValueError(f"unaligned TP slice: {name}")
                left, right = self.rank * part, min((self.rank + 1) * part, shape[axis])
                slices = [slice(None)] * len(shape)
                slices[axis] = slice(left, right)
                tensor = tensor[tuple(slices)]
                shape[axis] = part
            elif not isinstance(tensor, torch.Tensor):
                tensor = tensor[:]
            if entry.value is None:
                entry.value = torch.empty(shape, dtype=entry.dtype, device=self.device)
            target = entry.value
            if tensor.shape != target.shape:
                if not entry.padded or tensor.shape[0] > target.shape[0] or tensor.shape[1:] != target.shape[1:]:
                    raise ValueError(f"checkpoint shape mismatch: {name}: {tensor.shape}, expected {target.shape}")
                target.fill_(1 if entry.dtype == torch.float8_e8m0fnu else 0)
                target = target[: tensor.shape[0]]
            if entry.dtype == torch.float4_e2m1fn_x2 and tensor.dtype == torch.int8:
                tensor = tensor.view(torch.float4_e2m1fn_x2)
            if (
                entry.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2)
                and tensor.dtype != entry.dtype
            ):
                raise ValueError(f"checkpoint encoding mismatch: {name}: {tensor.dtype}")
            target.copy_(tensor)
            remaining.remove(name)

        if weight_dict is not None:
            for name in tuple(remaining):
                if name in weight_dict:
                    accept(name, weight_dict[name])
        else:
            index = json.loads((Path(directory) / "model.safetensors.index.json").read_text())["weight_map"]
            files = defaultdict(list)
            for name in remaining:
                if name in index:
                    files[index[name]].append(name)
            for filename, names in sorted(files.items()):
                with safe_open(str(Path(directory) / filename), framework="pt", device="cpu") as source:
                    for name in names:
                        accept(name, source.get_slice(name))
        if remaining:
            raise ValueError(
                f"missing required DeepSeek-V4.1 text weights: {sorted(remaining)[:8]} ({len(remaining)} total)"
            )
        for layer in self.layers:
            native = layer.attn_wo_a
            matrix = native.weight.value
            if native.scale is not None:
                scale = native.scale.value.float()
                matrix = (
                    (
                        matrix.float().view(matrix.shape[0] // 32, 32, matrix.shape[1] // 32, 32)
                        * scale[:, None, :, None]
                    )
                    .reshape(matrix.shape)
                    .to(torch.bfloat16)
                )
            layer.output_groups = matrix
            delattr(layer, "attn_wo_a")
