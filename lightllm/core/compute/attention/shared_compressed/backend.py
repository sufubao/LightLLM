# MIT License
#
# Copyright (c) 2023 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Adapted from deepseek-ai/DeepSeek-V4.1-Flash, revision 314113f8a7dadb388215396ba5fbee2c610a04d6.

from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F

from lightllm.core.contracts.provider import AttentionOffer, ProbeResult
from .spec import SharedCompressedAttentionSpec


def rms_norm(x, weight, eps):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight).to(x.dtype)


def rotary(x, frequencies, inverse=False):
    value = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    freq = frequencies.conj() if inverse else frequencies
    while freq.ndim < value.ndim:
        freq = freq.unsqueeze(-2)
    x.copy_(torch.view_as_real(value * freq).flatten(-2))
    return x


def rotary_frequencies(dim, length, base, original, factor, beta_fast, beta_slow, device):
    frequencies = 1 / base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    if original:

        def correction(rotations):
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low, high = max(math.floor(correction(beta_fast)), 0), min(math.ceil(correction(beta_slow)), dim - 1)
        smooth = 1 - ((torch.arange(dim // 2, device=device) - low) / max(high - low, 1e-3)).clamp(0, 1)
        frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth
    phases = torch.outer(torch.arange(length, device=device), frequencies)
    return torch.polar(torch.ones_like(phases), phases)


@dataclass(frozen=True)
class SharedCompressedParams:
    spec: SharedCompressedAttentionSpec
    frequencies: torch.Tensor
    sink: torch.Tensor
    norm_eps: float
    compressor: object = None
    indexer: object = None
    all_reduce: object = None


@dataclass
class SharedCompressedInputs:
    q: torch.Tensor
    kv: torch.Tensor
    hidden: torch.Tensor
    query_latent: torch.Tensor


@dataclass
class SharedCompressedMetadata:
    requests: list
    req_to_tokens: torch.Tensor
    shared: dict = field(default_factory=dict)


class SharedCompressedBackend:
    supports_auxiliary_states = True
    implementation = "tilelang_shared_compressed"

    def __init__(self, phase):
        self.phase, self.name = phase, f"{self.implementation}.{phase}"

    def probe(self, spec, fmt, caps, limits, auxiliary_formats=None):
        if not isinstance(spec, SharedCompressedAttentionSpec):
            return ProbeResult(reason="requires shared compressed attention semantics")
        if not caps.is_nvidia or caps.capability is None or caps.capability < (9, 0):
            return ProbeResult(reason="TileLang shared compressed attention requires NVIDIA Hopper or newer")
        if self.phase not in ("prefill", "decode") or limits.graph_phases or limits.microbatch_count != 1:
            return ProbeResult(
                reason="shared compressed attention currently supports eager prefill/decode without overlap"
            )
        if limits.dynamic_verify or limits.max_verify_tokens != 1 or limits.expert_parallel or limits.transfer_paths:
            return ProbeResult(
                reason="shared compressed attention does not implement verification, EP or cache transfer"
            )
        if (
            fmt.layout != "request_mxfp_window"
            or fmt.encoding != "fp8_roundtrip_bf16"
            or fmt.dtype != torch.bfloat16
            or fmt.version != 1
            or fmt.fields != ("window", "partial", "gate")
            or fmt.field_dtypes != (torch.bfloat16, torch.float32, torch.float32)
        ):
            return ProbeResult(reason="requires explicit FP8-roundtrip BF16 window storage")
        for other in (auxiliary_formats or {}).values():
            if (
                other.layout != "token_mxfp_latent"
                or other.encoding != "fp4_roundtrip_bf16"
                or other.dtype != torch.bfloat16
                or other.version != 1
                or other.fields != ("latent",)
            ):
                return ProbeResult(reason="requires explicit FP4-roundtrip BF16 history storage")
        return ProbeResult((AttentionOffer(self.name, self.phase, fmt),))

    def bind(self, offer, view, params, workspace, auxiliary_views=None):
        return BoundSharedCompressed(offer, view, params, auxiliary_views or {})


class BoundSharedCompressed:
    def __init__(self, offer, view, params, auxiliary):
        self.phase, self.name = offer.phase, offer.backend
        self.spec, self.params = params.spec, params
        self.window, self.partial, self.gate = view.fields
        self.compressed = auxiliary.get("compressed")
        self.index = auxiliary.get("index")
        self.metadata_key = (view.state.role, "shared_compressed", self.phase)

    def create_metadata(self, batch):
        reqs, lengths = batch.req_indices.tolist(), batch.seq_lengths.tolist()
        counts = (batch.cu_q[1:] - batch.cu_q[:-1]).tolist()
        starts = batch.cu_q[:-1].tolist()
        requests = [(r, n - count, count, offset) for r, n, count, offset in zip(reqs, lengths, counts, starts)]
        return SharedCompressedMetadata(requests, batch.req_to_tokens)

    def run(self, inputs, metadata, out=None):
        from lightllm.core.formats.kernels.mxfp import act_quant, fp4_act_quant
        from .tilelang import sparse_attn

        spec, p = self.spec, self.params
        result = torch.empty_like(inputs.q) if out is None else out
        for req, start, count, offset in metadata.requests:
            if not count:
                continue
            end = start + count
            x, qr = inputs.hidden[offset : offset + count], inputs.query_latent[offset : offset + count]
            q, kv = inputs.q[offset : offset + count], inputs.kv[offset : offset + count]
            frequencies = p.frequencies[start:end]
            rotary(q[..., -spec.rope_dim :], frequencies)
            rotary(kv[..., -spec.rope_dim :], frequencies)
            act_quant(kv, 32, "ue8m0", torch.float8_e8m0fnu, True)
            prefix = min(start, spec.window_size - 1)
            positions = torch.arange(start - prefix, start, device=kv.device)
            history = self.window[req, positions % spec.window_size]
            window = torch.cat((history, kv), dim=0)
            rows = torch.arange(count, device=kv.device)[:, None] + prefix
            if self.phase == "prefill":
                width = min(end, spec.window_size)
                window_indices = (rows - spec.window_size + 1).clamp_min(0) + torch.arange(width, device=kv.device)
                window_indices = window_indices.masked_fill(window_indices > rows, -1).int()
            else:
                window_indices = rows - spec.window_size + 1 + torch.arange(spec.window_size, device=kv.device)
                window_indices = window_indices.masked_fill(window_indices < 0, -1).int()
            keep = min(end, spec.window_size)
            self.window[req, torch.arange(end - keep, end, device=kv.device) % spec.window_size] = window[-keep:]
            values, indices = window, window_indices
            if spec.ratio:
                compressed, selected = self._compressed(x, qr, req, start, end, window.shape[0], metadata)
                values, indices = torch.cat((window, compressed)), torch.cat((indices, selected), -1)
            output = sparse_attn(
                q.unsqueeze(0), values.unsqueeze(0), p.sink, indices.unsqueeze(0).contiguous(), spec.head_dim ** -0.5
            )[0]
            rotary(output[..., -spec.rope_dim :], frequencies, inverse=True)
            result[offset : offset + count].copy_(output)
        return result

    def _compressed(self, x, qr, req, start, end, offset, metadata):
        from lightllm.core.formats.kernels.mxfp import fp4_act_quant

        s, p, ratio = self.spec, self.params, self.spec.ratio
        slots = metadata.req_to_tokens[req, ratio - 1 : end : ratio].long()
        latent = None
        first = start // ratio
        if s.layer_id == s.kv_source:
            compressor = p.compressor
            if ratio == 1:
                latent = rms_norm(compressor.wkv(x), compressor.norm, p.norm_eps)
            else:
                tail = start % ratio
                candidate, scores = compressor.wkv(x.float()), compressor.wgate(x.float())
                candidate = torch.cat((self.partial[req, :tail], candidate))
                scores = torch.cat((self.gate[req, :tail], scores))
                cutoff = candidate.shape[0] // ratio * ratio
                if cutoff:
                    pooled = (
                        (
                            candidate[:cutoff].view(-1, ratio, s.head_dim)
                            * scores[:cutoff].view(-1, ratio, s.head_dim).softmax(1)
                        )
                        .sum(1)
                        .to(x.dtype)
                    )
                    latent = rms_norm(pooled, compressor.norm, p.norm_eps)
                if cutoff < candidate.shape[0]:
                    self.partial[req, : candidate.shape[0] - cutoff] = candidate[cutoff:]
                    self.gate[req, : candidate.shape[0] - cutoff] = scores[cutoff:]
            if latent is not None:
                freq = p.frequencies[first * ratio : end : ratio][: latent.shape[0]]
                key = rms_norm(p.indexer.wk(latent), p.indexer.norm, p.norm_eps)
                rotary(key[..., -s.rope_dim :], freq)
                fp4_act_quant(key, 32, True)
                self.index.fields[0][slots[first:]] = key
                rotary(latent[..., -s.rope_dim :], freq)
                fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
                self.compressed.fields[0][slots[first:]] = latent
        shared_key = (req, s.index_source)
        if s.layer_id == s.index_source:
            if not len(slots):
                selected = torch.empty(x.shape[0], 0, dtype=torch.int32, device=x.device)
            else:
                indexer = p.indexer
                q = indexer.wq_b(qr).view(x.shape[0], s.index_heads, s.index_dim)
                rotary(q[..., -s.rope_dim :], p.frequencies[start:end])
                fp4_act_quant(q, 32, True)
                key = self.index.fields[0][slots]
                weight = indexer.weights_proj(x) * (s.index_dim ** -0.5 * s.total_index_heads ** -0.5)
                scores = (torch.einsum("shd,td->sht", q, key).relu_() * weight.unsqueeze(-1)).sum(1)
                if p.all_reduce is not None:
                    p.all_reduce(scores)
                visible = (torch.arange(start + 1, end + 1, device=x.device) // ratio)[:, None]
                scores.masked_fill_(torch.arange(len(slots), device=x.device) >= visible, -torch.inf)
                if s.layer_id == s.candidate_source:
                    block = s.candidate_block_size
                    block_scores = (
                        F.pad(scores, (0, -len(slots) % block), value=-torch.inf).unflatten(-1, (-1, block)).amax(-1)
                    )
                    block_scores.masked_fill_(
                        torch.arange(block_scores.shape[-1], device=x.device) == (visible - 1) // block, torch.inf
                    )
                    top = block_scores.topk(min(s.candidate_blocks, block_scores.shape[-1]), -1)
                    keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(
                        -1, top.indices, top.values > -torch.inf
                    )
                    metadata.shared[req, "candidates"] = keep.repeat_interleave(block, -1)[:, : len(slots)]
                elif s.candidate_source >= 0 and s.layer_id > s.candidate_source:
                    scores.masked_fill_(~metadata.shared[req, "candidates"], -torch.inf)
                selected = scores.topk(min(s.index_topk, len(slots)), -1, sorted=False).indices.sort(-1).values
                selected = torch.where(selected < visible, selected + offset, -1).int()
            metadata.shared[shared_key] = selected
        return self.compressed.fields[0][slots], metadata.shared[shared_key]
