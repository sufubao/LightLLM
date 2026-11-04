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

import torch
import torch.nn.functional as F

from lightllm.core.compute.attention.shared_compressed.backend import rms_norm, SharedCompressedInputs


class DeepseekV41Layer:
    is_moe = True

    def __init__(self, layer, config, world, all_reduce):
        self.layer_num_, self.config, self.world = layer, config, world
        self.all_reduce = all_reduce

    def mix(self, x, fn, scale, base):
        from .kernels import hc_split_sinkhorn

        c = self.config
        flat = x.flatten(1).float()
        mixes = F.linear(flat, fn.value) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + c.norm_eps)
        return hc_split_sinkhorn(mixes.unsqueeze(0), scale.value, base.value, c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps)

    @staticmethod
    def collapse(x, coefficients):
        return (x.float() * coefficients.squeeze(0).unsqueeze(-1)).sum(1).to(x.dtype)

    @staticmethod
    def expand(value, residual, post, comb):
        result = post.squeeze(0).unsqueeze(-1) * value.unsqueeze(1)
        result += (comb.squeeze(0).unsqueeze(-1) * residual.unsqueeze(2)).sum(1)
        return result.to(value.dtype)

    def engram(self, x, hashes, w):
        c = self.config
        rows = c.engram_num_embeddings[c.engram_layer_ids.index(self.layer_num_)]
        table, scale = w.engram_embed_weight.value, w.engram_embed_scale.value
        part = table.shape[0]
        first = w.tp_rank * part
        mask = (hashes < first) | (hashes >= min(first + part, rows))
        local = (hashes - first).masked_fill(mask, 0)

        values = F.embedding(local, table.view(torch.uint8)).view(table.dtype).float().unflatten(-1, (-1, 32))
        scales = F.embedding(local, scale.view(torch.uint8)).view(scale.dtype).float()
        values = (values * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16).masked_fill(mask.unsqueeze(-1), 0)
        if self.all_reduce is not None:
            self.all_reduce(values)
        key, value = w.engram_wkv(values.flatten(-2)).split((c.hc_mult * c.dim, c.dim), -1)
        key = key.float().view(-1, c.hc_mult, c.dim)
        hidden = x.float()
        weight = w.engram_q_weight.value.float() * w.engram_k_weight.value.float()
        inv = torch.rsqrt(hidden.square().mean(-1) + c.norm_eps) * torch.rsqrt(key.square().mean(-1) + c.norm_eps)
        dot = (hidden * weight * key).sum(-1) * inv * c.dim ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(1)).to(x.dtype)

    def forward(self, x, pre_mix, metadata, w, phase, hashes=None):
        c = self.config
        if hashes is not None:
            x = self.engram(x, hashes, w)
        residual = x
        attn_pre, post, comb = self.mix(x, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
        hidden = rms_norm(self.collapse(x, pre_mix), w.attn_norm_weight.value, c.norm_eps)
        qr = rms_norm(w.attn_wq_a(hidden), w.attn_q_norm_weight.value, c.norm_eps)
        q = w.attn_wq_b(qr).view(-1, c.n_heads // self.world, c.head_dim)
        kv = rms_norm(w.attn_wkv(hidden), w.attn_kv_norm_weight.value, c.norm_eps)
        bound = self.bound_prefill_attention if phase == "prefill" else self.bound_decode_attention
        attention = bound.run(SharedCompressedInputs(q, kv, hidden, qr), metadata)
        from lightllm.core.formats.kernels.mxfp import act_quant

        grouped = attention.reshape(-1, c.o_groups // self.world, c.n_heads * c.head_dim // c.o_groups)
        act_quant(grouped, 32, "ue8m0", torch.float8_e8m0fnu, True)
        matrix = w.output_groups.view(c.o_groups // self.world, c.o_lora_rank, -1)
        projected = torch.einsum("sgd,grd->sgr", grouped, matrix).flatten(1)
        hidden = w.attn_wo_b(projected)
        if self.all_reduce is not None:
            hidden = hidden.float()
            self.all_reduce(hidden)
            hidden = hidden.to(x.dtype)
        x = self.expand(hidden, residual, post, comb)
        residual = x
        next_pre, post, comb = self.mix(x, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base)
        hidden = rms_norm(self.collapse(x, attn_pre), w.ffn_norm_weight.value, c.norm_eps)
        logits = F.linear(hidden.float(), w.ffn_gate_weight.value.float())
        routed = self.bound_moe.run(hidden, logits)
        if self.all_reduce is not None:
            self.all_reduce(routed)
        gate, up = w.ffn_shared_experts_w1(hidden).float(), w.ffn_shared_experts_w3(hidden).float()
        if c.swiglu_limit > 0:
            gate, up = gate.clamp(max=c.swiglu_limit), up.clamp(-c.swiglu_limit, c.swiglu_limit)
        shared = w.ffn_shared_experts_w2((F.silu(gate) * up).to(hidden.dtype))
        x = self.expand((routed + shared).to(hidden.dtype), residual, post, comb)
        return x, next_pre
