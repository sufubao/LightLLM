import math

import torch
import torch.nn.functional as F

from lightllm.distributed import all_gather_into_tensor
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import (
    Qwen35TransformerLayerInfer,
)
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.qwen4_exp.mem_manager import Qwen4ExpMemManager
from lightllm.models.qwen4_exp.triton_kernel.qsa import (
    qsa_mrope_fwd,
    qsa_select_tokens,
    qsa_sparse_attention,
    qsa_store_rows,
)
from ..hyperconnection import (
    grouped_gemma_rmsnorm,
    hyperconnection_combine,
    hyperconnection_combine_norm,
    hyperconnection_mix,
    hyperconnection_silu,
)
from ..ple import (
    build_decode_ngram_ids,
    build_mtp_conv_window,
    build_packed_ngram_ids,
    expand_mtp_decode_contexts,
    packed_ple_conv1d,
    reset_ple_new_request_state,
)


class Qwen4ExpTransformerLayerInfer(Qwen35TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.hidden_size = network_config["hidden_size"]
        self.hc_count = network_config["hc_count"]
        self.hyper_hidden_size = self.hidden_size * self.hc_count
        self.ple_layer_index = (
            network_config.get("ple_layer_ids", []).index(layer_num + 1)
            if layer_num + 1 in network_config.get("ple_layer_ids", [])
            else None
        )
        self.ngram_size = network_config.get("ngram_size", 3)
        self.heads_per_ngram = network_config.get("heads_per_ngram", 8)
        self.eos_token_id = network_config["eos_token_id"]
        if isinstance(self.eos_token_id, list):
            self.eos_token_id = self.eos_token_id[0]
        self.qsa_runtime_enabled = network_config.get(
            "_qsa_runtime_enabled", False
        )
        self.qsa_token_topk = network_config.get("indexer_budget", 2048)
        self.qsa_compress_ratio = network_config.get(
            "indexer_compress_ratio", 4
        )
        self.qsa_index_n_heads = network_config.get("indexer_n_heads", 0)
        self.qsa_index_head_dim = network_config.get("indexer_head_dim", 0)
        self.qsa_tp_index_n_heads = (
            self.qsa_index_n_heads // self.tp_world_size_
            if self.qsa_index_n_heads
            else 0
        )
        self.qsa_rotary_dim = int(
            self.head_dim_ * network_config.get("partial_rotary_factor", 1.0)
        )

    def _qsa_row_metadata(self, infer_state):
        if infer_state.is_prefill:
            batch_ids = torch.repeat_interleave(
                torch.arange(
                    infer_state.b_q_seq_len.numel(),
                    dtype=torch.int32,
                    device=infer_state.b_q_seq_len.device,
                ),
                infer_state.b_q_seq_len,
                output_size=infer_state.input_ids.shape[0],
            )
            row_offsets = (
                torch.arange(
                    batch_ids.numel(),
                    dtype=torch.int32,
                    device=batch_ids.device,
                )
                - infer_state.b1_cu_q_seq_len.index_select(
                    0, batch_ids.long()
                )
            )
            logical_positions = (
                infer_state.b_ready_cache_len.index_select(
                    0, batch_ids.long()
                )
                + row_offsets
            )
        else:
            batch_ids = torch.arange(
                infer_state.b_seq_len.numel(),
                dtype=torch.int32,
                device=infer_state.b_seq_len.device,
            )
            logical_positions = infer_state.b_seq_len - 1
        row_req_ids = infer_state.b_req_idx.index_select(0, batch_ids.long())
        row_sequence_lengths = infer_state.b_seq_len.index_select(
            0, batch_ids.long()
        )
        return (
            row_req_ids.contiguous(),
            logical_positions.to(torch.int32).contiguous(),
            row_sequence_lengths.to(torch.int32).contiguous(),
        )

    def _run_qsa_indexer(self, input, infer_state, indexer_weight):
        if infer_state.need_dp_prefill_balance:
            raise RuntimeError("Qwen4 QSA does not support DP prefill balancing yet")
        mem_manager = infer_state.mem_manager
        if not isinstance(mem_manager, Qwen4ExpMemManager):
            raise TypeError("Qwen4 QSA requires Qwen4ExpMemManager")

        rows = input.shape[0]
        local_q = indexer_weight.q_proj.mm(input).view(
            rows, self.qsa_tp_index_n_heads, self.qsa_index_head_dim
        )
        if self.tp_world_size_ > 1:
            gathered_q = torch.empty(
                self.tp_world_size_ * local_q.numel(),
                dtype=local_q.dtype,
                device=local_q.device,
            )
            all_gather_into_tensor(
                output_=gathered_q,
                input_=local_q.view(-1),
                group=infer_state.dist_group,
                async_op=False,
            )
            index_q = (
                gathered_q.view(
                    self.tp_world_size_,
                    rows,
                    self.qsa_tp_index_n_heads,
                    self.qsa_index_head_dim,
                )
                .transpose(0, 1)
                .contiguous()
                .view(rows, self.qsa_index_n_heads, self.qsa_index_head_dim)
            )
        else:
            index_q = local_q
        raw_keys = indexer_weight.k_proj.mm(input).view(
            rows, self.qsa_index_head_dim
        )

        index_q = grouped_gemma_rmsnorm(
            index_q.reshape(-1, self.qsa_index_head_dim),
            indexer_weight.q_norm.weight,
            hidden_size=self.qsa_index_head_dim,
            eps=self.eps_,
        ).view_as(index_q)
        qsa_mrope_fwd(
            index_q,
            infer_state.position_cos,
            infer_state.position_sin,
            self.mrope_section,
            rotary_dim=self.qsa_rotary_dim,
            is_interleaved=True,
        )

        raw_cache = mem_manager.get_qsa_raw_key_buffer(self.layer_num_)
        compressed_cache = mem_manager.get_qsa_compressed_key_buffer(
            self.layer_num_
        )
        mem_indices = infer_state.mem_index.long()
        raw_cache.index_copy_(0, mem_indices, raw_keys)

        position_cos_rows = infer_state.position_cos.permute(1, 0, 2).contiguous()
        position_sin_rows = infer_state.position_sin.permute(1, 0, 2).contiguous()
        mem_manager.qsa_position_cos_buffer.index_copy_(
            0, mem_indices, position_cos_rows
        )
        mem_manager.qsa_position_sin_buffer.index_copy_(
            0, mem_indices, position_sin_rows
        )

        row_req_ids, logical_positions, row_sequence_lengths = (
            self._qsa_row_metadata(infer_state)
        )
        group_offsets = torch.arange(
            self.qsa_compress_ratio - 1,
            -1,
            -1,
            dtype=torch.int32,
            device=logical_positions.device,
        )
        group_positions = (
            logical_positions[:, None] - group_offsets[None, :]
        ).clamp_(min=0)
        req_to_token_indexs = infer_state.req_manager.req_to_token_indexs
        group_mem_indices = req_to_token_indexs[
            row_req_ids.long().unsqueeze(1), group_positions.long()
        ]
        pooled_keys = (
            raw_cache[group_mem_indices.long()]
            .to(torch.float32)
            .mean(dim=1)
            .to(raw_keys.dtype)
        )
        compressed_keys = grouped_gemma_rmsnorm(
            pooled_keys,
            indexer_weight.k_norm.weight,
            hidden_size=self.qsa_index_head_dim,
            eps=self.eps_,
        ).view(rows, 1, self.qsa_index_head_dim)
        first_mem_indices = group_mem_indices[:, 0].long()
        first_cos = mem_manager.qsa_position_cos_buffer.index_select(
            0, first_mem_indices
        ).permute(1, 0, 2)
        first_sin = mem_manager.qsa_position_sin_buffer.index_select(
            0, first_mem_indices
        ).permute(1, 0, 2)
        qsa_mrope_fwd(
            compressed_keys,
            first_cos,
            first_sin,
            self.mrope_section,
            rotary_dim=self.qsa_rotary_dim,
            is_interleaved=True,
        )
        completed_groups = (
            logical_positions % self.qsa_compress_ratio
        ) == (self.qsa_compress_ratio - 1)
        qsa_store_rows(
            compressed_cache,
            group_mem_indices[:, 0],
            compressed_keys[:, 0],
            completed_groups,
        )

        infer_state.qsa_row_req_ids = row_req_ids
        if infer_state.max_kv_seq_len > self.qsa_token_topk:
            infer_state.qsa_logical_indices = qsa_select_tokens(
                index_q,
                compressed_cache,
                req_to_token_indexs,
                row_req_ids,
                logical_positions,
                row_sequence_lengths,
                max_sequence_length=infer_state.max_kv_seq_len,
                token_topk=self.qsa_token_topk,
                compress_ratio=self.qsa_compress_ratio,
            )
        else:
            infer_state.qsa_logical_indices = None

    def _get_qkv(self, input, infer_state, layer_weight):
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        if self.qsa_runtime_enabled and not self.is_linear_attention_layer:
            self._run_qsa_indexer(input, infer_state, layer_weight.qsa_indexer)

        qkv_gate_out = layer_weight.qkvo_gate_proj.mm(input)
        qkv_out, o_gate = qkv_gate_out.split(
            [
                self.tp_q_head_num_ * self.head_dim_
                + (self.tp_k_head_num_ + self.tp_v_head_num_)
                * self.head_dim_,
                self.tp_q_head_num_ * self.head_dim_,
            ],
            dim=-1,
        )
        q, cache_kv = qkv_out.split(
            [
                self.tp_q_head_num_ * self.head_dim_,
                (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_,
            ],
            dim=-1,
        )
        infer_state.gate_logics_value = o_gate
        layer_weight.qk_norm_weight_(
            q,
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            eps=self.eps_,
        )
        cache_kv = cache_kv.view(
            -1,
            self.tp_k_head_num_ + self.tp_v_head_num_,
            self.head_dim_,
        )
        mrope_triton_fused(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            self.mrope_section,
            is_interleaved=True,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
        return q, cache_kv

    def _qsa_attention_kernel(self, q, infer_state):
        if infer_state.qsa_logical_indices is None:
            raise RuntimeError("QSA sparse attention is missing selected indices")
        key_cache, value_cache = infer_state.mem_manager.get_att_input_params(
            layer_index=self.layer_num_
        )
        return qsa_sparse_attention(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            key_cache,
            value_cache,
            infer_state.qsa_logical_indices,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.qsa_row_req_ids,
        ).view(q.shape)

    def _context_attention_kernel(
        self, q, kv, infer_state, layer_weight, out=None
    ):
        if (
            self.qsa_runtime_enabled
            and not self.is_linear_attention_layer
            and infer_state.max_kv_seq_len > self.qsa_token_topk
        ):
            return self._qsa_attention_kernel(q, infer_state)
        return super()._context_attention_kernel(q, kv, infer_state, layer_weight)

    def _token_attention_kernel(
        self, q, infer_state, layer_weight, out=None
    ):
        if (
            self.qsa_runtime_enabled
            and not self.is_linear_attention_layer
            and infer_state.max_kv_seq_len > self.qsa_token_topk
        ):
            return self._qsa_attention_kernel(q, infer_state)
        return super()._token_attention_kernel(q, infer_state, layer_weight)

    def _hyper_mix(
        self,
        hidden_states,
        hyper_weight,
        pending_output=None,
        pending_injection=None,
    ):
        if pending_output is None:
            normalized = grouped_gemma_rmsnorm(
                hidden_states,
                hyper_weight.hc_norm.weight,
                hidden_size=self.hidden_size,
                eps=self.eps_,
            )
        else:
            hidden_states, normalized = hyperconnection_combine_norm(
                hidden_states,
                pending_output,
                pending_injection,
                hyper_weight.hc_norm.weight,
                hidden_size=self.hidden_size,
                eps=self.eps_,
                hc_count=self.hc_count,
            )
        if hyper_weight.input_mix_weight_down_block_inject is not None:
            down_and_injection = hyper_weight.input_mix_weight_down_block_inject.mm(
                normalized
            )
            split_outputs = down_and_injection.split(
                hyper_weight.input_mix_weight_down_block_inject.out_dims,
                dim=-1,
            )
            lowrank, injection_logits = split_outputs[:2]
        else:
            lowrank = hyper_weight.input_mix_weight_down.mm(normalized)
            injection_logits = None
        lowrank = hyperconnection_silu(lowrank, self.hc_count)
        gate_logits = hyper_weight.input_mix_weight_up.mm(lowrank)
        mixed = hyperconnection_mix(normalized, gate_logits, hc_count=self.hc_count)
        return hidden_states, mixed, injection_logits

    def _ple_forward(self, hidden_states, infer_state, ple_weight):
        if infer_state.need_dp_prefill_balance:
            raise RuntimeError("Qwen4 PLE does not support DP prefill balancing yet")
        req_ids = infer_state.b_req_idx.long()
        req_manager = infer_state.req_manager
        context_buffer = req_manager.req_to_ple_token_context
        conv_buffer = req_manager.req_to_ple_conv_state
        state_width = context_buffer.shape[1]
        flat_context_buffer = context_buffer.flatten(0, 1)
        flat_conv_buffer = conv_buffer.flatten(0, 1)

        if infer_state.is_prefill:
            new_request_mask = infer_state.b_ready_cache_len == 0
            reset_ple_new_request_state(
                req_ids=req_ids,
                new_request_mask=new_request_mask,
                state_indices=req_manager.req_to_ple_state_index,
                context_buffer=context_buffer,
                conv_buffer=conv_buffer,
                eos_token_id=self.eos_token_id,
            )
            cu_seqlens = infer_state.b1_cu_q_seq_len
        base_state_slots = req_manager.req_to_ple_state_index.index_select(
            0, req_ids
        ).long()
        base_flat_slots = req_ids * state_width + base_state_slots
        contexts = flat_context_buffer.index_select(0, base_flat_slots)
        base_conv_states = flat_conv_buffer.index_select(0, base_flat_slots)
        if infer_state.is_prefill:
            ngram_ids, next_contexts = build_packed_ngram_ids(
                infer_state.input_ids,
                cu_seqlens,
                contexts,
                layer_multipliers=req_manager.ple_layer_multipliers,
                head_vocab_sizes=req_manager.ple_head_vocab_sizes,
                head_offsets=req_manager.ple_head_offsets,
                ngram_size=self.ngram_size,
                heads_per_ngram=self.heads_per_ngram,
                eos_token_id=self.eos_token_id,
            )
        else:
            contexts = expand_mtp_decode_contexts(
                infer_state.input_ids,
                contexts,
                infer_state.b_mtp_index,
            )
            ngram_ids, next_contexts = build_decode_ngram_ids(
                infer_state.input_ids,
                contexts,
                layer_multipliers=req_manager.ple_layer_multipliers,
                head_vocab_sizes=req_manager.ple_head_vocab_sizes,
                head_offsets=req_manager.ple_head_offsets,
                ngram_size=self.ngram_size,
                heads_per_ngram=self.heads_per_ngram,
                eos_token_id=self.eos_token_id,
            )
        output_flat_slots = (
            req_ids * state_width + infer_state.b_mtp_index.long()
        )
        flat_context_buffer.index_copy_(0, output_flat_slots, next_contexts)

        token_num, ngram_heads = ngram_ids.shape
        embeddings = ple_weight.ngram_embedding(ngram_ids.reshape(-1)).view(
            token_num, -1
        )
        embeddings = self._tpsp_reduce(
            input=embeddings, infer_state=infer_state
        )

        key = ple_weight.key_proj.mm(embeddings)
        value = ple_weight.value_proj.mm(embeddings)
        key_normed = grouped_gemma_rmsnorm(
            key, ple_weight.norm_key.weight, hidden_size=self.hidden_size, eps=self.eps_
        ).unflatten(-1, (self.hc_count, self.hidden_size))
        query_normed = grouped_gemma_rmsnorm(
            hidden_states,
            ple_weight.norm_query.weight,
            hidden_size=self.hidden_size,
            eps=self.eps_,
        ).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(-1, keepdim=True) / math.sqrt(
            self.hidden_size
        )
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        conv_input = grouped_gemma_rmsnorm(
            gated_value,
            ple_weight.norm_conv.weight,
            hidden_size=self.hidden_size,
            eps=self.eps_,
        )

        if not infer_state.is_prefill:
            history = build_mtp_conv_window(
                conv_input,
                base_conv_states,
                infer_state.b_mtp_index,
            )
            flat_conv_buffer.index_copy_(
                0, output_flat_slots, history[:, :, 1:]
            )
            conv = F.conv1d(
                history,
                ple_weight.conv1d.weight,
                groups=self.hyper_hidden_size,
                dilation=self.ngram_size,
            )
            return gated_value + F.silu(conv.squeeze(-1))

        conv_output, next_conv_states = packed_ple_conv1d(
            conv_input,
            base_conv_states,
            cu_seqlens,
            ple_weight.conv1d.weight,
            dilation=self.ngram_size,
            max_query_len=infer_state.max_q_seq_len,
        )
        flat_conv_buffer.index_copy_(
            0, output_flat_slots, next_conv_states
        )
        return gated_value + conv_output

    def _forward(self, hidden_states, infer_state, layer_weight, *, is_prefill):
        if layer_weight.ple is not None:
            hidden_states = hidden_states + self._ple_forward(
                hidden_states, infer_state, layer_weight.ple
            )

        hidden_states, mixed, injection_logits = self._hyper_mix(
            hidden_states, layer_weight.attn_hyper_connection
        )
        if is_prefill:
            block_output = self.context_attention_forward(
                mixed, infer_state, layer_weight
            )
        else:
            block_output = self.token_attention_forward(
                mixed, infer_state, layer_weight
            )
        hidden_states, mixed, mlp_injection_logits = self._hyper_mix(
            hidden_states,
            layer_weight.mlp_hyper_connection,
            pending_output=block_output.view(-1, self.hidden_size),
            pending_injection=injection_logits,
        )
        block_output = self._ffn(mixed, infer_state, layer_weight)
        hidden_states = hyperconnection_combine(
            hidden_states,
            block_output.view(-1, self.hidden_size),
            mlp_injection_logits,
            hc_count=self.hc_count,
        )
        return hidden_states

    def context_forward(self, input_embdings, infer_state, layer_weight):
        return self._forward(input_embdings, infer_state, layer_weight, is_prefill=True)

    def token_forward(self, input_embdings, infer_state, layer_weight):
        return self._forward(
            input_embdings, infer_state, layer_weight, is_prefill=False
        )
