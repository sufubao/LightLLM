"""Reference context-parallel execution for standard dense GQA attention.

The production attention backends are intentionally left unchanged here.  PCP
and DCP use this small, explicit path while their backend contracts are being
integrated: the data ownership and collective boundaries are visible, and the
same code can be exercised with a Gloo process group.  Startup validation keeps
unsupported attention layouts from silently taking this reference path.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.distributed as dist

from lightllm.common.attention_state.full_att import FullAttentionState
from lightllm.common.context_parallel.dcp import merge_attention_partials
from lightllm.common.context_parallel.pcp import PrefillContextParallelConfig, assemble_prefill_slices


def _all_gather_variable(tensor: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    """Gather tensors whose first dimension can differ between ranks."""

    world_size = dist.get_world_size(group)
    local_length = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=tensor.device)
    lengths = [torch.empty_like(local_length) for _ in range(world_size)]
    dist.all_gather(lengths, local_length, group=group)
    lengths = [int(item.item()) for item in lengths]
    max_length = max(lengths, default=0)
    padded = tensor.new_zeros((max_length, *tensor.shape[1:]))
    padded[: tensor.shape[0]].copy_(tensor)
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)
    return [value[:length] for value, length in zip(gathered, lengths)]


def _attention_from_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_positions: torch.Tensor,
    query_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one GQA query row and return its output and log denominator."""

    key_mask = key_positions <= query_position
    key = key[key_mask]
    value = value[key_mask]
    if key.shape[0] == 0:
        return torch.zeros_like(query), torch.full(
            (query.shape[0],), -torch.inf, device=query.device, dtype=torch.float32
        )

    query_heads, head_dim = query.shape
    kv_heads = key.shape[1]
    if query_heads % kv_heads != 0:
        raise ValueError(f"query heads {query_heads} must be divisible by KV heads {kv_heads}")
    groups = query_heads // kv_heads
    grouped_query = query.view(kv_heads, groups, head_dim).float()
    key = key.float()
    value = value.float()
    scores = torch.einsum("hgd,khd->hgk", grouped_query, key) / (head_dim ** 0.5)
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("hgk,khd->hgd", probs, value).reshape(query_heads, head_dim)
    return output.to(dtype=query.dtype), lse.reshape(-1)


def _state_layer_view(infer_state, layer_num: int) -> tuple[torch.Tensor, torch.Tensor]:
    state = infer_state.att_state_manager.get_state(
        model_role=infer_state.att_state_role,
        state_type_filter="gqa",
        logical_layer_id=layer_num,
        state_class=FullAttentionState,
    )
    key, value = state.get_layer_view(logical_layer_id=layer_num)
    if key.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise RuntimeError("context-parallel reference attention requires floating-point KV storage")
    return key, value


def _request_rows(state, token_count: int, *, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Map a flattened query token to ``(request_index, position_in_request)``."""

    q_lengths = (state.b_seq_len - state.b_ready_cache_len).detach().cpu().tolist()
    rows: list[tuple[int, int]] = []
    for request_index, length in enumerate(q_lengths):
        rows.extend((request_index, offset) for offset in range(int(length)))
    end = len(rows) if end is None else end
    local_rows = rows[start:end]
    if len(local_rows) != token_count:
        raise ValueError(f"prefill metadata describes {len(local_rows)} local tokens, received {token_count}")
    return local_rows


def _request_kv(state, key: torch.Tensor, value: torch.Tensor, request_index: int, sequence_length: int):
    request_id = state.b_req_idx[request_index]
    slots = state.req_manager.req_to_token_indexs[request_id, :sequence_length]
    return key.index_select(0, slots), value.index_select(0, slots)


@dataclass
class ContextParallelState:
    """Per-inference state attached to an :class:`InferStateInfo`."""

    executor: "ContextParallelExecutor"
    prefill_metadata: object | None = None
    prefill_config: PrefillContextParallelConfig | None = None
    _saved: dict[str, object] | None = None

    @property
    def is_prefill(self) -> bool:
        return self.executor.pcp_size > 1

    @property
    def is_decode(self) -> bool:
        return self.executor.dcp_size > 1

    def prepare_prefill(self, infer_state) -> None:
        if self.executor.pcp_size == 1:
            return
        if not infer_state.is_prefill:
            raise ValueError("PCP state can only prepare prefill batches")
        token_count = int(infer_state.input_ids.shape[0])
        config = PrefillContextParallelConfig(
            global_rank=self.executor.groups.layout.global_rank,
            tp_size=self.executor.groups.layout.tp_size,
            pcp_size=self.executor.pcp_size,
            dp_size=self.executor.groups.layout.dp_size,
        )
        token_range = config.token_range(token_count)
        self.prefill_metadata = config.token_metadata(token_count)
        self.prefill_config = config
        self._saved = {"total_token_num": infer_state.total_token_num}
        for name in ("input_ids", "mem_index", "position_ids", "position_cos", "position_sin"):
            value = getattr(infer_state, name, None)
            if isinstance(value, torch.Tensor) and value.shape[0] == token_count:
                self._saved[name] = value
                setattr(infer_state, name, value[token_range.start : token_range.end].contiguous())
        infer_state.total_token_num = token_range.length

    def finish_prefill(self, infer_state, local_output: torch.Tensor) -> torch.Tensor:
        if self.executor.pcp_size == 1:
            return local_output
        if self.prefill_metadata is None or self._saved is None:
            raise RuntimeError("PCP output was finalized without preparing the state")
        local_slices = _all_gather_variable(local_output, self.executor.groups.context)
        ranges = self.prefill_config.token_ranges(self.prefill_metadata.token_count)
        output = assemble_prefill_slices(local_slices, ranges, self.prefill_metadata.token_count)
        for name, value in self._saved.items():
            if name == "total_token_num":
                setattr(infer_state, name, value)
            else:
                setattr(infer_state, name, value)
        self._saved = None
        self.prefill_metadata = None
        self.prefill_config = None
        return output

    def prefill_attention(self, layer, query: torch.Tensor, cache_kv: torch.Tensor, infer_state) -> torch.Tensor:
        key, value = _state_layer_view(infer_state, layer.layer_num_)
        update_indices = infer_state.mem_index
        self._replicate_prefill_updates(infer_state, update_indices, cache_kv, layer.layer_num_)
        rows = _request_rows(
            infer_state,
            query.shape[0],
            start=self.prefill_metadata.global_start,
            end=self.prefill_metadata.global_end,
        )
        if not rows:
            return query.new_empty(query.shape)
        outputs = []
        q = query.view(-1, layer.tp_q_head_num_, layer.head_dim_)
        for row, (request_index, offset) in enumerate(rows):
            sequence_length = int(infer_state.b_seq_len[request_index].item())
            ready_length = int(infer_state.b_ready_cache_len[request_index].item())
            request_key, request_value = _request_kv(infer_state, key, value, request_index, sequence_length)
            positions = torch.arange(sequence_length, device=query.device, dtype=torch.int64)
            output, _ = _attention_from_kv(q[row], request_key, request_value, positions, ready_length + offset)
            outputs.append(output.reshape(-1))
        return torch.stack(outputs, dim=0).to(dtype=query.dtype)

    def _replicate_prefill_updates(self, infer_state, indices: torch.Tensor, kv: torch.Tensor, layer_num: int) -> None:
        gathered_indices = _all_gather_variable(indices, self.executor.groups.context)
        gathered_kv = _all_gather_variable(kv, self.executor.groups.context)
        state = infer_state.att_state_manager.get_state(
            model_role=infer_state.att_state_role,
            state_type_filter="gqa",
            logical_layer_id=layer_num,
            state_class=FullAttentionState,
        )
        for remote_indices, remote_kv in zip(gathered_indices, gathered_kv):
            if remote_indices.numel():
                state.write_full_att_kv(logical_layer_id=layer_num, token_indices=remote_indices, kv=remote_kv)

    def decode_attention(self, layer, query: torch.Tensor, infer_state) -> torch.Tensor:
        if self.executor.dcp_size == 1:
            return layer._token_attention_kernel(query, infer_state, None)
        key, value = _state_layer_view(infer_state, layer.layer_num_)
        q = query.view(-1, layer.tp_q_head_num_, layer.head_dim_)
        context_rank = self.executor.groups.layout.context_rank
        virtual_block_size = self.executor.block_size * self.executor.dcp_size
        outputs = []
        lses = []
        for request_index in range(q.shape[0]):
            sequence_length = int(infer_state.b_seq_len[request_index].item())
            request_key, request_value = _request_kv(infer_state, key, value, request_index, sequence_length)
            positions = torch.arange(sequence_length, device=query.device, dtype=torch.int64)
            owner = (positions % virtual_block_size) % self.executor.dcp_size == context_rank
            output, lse = _attention_from_kv(
                q[request_index], request_key[owner], request_value[owner], positions[owner], sequence_length - 1
            )
            outputs.append(output)
            lses.append(lse)
        local_output = torch.stack(outputs, dim=0)
        local_lse = torch.stack(lses, dim=0)
        gathered_output = [torch.empty_like(local_output) for _ in range(self.executor.dcp_size)]
        gathered_lse = [torch.empty_like(local_lse) for _ in range(self.executor.dcp_size)]
        dist.all_gather(gathered_output, local_output, group=self.executor.groups.context)
        dist.all_gather(gathered_lse, local_lse, group=self.executor.groups.context)
        merged, _ = merge_attention_partials(torch.stack(gathered_output), torch.stack(gathered_lse))
        return merged.reshape(query.shape).to(dtype=query.dtype)


class ContextParallelExecutor:
    """Own the immutable topology and create short-lived per-batch state."""

    def __init__(self, groups, *, pcp_size: int, dcp_size: int, block_size: int = 1):
        if pcp_size < 1 or dcp_size < 1:
            raise ValueError("context-parallel degrees must be at least one")
        if pcp_size > 1 and dcp_size > 1 and pcp_size != dcp_size:
            raise ValueError("PCP and DCP degrees must match when both are enabled")
        self.groups = groups
        self.pcp_size = pcp_size
        self.dcp_size = dcp_size
        self.block_size = block_size

    @property
    def enabled(self) -> bool:
        return self.pcp_size > 1 or self.dcp_size > 1

    def attach(self, infer_state) -> ContextParallelState:
        state = ContextParallelState(self)
        infer_state.context_parallel = state
        return state
