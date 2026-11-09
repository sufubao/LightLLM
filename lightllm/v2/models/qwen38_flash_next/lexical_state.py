from dataclasses import dataclass
from math import prod

import torch

from lightllm.v2.attention_state.bounded_att import BoundedAttentionState
from lightllm.v2.attention_state.types import AxisMapping, TransferRegion


@dataclass(frozen=True)
class LexicalStateConfig:
    logical_layer_ids: tuple[int, ...]
    ngram_context: int
    conv_context: int
    width: int
    eos_token_id: int
    dtype: torch.dtype
    tp_rank: int = 0


class LexicalPages:
    def __init__(self, capacity, config, device, pin_memory):
        pin = pin_memory and torch.device(device).type == "cpu" and torch.cuda.is_available()
        self.ngram = torch.empty(
            (capacity, len(config.logical_layer_ids), config.ngram_context),
            dtype=torch.int64,
            device=device,
            pin_memory=pin,
        )
        self.convolution = torch.empty(
            (capacity, len(config.logical_layer_ids), config.conv_context, config.width),
            dtype=config.dtype,
            device=device,
            pin_memory=pin,
        )


class Qwen38LexicalState(BoundedAttentionState):
    def __init__(self, config, request_capacity, device="meta"):
        self.config = config
        self.request_capacity = request_capacity
        self.state_name = None
        self.ngram = torch.full(
            (len(config.logical_layer_ids), request_capacity, config.ngram_context),
            config.eos_token_id,
            dtype=torch.int64,
            device=device,
        )
        self.convolution = torch.zeros(
            (len(config.logical_layer_ids), request_capacity, config.conv_context, config.width),
            dtype=config.dtype,
            device=device,
        )
        self.big_pages_pool = None
        self.small_pages_pool = None

    def get_state_type(self):
        return "qwen38_lexical"

    def contains_logical_layer(self, logical_layer_id):
        return logical_layer_id in self.config.logical_layer_ids

    def get_layer_view(self, logical_layer_id):
        index = self.config.logical_layer_ids.index(logical_layer_id)
        return self.ngram[index], self.convolution[index]

    def initialize_page_pools(self, big_page_capacity, small_page_capacity=0, device="cpu", pin_memory=True):
        self.big_pages_pool = (
            LexicalPages(big_page_capacity, self.config, device, pin_memory) if big_page_capacity else None
        )
        self.small_pages_pool = (
            LexicalPages(small_page_capacity, self.config, device, pin_memory) if small_page_capacity else None
        )

    def reset_bounded_att_state(self, req_indices):
        for request in req_indices:
            self.ngram[:, request].fill_(self.config.eos_token_id)
            self.convolution[:, request].zero_()

    def _pool(self, page_kind):
        pool = self.big_pages_pool if page_kind == "big" else self.small_pages_pool
        if pool is None:
            raise ValueError(f"{page_kind} lexical pages are disabled")
        return pool

    def _copy_pages(self, req_indices, page_kind, page_slot_ids, write):
        pool = self._pool(page_kind)
        for request, slot in zip(req_indices, page_slot_ids):
            runtime = (self.ngram[:, request], self.convolution[:, request])
            snapshot = (pool.ngram[slot], pool.convolution[slot])
            for source, destination in zip(runtime, snapshot) if write else zip(snapshot, runtime):
                destination.copy_(source)

    def save_bounded_att_runtime_to_pages(self, req_indices, state_token_lens, page_kind, page_slot_ids):
        self._copy_pages(req_indices, page_kind, page_slot_ids, True)

    def restore_bounded_att_runtime_from_pages(self, req_indices, state_token_lens, page_kind, page_slot_ids):
        self._copy_pages(req_indices, page_kind, page_slot_ids, False)

    def commit_bounded_att_state(self, req_indices, accepted_steps):
        return None

    def get_cpu_transfer_regions(self):
        regions = []
        offset = 0
        for name, tensor, axes in (
            ("ngram", self.ngram, ("layer", "context")),
            ("convolution", self.convolution, ("layer", "context", "dim")),
        ):
            shape = tensor.shape[:1] + tensor.shape[2:]
            end = offset + prod(shape) * tensor.element_size()
            regions.append(
                TransferRegion(
                    field=name,
                    byte_start=offset,
                    byte_end=end,
                    dtype=tensor.dtype,
                    axes=axes,
                    shape=tuple(shape),
                    axis_mappings=tuple(
                        AxisMapping(axis=axis, local_start=0, external_start=0, length=size)
                        for axis, size in zip(axes[1:], shape[1:])
                    ),
                    write_enabled=self.config.tp_rank == 0,
                )
            )
            offset = end
        return tuple(regions)

    get_pd_transfer_regions = get_cpu_transfer_regions

    def _copy_transfer(self, page_kind, page_slot_ids, buffers, page_indices, write):
        pool = self._pool(page_kind)
        regions = self.get_cpu_transfer_regions()
        for slot, page in zip(page_slot_ids, page_indices):
            for name, region in (("ngram", regions[0]), ("convolution", regions[1])):
                source = getattr(pool, name)[slot].contiguous()
                target = buffers[page, region.byte_start : region.byte_end].view(source.dtype).view(source.shape)
                if write:
                    target.copy_(source)
                else:
                    source.copy_(target)

    def write_state_pages_to_cpu(
        self, page_kind, page_slot_ids, cpu_page_buffers, cpu_page_indices, layout, flags=None
    ):
        selected = [i for i, ready in enumerate(flags or ()) if not ready]
        if not flags:
            selected = range(len(page_slot_ids))
        self._copy_transfer(
            page_kind,
            [page_slot_ids[i] for i in selected],
            cpu_page_buffers,
            [cpu_page_indices[i] for i in selected],
            True,
        )

    def read_state_pages_from_cpu(self, cpu_page_buffers, cpu_page_indices, big_page_slot_ids, layout):
        self._copy_transfer("big", big_page_slot_ids, cpu_page_buffers, cpu_page_indices, False)

    def copy_big_pages_from_dp(self, src_page_ids, dst_page_ids, remote_dp_bounded_state):
        source_pool, target_pool = remote_dp_bounded_state._pool("big"), self._pool("big")
        for name in ("ngram", "convolution"):
            source, target = getattr(source_pool, name), getattr(target_pool, name)
            for src, dst in zip(src_page_ids, dst_page_ids):
                target[dst].copy_(source[src])

    def copy_runtime_state_from_dp(self, src_req_indices, dst_req_indices, remote_dp_bounded_state):
        for name in ("ngram", "convolution"):
            source, target = getattr(remote_dp_bounded_state, name), getattr(self, name)
            for src, dst in zip(src_req_indices, dst_req_indices):
                target[:, dst].copy_(source[:, src])

    def write_to_pd_page(self, req_idx, pd_page_buffers, pd_page_index, state_token_len, layout):
        self._copy_transfer("big", [req_idx], pd_page_buffers, [pd_page_index], True)

    def read_from_pd_page(self, pd_page_buffers, req_idx, pd_page_index, state_token_len, layout):
        self._copy_transfer("big", [req_idx], pd_page_buffers, [pd_page_index], False)
