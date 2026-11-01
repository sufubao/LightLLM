from __future__ import annotations

from functools import lru_cache

import torch

from lightllm.utils.envs_utils import get_env_start_args
from ..utils import deepcopy_result
from .base_state import FullAttentionState
from lightllm.common.attention_state.storage.token import TokenStorage
from lightllm.common.attention_state.layouts import TokenStorageSpec
from lightllm.common.attention_state.transfer.token import TokenTransfer


class TokenAttentionState(FullAttentionState):
    def __init__(self, config, spec: TokenStorageSpec, writer, page_size: int, device):
        self.config = config
        self.resource = TokenStorage(spec)
        self.writer = writer
        self.transfer = TokenTransfer(self.resource)
        self.state_name = None
        self.page_size = page_size
        self.device = torch.device(device)

    @property
    def capacity(self):
        return self.resource.capacity

    @property
    def storage_dtype(self):
        return self.resource.spec.fields[0].dtype

    def get_state_type(self) -> str:
        return self.resource.spec.encoding

    def allocate_storage(self, token_capacity: int) -> None:
        self.writer.prepare(torch.device(self.device))
        self.resource.allocate(token_capacity, self.page_size, self.device)

    def get_token_bytes(self) -> int:
        return self.resource.spec.token_bytes

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        return self.resource.contains_layer(logical_layer_id)

    def get_storage_layer_index(self, logical_layer_id: int) -> int:
        return self.resource.layer_index(logical_layer_id)

    def get_layer_view(self, logical_layer_id: int):
        return self.writer.layer_view(self.resource, logical_layer_id)

    def write_full_att_kv(self, logical_layer_id, token_indices, kv) -> None:
        self.writer.write(self.resource, logical_layer_id, token_indices, kv)

    def copy_full_att_tokens(self, src_indices, dst_indices) -> None:
        self.transfer.copy_tokens(src_indices, dst_indices)

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self):
        return self._get_transfer_regions(get_env_start_args().cpu_cache_token_page_size)

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self):
        return self._get_transfer_regions(get_env_start_args().pd_kv_page_size)

    def write_to_cpu(self, token_indices, cpu_page_buffers, cpu_page_indices, layout, cpu_page_ready_flags=None):
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        self.transfer.pages(token_indices, cpu_page_buffers, cpu_page_indices, layout, True, cpu_page_ready_flags)

    def read_from_cpu(self, cpu_page_buffers, cpu_page_indices, token_indices, layout):
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        self.transfer.pages(token_indices, cpu_page_buffers, cpu_page_indices, layout, False)

    def write_to_pd_page(self, token_indices, pd_page_buffers, pd_page_index, layout):
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        self.transfer.pages(token_indices, pd_page_buffers, [pd_page_index], layout, True)

    def read_from_pd_page(self, pd_page_buffers, token_indices, pd_page_index, layout):
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        self.transfer.pages(token_indices, pd_page_buffers, [pd_page_index], layout, False)

    def copy_tokens_from_dp(self, src_token_indices, dst_token_indices, remote_dp_full_att_state):
        assert type(remote_dp_full_att_state) is type(self), "DP copy requires the same State type"
        self.transfer.copy_from_peer(remote_dp_full_att_state.resource, src_token_indices, dst_token_indices)

    def _get_transfer_regions(self, page_tokens: int):
        return self.resource.spec.transfer_regions(page_tokens)


class SingleTensorAttentionState(TokenAttentionState):
    @property
    def storage(self):
        return self.resource.buffers.get("kv")

    @storage.setter
    def storage(self, buffer):
        self.resource.buffers["kv"] = buffer
