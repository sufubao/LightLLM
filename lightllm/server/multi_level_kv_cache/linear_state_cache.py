"""Shared CPU recurrent states, with independent capacity and KV tail aliases.

Each state owns one reference to the KV page containing its exact endpoint.
This lets a 10K checkpoint address the first 2K tokens of an 8K physical page
without keying that checkpoint by the unrelated suffix of the physical page.
Earlier KV pages remain independently evictable; a hit needs all of them.

Lock order is always state lock, then KV lock. Callers hold both for metadata
transactions. Tensor transfers happen while allocation references are held;
READY is published only after every TP rank has finished copying.
"""

import ctypes

import torch

from lightllm.common.linear_att_cache_manager.checkpoints import prefix_hashes
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
from lightllm.utils.kv_cache_utils import CpuKVCacheMeta
from .cpu_cache_client import CpuKvCacheClient, _CpuPageStatus


class _LinearStateStatus(_CpuPageStatus):
    _pack_ = 4
    _fields_ = [("token_count", ctypes.c_int64), ("tail_page", ctypes.c_int64)]

    def init(self):
        super().init()
        self.token_count = 0
        self.tail_page = -1


class CpuLinearStateCacheClient(CpuKvCacheClient):
    def __init__(self, kv_client, only_create_meta_data, init_shm_data):
        args = get_env_start_args()
        self.kv_client = kv_client
        self.config = LinearAttCacheConfig.load_from_args()
        self.rank_conv_bytes = self.config.get_cpu_cache_conv_bytes() // self.config.tp_world_size
        self.rank_ssm_bytes = self.config.get_cpu_cache_ssm_bytes() // self.config.tp_world_size
        state_bytes = (self.rank_conv_bytes + self.rank_ssm_bytes) * self.config.tp_world_size
        meta = CpuKVCacheMeta(
            page_num=args.linear_att_cpu_cache_size,
            token_page_size=1,
            layer_num=1,
            num_heads=1,
            head_dim=state_bytes,
            data_type=torch.uint8,
            scale_head_dim=0,
            scale_data_type=torch.uint8,
        )
        super().__init__(
            only_create_meta_data,
            init_shm_data,
            cache_name=f"{get_unique_server_name()}_cpu_linear_state",
            tensor_meta=meta,
            # Startup KV/MM keys occupy [0, 123456789). Use a disjoint range
            # for the state segment, tied to the same server lifetime.
            tensor_shm_key=args.cpu_kv_cache_shm_id + 123456789,
            item_class=_LinearStateStatus,
        )

    def allocate_checkpoint(self, prefix_hash, token_count, tail_page):
        """Reserve a new state; an existing writer keeps exclusive ownership."""
        if self.page_hash_dict.get(prefix_hash) is not None:
            return None
        page = self.page_items.head.get_next_item()
        if page.self_index == self.page_items.tail.self_index or not page.can_realloc(False):
            return None
        if page.tail_page >= 0:
            self.kv_client.deref_one_page(page.tail_page)
            page.tail_page = -1
        page_idx = self.get_one_empty_page(prefix_hash, disk_offload_enable=False)
        assert page_idx is not None
        page.token_count = token_count
        page.tail_page = tail_page
        tail = self.kv_client.page_items.get_item_by_index(tail_page)
        tail.ref_count += 1
        if tail.ref_count == 1:
            tail.del_self_from_list()
        return page_idx

    def match(self, tokens, max_tokens):
        """Return a referenced state and complete KV coverage, or a miss."""
        from lightllm.server.core.objs.token_chunck_hash_list import LIGHTLLM_TOKEN_HASH_LIST_SIZE

        max_tokens = min(max_tokens, LIGHTLLM_TOKEN_HASH_LIST_SIZE * self.args.cpu_cache_token_page_size)
        lengths = {
            page.token_count
            for page in self.page_items.linked_items
            if page.is_data_ready() and 0 < page.token_count <= max_tokens
        }
        hashes = prefix_hashes(tokens, lengths)
        page_size = self.args.cpu_cache_token_page_size
        full_lengths = range(page_size, max_tokens + 1, page_size)
        full_hashes = prefix_hashes(tokens, full_lengths)
        for length in sorted(lengths, reverse=True):
            state_idx, _ = self.query_one_page(hashes[length])
            if state_idx is None:
                continue
            state = self.page_items.get_item_by_index(state_idx)
            assert state.token_count == length
            pages = []
            endpoints = list(range(page_size, length, page_size))
            for endpoint in endpoints:
                page_idx, _ = self.kv_client.query_one_page(full_hashes[endpoint])
                if page_idx is None:
                    break
                pages.append(page_idx)
            tail = self.kv_client.page_items.get_item_by_index(state.tail_page)
            if len(pages) == len(endpoints) and tail.is_data_ready():
                tail.ref_count += 1  # already pinned by this checkpoint
                pages.append(state.tail_page)
                return state_idx, pages, endpoints + [length]
            self.kv_client.deref_pages(pages)
            self.deref_one_page(state_idx)
        return -1, [], []

    def evict_one_checkpoint(self):
        """Release an idle LRU state's KV pin under KV allocation pressure.

        Borrowed states and pending writers are absent from the idle list.
        Both metadata locks must be held, in state -> KV order.
        """
        page = self.page_items.head.get_next_item()
        while page.self_index != self.page_items.tail.self_index:
            if page.is_data_ready() and page.ref_count == 0:
                self.page_hash_dict.remove(page.hash_key)
                self.kv_client.deref_one_page(page.tail_page)
                page.tail_page = -1
                page.token_count = 0
                page.hash_key = 0
                page.status = page.EMPTY
                page.del_self_from_list()
                self.page_items.add_item_to_head(page.self_index)
                return True
            page = page.get_next_item()
        return False

    def allocate_kv_pages(self, hash_keys, disk_offload_enable):
        """Allocate KV, evicting idle checkpoint pins only when necessary."""
        pages, ready = [], []
        for key in hash_keys:
            while True:
                index, is_ready = self.kv_client.allocate_one_page(
                    self.kv_client.page_items.linked_items, key, disk_offload_enable
                )
                if index is not None or not self.evict_one_checkpoint():
                    break
            if index is None:
                break
            pages.append(index)
            ready.append(is_ready)
        missing = len(hash_keys) - len(pages)
        return pages + [-1] * missing, ready + [False] * missing

    def state_views(self, page_index, tp_rank):
        page = self.cpu_kv_cache_tensor[page_index].view(-1)
        rank_bytes = self.rank_conv_bytes + self.rank_ssm_bytes
        page = page[tp_rank * rank_bytes : (tp_rank + 1) * rank_bytes]
        conv = (
            page[: self.rank_conv_bytes]
            .view(self.config.conv_state_dtype)
            .view(self.config.linear_layer_num, *self.config.get_conv_state_shape())
        )
        ssm = (
            page[self.rank_conv_bytes :]
            .view(self.config.ssm_state_dtype)
            .view(self.config.linear_layer_num, *self.config.get_ssm_state_shape())
        )
        return conv, ssm
