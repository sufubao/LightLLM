import torch
import triton
from lightllm.utils.envs_utils import get_env_start_args

from lightllm.common.state_cache_manager import LinearAttCacheManager


class CacheCoordinator:
    has_state_cache = False
    linear_state_manager = None
    indexer_state_manager = None

    def __init__(self):
        self.mem_manager = None

    def bind_mem_manager(self, mem_manager):
        if self.mem_manager is not None and self.mem_manager is not mem_manager:
            raise ValueError("CacheCoordinator is already bound to a different memory manager")
        self.mem_manager = mem_manager

    def validate_radix_cache(self, radix_cache):
        if radix_cache is not None and radix_cache.mem_manager is not self.mem_manager:
            raise ValueError("Radix cache and coordinator must share the memory manager")

    def create_state_transfer(self):
        return None

    def restore_prefix(self, req, req_manager, radix_cache):
        enabled = not req.sampling_param.disable_prompt_cache and radix_cache is not None
        if enabled and req.get_cur_total_len() > 1 and req.cur_kv_len == 0:
            tokens = req.shm_req.shm_prompt_ids.arr[: req.get_cur_total_len()]
            key = torch.tensor(tokens, dtype=torch.int64, device="cpu")[:-1]
            node, _, indexes = radix_cache.match_prefix(key, update_refs=True, mm_input_spans=req.mm_input_spans)
            if node is not None:
                length = node.node_prefix_total_len
                req_manager.req_to_token_indexs[req.req_idx, :length] = indexes
                req.shared_kv_node = node
                req.cur_kv_len = req.hold_kv_len = int(length)
                assert req.hold_kv_len % req.args.page_size == 0
                req.shm_req.prompt_cache_len = req.cur_kv_len
        req.shm_req.shm_cur_kv_len = req.cur_kv_len

    def save_state_cache(self, b_req_idx, reqs, radix_cache, args):
        """Snapshot after forward, before advancing req.cur_kv_len from the chunk start."""
        return

    def release_request_cache(self, req, req_manager, radix_cache, args, publish_cache) -> list[torch.Tensor]:
        """Publish reusable cache and release unpublished checkpoints and held radix refs.

        Returns borrowed request-index slices for the caller to batch-free. The caller
        resets request lengths and must consume these slices before reusing the request
        row. Call only after pending CPU offload reads have completed. Published KV and
        checkpoints belong to radix; returned KV indexes still need allocator release.
        """
        indexes = req_manager.req_to_token_indexs[req.req_idx]
        shared_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
        if publish_cache and radix_cache is not None:
            cache_len = req.cur_kv_len // args.page_size * args.page_size
            key = torch.tensor(req.get_input_token_ids()[:cache_len], dtype=torch.int64, device="cpu")
            prefix_len, _ = radix_cache.insert(key, indexes[:cache_len].detach().cpu())
            assert shared_len <= prefix_len
            free_indexes = [indexes[cache_len : req.hold_kv_len], indexes[shared_len:prefix_len]]
        else:
            free_indexes = [indexes[shared_len : req.hold_kv_len]]
        if req.shared_kv_node is not None:
            radix_cache.dec_node_ref_counter(req.shared_kv_node)
            req.shared_kv_node = None
        return free_indexes

    def load_cpu_cache_to_gpu(self, mem_indexes, page_indexes, cpu_cache_client, req) -> None:
        """Enqueue KV and any live request-state restoration on the current stream."""
        self.mem_manager.operator.load_cpu_cache_to_gpu(mem_indexes, page_indexes, cpu_cache_client, req)

    def offload_gpu_kv_to_cpu_cache(
        self, mem_indexes, page_indexes, page_readies, cpu_cache_client, req, radix_cache=None
    ):
        return self.mem_manager.operator.offload_gpu_kv_to_cpu_cache(
            mem_indexes, page_indexes, page_readies, cpu_cache_client, req
        )


class HybridCacheCoordinator(CacheCoordinator):
    """Coordinate KV prefix reuse and hybrid attention state checkpoints."""

    def __init__(self, linear_state_manager=None, indexer_state_manager=None):
        self.linear_state_manager = linear_state_manager
        self.indexer_state_manager = indexer_state_manager
        self.mem_manager = None
        self.small_page_buffers = None
        self.big_page_buffers = None

    @property
    def has_state_cache(self):
        return self.linear_state_manager is not None

    def bind_mem_manager(self, mem_manager):
        if self.mem_manager is mem_manager:
            return
        if self.mem_manager is not None:
            raise ValueError("HybridCacheCoordinator is already bound to a different memory manager")
        if self.has_state_cache:
            args = get_env_start_args()
            big_page_tokens = args.linear_att_page_block_num * args.linear_att_hash_page_size
            # The last two slots are reserved for CPU-cache tail staging, never allocated to requests.
            self.big_page_buffers = LinearAttCacheManager(
                triton.cdiv(mem_manager.size, big_page_tokens) + 2,
                self.linear_state_manager.linear_config,
                keep_num=2,
            )
            self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = self.big_page_buffers.size - 2
            self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = self.big_page_buffers.size - 1
            from lightllm.common.hybrid_cache_transfer import HybridCpuCacheTransfer

            self.cpu_cache_transfer = HybridCpuCacheTransfer(
                mem_manager.kv_buffer, self.linear_state_manager.linear_config, self.big_page_buffers
            )
        self.mem_manager = mem_manager

    def create_state_transfer(self):
        if not self.has_state_cache:
            return None
        from lightllm.common.linear_state_transfer import LinearAttentionStateTransfer

        return LinearAttentionStateTransfer(self.linear_state_manager, self.indexer_state_manager)

    def validate_radix_cache(self, radix_cache):
        super().validate_radix_cache(radix_cache)
        if radix_cache is not None and self.has_state_cache:
            if (
                radix_cache.big_page_buffers is not self.big_page_buffers
                or radix_cache.small_page_buffers is not self.small_page_buffers
            ):
                raise ValueError("Radix cache and coordinator must share checkpoint pools")

    def create_small_page_cache_manager(self, size):
        if self.small_page_buffers is not None:
            if self.small_page_buffers.size != size:
                raise ValueError("Cannot replace an initialized checkpoint pool")
            return self.small_page_buffers
        self.small_page_buffers = LinearAttCacheManager(size, self.linear_state_manager.linear_config)
        return self.small_page_buffers

    def reset(self, req_idx):
        if self.linear_state_manager is not None:
            self.linear_state_manager.reset(req_idx)
        if self.indexer_state_manager is not None:
            self.indexer_state_manager.reset(req_idx)

    def restore_state(self, req_idx, state_cache_manager, buffer_idx):
        self.linear_state_manager.restore(req_idx, *state_cache_manager.get_state_cache(buffer_idx))
        if self.indexer_state_manager is not None:
            self.indexer_state_manager.reset(req_idx)

    def save_state(self, req_idx, buffer_idx, state_cache_manager):
        self.linear_state_manager.copy_state_to(req_idx, *state_cache_manager.get_state_cache(buffer_idx))

    def save_big_page_states(self, b_req_idx, req_indexes, buffer_indexes):
        self.linear_state_manager.save_checkpoints(b_req_idx, req_indexes, buffer_indexes, self.big_page_buffers)

    def _restore_shared_prefix(
        self, req_idx, token_indexes, cached_indexes, restored_len, checkpoint_pool, checkpoint_slot
    ):
        token_indexes[:restored_len] = cached_indexes[:restored_len]
        self.restore_state(req_idx, checkpoint_pool, checkpoint_slot)

    def restore_prefix(self, req, req_manager, radix_cache):
        enable_prompt_cache = (not req.sampling_param.disable_prompt_cache) and radix_cache is not None
        block_hashs = req.shm_req.hybrid_token_hash_list.get_all()
        hash_page_size = req.args.linear_att_hash_page_size
        match_tokens = max(0, min(len(block_hashs) * hash_page_size, req.get_cur_total_len() - 1))
        match_tokens = (match_tokens // hash_page_size) * hash_page_size
        block_hashs = block_hashs[: match_tokens // hash_page_size]
        assert len(block_hashs) == req.shm_req.hybrid_token_hash_list.size
        big_page_token_num = hash_page_size * req.args.linear_att_page_block_num
        big_page_is_disable = big_page_token_num > req.args.max_req_total_len
        restored_kv_len = req.cur_kv_len
        retained_node = req.shared_kv_node
        if enable_prompt_cache and match_tokens > 1 and block_hashs and req.cur_kv_len == 0:
            input_token_ids = req.shm_req.shm_prompt_ids.arr[: req.get_cur_total_len()]
            key = torch.tensor(input_token_ids[:match_tokens], dtype=torch.int64, device="cpu")
            share_node, _, value_tensor = radix_cache.match_prefix(
                key, block_hashs=block_hashs, update_refs=True, mm_input_spans=req.mm_input_spans
            )
            if share_node is not None:
                assert req.tail_small_page_buffer_id is None
                token_indexes = req_manager.req_to_token_indexs[req.req_idx]
                if not share_node.is_big_page_node() and not big_page_is_disable:
                    matched_len = share_node.node_prefix_total_len
                    big_page_len = (matched_len // big_page_token_num) * big_page_token_num
                    need_tokens = matched_len - big_page_len
                    available_tokens = (
                        self.mem_manager.allocator.can_use_mem_size
                        + radix_cache.get_tree_total_tokens_num()
                        - radix_cache.get_refed_tokens_num()
                    )
                    if available_tokens > need_tokens:
                        assert need_tokens % req.args.page_size == 0
                        radix_cache.free_radix_cache_to_get_enough_token(need_token_num=need_tokens)
                        tail_mems = self.mem_manager.alloc(need_size=need_tokens)
                        token_indexes[:big_page_len] = value_tensor[:big_page_len]
                        token_indexes[big_page_len:matched_len] = tail_mems
                        self.mem_manager.operator.copy_mem_to_mem(value_tensor[big_page_len:matched_len], tail_mems)
                        req.prompt_selected_logprobs.copy_capture_slots_if_needed(
                            source_indexes=value_tensor[big_page_len:matched_len], destination_indexes=tail_mems
                        )
                        self.restore_state(req.req_idx, self.small_page_buffers, share_node.small_page_buffer_idx)
                        restored_kv_len = matched_len
                        retained_node = radix_cache.deref_to_first_big_page_node(node=share_node)
                        share_node = None
                    else:
                        share_node = radix_cache.deref_to_first_big_page_node(
                            node=share_node, mm_input_spans=req.mm_input_spans
                        )
                if share_node is not None:
                    restored_kv_len = share_node.node_prefix_total_len
                    retained_node = share_node
                    if share_node.is_big_page_node():
                        pool, slot = self.big_page_buffers, share_node.big_page_buffer_idx
                    else:
                        pool, slot = self.small_page_buffers, share_node.small_page_buffer_idx
                    self._restore_shared_prefix(req.req_idx, token_indexes, value_tensor, restored_kv_len, pool, slot)
                if restored_kv_len > 0:
                    req.shm_req.prompt_cache_len = int(restored_kv_len)

        req.shared_kv_node = retained_node
        req.cur_kv_len = int(restored_kv_len)
        req.shm_req.shm_cur_kv_len = req.cur_kv_len
        req.hold_kv_len = req.cur_kv_len
        assert req.hold_kv_len % req.args.page_size == 0
        if req.cur_kv_len == 0:
            self.reset(req.req_idx)

    def save_state_cache(self, b_req_idx, reqs, radix_cache, args):
        """Snapshot after forward, before advancing req.cur_kv_len from the chunk start."""
        if radix_cache is None:
            return
        # Request-state snapshot at a big-page boundary.
        big_page_token_num = args.linear_att_hash_page_size * args.linear_att_page_block_num
        big_page_buffer_ids = []
        for req in reqs:
            # 保存时尚未更新 req.cur_kv_len；它是本轮 chunk 起点，cur_input_len 是终点。
            cur_input_len = req.get_chuncked_input_token_len()
            first_page_end = (req.cur_kv_len // big_page_token_num + 1) * big_page_token_num
            state_buffer_id = -1
            last_page_end = min(cur_input_len, req.hybrid_cache_len)
            # 完整图片可能跨过多个大页结束点，例如页长 256、chunk [128, 768) 覆盖 256/512/768。
            # 这些页的 KV 都有效，仍需为 radix 插入和 CPU offload 分配对应槽位。
            for page_end in range(first_page_end, last_page_end + 1, big_page_token_num):
                big_page_id = self.big_page_buffers.alloc_one_state_cache()
                assert big_page_id is not None
                assert page_end not in req.hybrid_len_to_big_page_id
                req.hybrid_len_to_big_page_id[page_end] = big_page_id
                # 当前运行态只对应 chunk 终点；图片内部页尾不保存 state，匹配时也不能在此恢复。
                if page_end == cur_input_len:
                    state_buffer_id = big_page_id
            big_page_buffer_ids.append(state_buffer_id)

        assert len(b_req_idx) == len(big_page_buffer_ids)
        if any(buffer_id != -1 for buffer_id in big_page_buffer_ids):
            self.save_big_page_states(
                b_req_idx=b_req_idx,
                req_indexes=[req.req_idx for req in reqs],
                buffer_indexes=big_page_buffer_ids,
            )

        assert not args.disable_chunked_prefill, "chunked prefill must be enabled for hybrid attention models"

        # Request-state snapshot at the final small-page boundary.
        for req in reqs:
            # 判断本次prefill 完以后 kv 的长度是否到达 hybrid checkpoint 的存储边界。
            if req.get_chuncked_input_token_len() == req.hybrid_cache_len:
                assert req.tail_small_page_buffer_id is None
                if req.hybrid_cache_len % big_page_token_num != 0:
                    radix_cache.free_one_small_page_buffer()
                    req.tail_small_page_buffer_id = self.small_page_buffers.alloc_one_state_cache()
                    if req.tail_small_page_buffer_id is not None:
                        dst_buffer_idx = req.tail_small_page_buffer_id
                        self.save_state(
                            req_idx=req.req_idx,
                            buffer_idx=dst_buffer_idx,
                            state_cache_manager=self.small_page_buffers,
                        )
        return

    def release_request_cache(self, req, req_manager, radix_cache, args, publish_cache) -> list[torch.Tensor]:
        """Publish reusable cache and release unpublished checkpoints and held radix refs.

        Returns borrowed request-index slices for the caller to batch-free. The caller
        resets request lengths and must consume these slices before reusing the request
        row. Call only after pending CPU offload reads have completed. Published KV and
        checkpoints belong to radix; returned KV indexes still need allocator release.
        """
        token_indexes = req_manager.req_to_token_indexs[req.req_idx]
        shared_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
        publish_len = None
        if publish_cache and radix_cache is not None:
            hash_page_size = args.linear_att_hash_page_size
            big_page_tokens = hash_page_size * args.linear_att_page_block_num
            last_big_page = req.hybrid_cache_len // big_page_tokens * big_page_tokens
            assert req.hybrid_cache_len >= shared_len
            if req.tail_small_page_buffer_id is not None:
                assert req.hybrid_cache_len <= req.cur_kv_len
                assert req.hybrid_cache_len % big_page_tokens != 0
                publish_len = req.hybrid_cache_len
            elif shared_len < last_big_page <= req.cur_kv_len:
                publish_len = last_big_page
            else:
                assert shared_len <= req.cur_kv_len

        if publish_len is not None:
            page_num = publish_len // hash_page_size
            block_hashs = req.shm_req.hybrid_token_hash_list.get_all()[:page_num]
            state_idxs = [None] * page_num
            if req.tail_small_page_buffer_id is not None:
                state_idxs[-1] = req.tail_small_page_buffer_id
            key = torch.tensor(req.get_input_token_ids()[:publish_len], dtype=torch.int64, device="cpu")
            value = token_indexes[:publish_len].detach().cpu()
            prefix_len, _ = radix_cache.insert(
                key,
                value,
                block_hashs=block_hashs,
                block_state_idxs=state_idxs,
                len_to_big_page_id=req.hybrid_len_to_big_page_id,
            )
            assert not req.hybrid_len_to_big_page_id
            req.tail_small_page_buffer_id = None
            assert shared_len <= prefix_len
            free_indexes = [token_indexes[publish_len : req.hold_kv_len], token_indexes[shared_len:prefix_len]]
        else:
            free_indexes = [token_indexes[shared_len : req.hold_kv_len]]
            if req.tail_small_page_buffer_id is not None:
                self.small_page_buffers.free_state_cache([req.tail_small_page_buffer_id])
                req.tail_small_page_buffer_id = None
            if req.hybrid_len_to_big_page_id:
                self.big_page_buffers.free_state_cache(list(req.hybrid_len_to_big_page_id.values()))
                req.hybrid_len_to_big_page_id.clear()

        if req.shared_kv_node is not None:
            radix_cache.dec_node_ref_counter(req.shared_kv_node)
            req.shared_kv_node = None
        return free_indexes

    def load_cpu_cache_to_gpu(self, mem_indexes, page_indexes, cpu_cache_client, req) -> None:
        """Enqueue KV and any live request-state restoration on the current stream."""
        args = get_env_start_args()
        page_size = args.cpu_cache_token_page_size
        assert page_size == args.linear_att_hash_page_size * args.linear_att_page_block_num
        assert len(mem_indexes) % args.linear_att_hash_page_size == 0
        assert triton.cdiv(len(mem_indexes), page_size) == len(page_indexes)
        page_count = len(mem_indexes) // page_size
        end = req.cur_kv_len // page_size * page_size
        slots = []
        for _ in range(page_count):
            slot = self.big_page_buffers.alloc_one_state_cache()
            assert slot is not None
            req.hybrid_len_to_big_page_id[end] = slot
            slots.append(slot)
            end -= page_size
        slots.reverse()
        if len(mem_indexes) % page_size:
            slots.append(self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID)
        self.cpu_cache_transfer.load(mem_indexes, page_indexes, cpu_cache_client, slots)
        self.restore_state(req.req_idx, self.big_page_buffers, slots[-1])

    def offload_gpu_kv_to_cpu_cache(
        self, mem_indexes, page_indexes, page_readies, cpu_cache_client, req, radix_cache=None
    ):
        page_size = get_env_start_args().cpu_cache_token_page_size
        slots = radix_cache.get_big_page_ids_by_node(req.shared_kv_node)
        end = len(mem_indexes) // page_size * page_size
        for length in range((len(slots) + 1) * page_size, end + 1, page_size):
            slots.append(req.hybrid_len_to_big_page_id[length])
        tail_state = None
        if len(mem_indexes) % page_size:
            assert req.tail_small_page_buffer_id is not None
            tail_state = self.small_page_buffers.get_state_cache(req.tail_small_page_buffer_id)
            slots.append(self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID)
        self.cpu_cache_transfer.offload(mem_indexes, page_indexes, page_readies, cpu_cache_client, slots, tail_state)
