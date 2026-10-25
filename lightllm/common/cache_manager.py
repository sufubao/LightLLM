import torch
import torch.distributed as dist
import triton
from lightllm.server.multi_level_kv_cache import CacheTier
from lightllm.utils.custom_kernel_utis import custom_cat
from lightllm.utils.log_utils import init_logger
from lightllm.utils.multimodal_chunk_utils import clamp_mm_cache_len


logger = init_logger(__name__)


class CacheManager:
    has_state_cache = False
    state_manager = None
    linear_state_manager = None
    indexer_state_manager = None

    def __init__(self, args, max_request_num, max_sequence_length, *, device="cuda"):
        self.args = args
        self.mem_manager = None
        # 这里对最大请求数量的管理在默认上多申请了一个，主要是 index 为 max_request_num 代表
        # 的这个请求管理 id， 主要是为了兼容 DP 运行模式下，让各个 DP 能 padding 到 DP 中最大
        # 的那个batch size 进行运行，所有 padding 的请求都会使用预留的这个请求管理 id 进行处理
        # 这样让 DP 的实现更为简化一些。
        page_size = args.page_size
        max_sequence_length = triton.cdiv(max_sequence_length, page_size) * page_size
        self.req_to_token_indexs = torch.zeros(
            (max_request_num + 1, max_sequence_length), dtype=torch.int32, device=device
        )
        self.radix_cache = None
        self.cpu_cache_client = None

    def init_hold_request_indexs(self, hold_token_indexes):
        # HOLD_REQUEST_ID 对应的请求行供 DP padding、overlap microbatch 等占位请求使用。将该行
        # 按 page_size 划分后，每一页都映射到 mem_manager 额外保留的同一个物理页；这样占位请求
        # 无论访问哪一个逻辑位置，都会落到合法且不会参与正常分配的 KV cache 地址上。
        hold_row = self.req_to_token_indexs[-1]
        hold_page = torch.tensor(hold_token_indexes, dtype=hold_row.dtype, device=hold_row.device)
        assert hold_page.numel() == self.args.page_size
        hold_row.view(-1, self.args.page_size).copy_(hold_page)

    def bind_radix_cache(self, radix_cache):
        if self.radix_cache is not None and self.radix_cache is not radix_cache:
            raise ValueError("CacheManager is already bound to a different radix cache")
        self.validate_radix_cache(radix_cache)
        self.radix_cache = radix_cache

    def flush_cache(self):
        if self.radix_cache is not None:
            self.radix_cache.flush_cache()

    def merge_unreferenced_nodes(self):
        if self.radix_cache is not None:
            self.radix_cache.merge_unreferenced_nodes()

    def bind_cpu_cache_client(self, cpu_cache_client):
        if self.cpu_cache_client is not None and self.cpu_cache_client is not cpu_cache_client:
            raise ValueError("CacheManager is already bound to a different CPU cache client")
        self.cpu_cache_client = cpu_cache_client

    def get_available_token_num(self):
        cache_tokens = 0
        if self.radix_cache is not None:
            cache_tokens = self.radix_cache.get_tree_total_tokens_num() - self.radix_cache.get_refed_tokens_num()
        return self.mem_manager.allocator.can_use_mem_size + cache_tokens

    def alloc_req_kv(self, req, alloc_token_num, non_blocking=False):
        if alloc_token_num == 0:
            return None
        assert alloc_token_num > 0 and alloc_token_num % self.args.page_size == 0
        if self.radix_cache is not None:
            self.radix_cache.free_radix_cache_to_get_enough_token(alloc_token_num)
        old_end = req.hold_kv_len
        new_end = old_end + alloc_token_num
        indexes = self.mem_manager.alloc(alloc_token_num)
        # 高频调度路径允许异步写入请求索引表，其他调用方默认保持原有的同步拷贝语义。
        self.req_to_token_indexs[req.req_idx, old_end:new_end].copy_(indexes, non_blocking=non_blocking)
        req.hold_kv_len = new_end
        return indexes

    def release_request_caches(self, reqs):
        free_indexes = []
        for req in reqs:
            free_indexes.extend(self._release_request_cache(req))
            req.cur_kv_len = req.hold_kv_len = 0
            req.shm_req.shm_cur_kv_len = 0
        if free_indexes:
            self.mem_manager.free(custom_cat(free_indexes))

    def preload_prompt_cache_kv_buffer(self, prompt_cache_kv_buffer_path, token_ids):
        page_size = self.args.page_size
        intact_kv_len = len(token_ids) // page_size * page_size
        if intact_kv_len == 0:
            return

        prompt_cache_kv_buffer = torch.load(prompt_cache_kv_buffer_path, weights_only=True, map_location="cpu")
        prompt_cache_kv_buffer = {name: buffer[:, :intact_kv_len] for name, buffer in prompt_cache_kv_buffer.items()}
        intact_kv_index = self.mem_manager.alloc(intact_kv_len)
        self.mem_manager.load_index_kv_buffer(intact_kv_index, prompt_cache_kv_buffer)
        intact_token_ids = torch.tensor(token_ids[:intact_kv_len], dtype=torch.int64, device="cpu")
        self.radix_cache.insert(intact_token_ids, intact_kv_index)
        self.radix_cache.match_prefix(intact_token_ids, update_refs=True)

    def _save_promptcache_kvbuffer(self):
        """
        save prompt cache kv buffer
        这个接口是用于保存非量化的缓存prompt cache资源，是定制场景使用的接口，当前代码中不会有调用。
        其保存的 kv 会配合量化推理模式, 加载到量化推理的prompt cache中, 提升量化推理的精度。
        like paper:
        https://arxiv.org/abs/2403.01241
        """
        prompt_cache_token_id = list(self.radix_cache.root_node.children.values())[0].token_id_key
        print(f"prompt_cache_token_id : {prompt_cache_token_id}")
        index = range(len(prompt_cache_token_id))
        prompt_cache_kv_buffer = self.mem_manager.get_index_kv_buffer(index)
        torch.save(prompt_cache_kv_buffer, f"prompt_cache_rank_{dist.get_rank()}.pt")

    def log_cache_stats(self):
        if self.radix_cache is not None:
            logger.debug(
                f"free a batch state:\n"
                f"radix refed token num {self.radix_cache.get_refed_tokens_num()}\n"
                f"radix hold token num {self.radix_cache.get_tree_total_tokens_num()}\n"
                f"mem manager can alloc token num {self.mem_manager.allocator.can_use_mem_size}\n"
                f"mem manager total size {self.mem_manager.allocator.size}\n"
            )

    def bind_mem_manager(self, mem_manager):
        if self.mem_manager is not None and self.mem_manager is not mem_manager:
            raise ValueError("CacheManager is already bound to a different memory manager")
        self.mem_manager = mem_manager

    def validate_radix_cache(self, radix_cache):
        if radix_cache is not None and radix_cache.mem_manager is not self.mem_manager:
            raise ValueError("Radix cache and request cache manager must share the memory manager")

    def create_state_transfer(self):
        return None

    def restore_prefix(self, req):
        radix_cache = self.radix_cache
        enabled = not req.sampling_param.disable_prompt_cache and radix_cache is not None
        if enabled and req.get_cur_total_len() > 1 and req.cur_kv_len == 0:
            tokens = req.shm_req.shm_prompt_ids.arr[: req.get_cur_total_len()]
            key = torch.tensor(tokens, dtype=torch.int64, device="cpu")[:-1]
            node, _, indexes = radix_cache.match_prefix(key, update_refs=True, mm_input_spans=req.mm_input_spans)
            if node is not None:
                length = node.node_prefix_total_len
                self.req_to_token_indexs[req.req_idx, :length] = indexes
                req.shared_kv_node = node
                req.cur_kv_len = req.hold_kv_len = int(length)
                assert req.hold_kv_len % self.args.page_size == 0
                req.shm_req.prompt_cache_len = req.cur_kv_len
        req.shm_req.shm_cur_kv_len = req.cur_kv_len

    def save_state_cache(self, b_req_idx, reqs):
        """Snapshot after forward, before advancing req.cur_kv_len from the chunk start."""
        return

    def get_cacheable_prefix_len(self, req):
        return req.cur_kv_len

    def _release_request_cache(self, req) -> list[torch.Tensor]:
        """Publish reusable cache and release unpublished checkpoints and held radix refs.

        Returns borrowed request-index slices for the caller to batch-free. The caller
        resets request lengths and must consume these slices before reusing the request
        row. Call only after pending CPU offload reads have completed. Published KV and
        checkpoints belong to radix; returned KV indexes still need allocator release.
        """
        radix_cache, args = self.radix_cache, self.args
        publish_cache = radix_cache is not None and CacheTier.GPU in req.cache_tiers
        indexes = self.req_to_token_indexs[req.req_idx]
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

    def publish_request_prefix(self, master_req):
        key = master_req.get_input_token_ids()[0 : master_req.cur_kv_len]
        key = torch.tensor(key, dtype=torch.int64, device="cpu")
        value = self.req_to_token_indexs[master_req.req_idx][: master_req.cur_kv_len].detach().cpu()
        prefix_len, new_shared_kv_node = self.radix_cache.insert(key, value)
        old_prefix_len = 0 if master_req.shared_kv_node is None else master_req.shared_kv_node.node_prefix_total_len
        assert old_prefix_len <= master_req.cur_kv_len
        self.mem_manager.free(self.req_to_token_indexs[master_req.req_idx][old_prefix_len:prefix_len])

        # 将原有共享节点替换为新共享节点，新共享节点对应的长度为当前的cur_kv_len
        self.radix_cache.dec_node_ref_counter(master_req.shared_kv_node)
        self.radix_cache.add_node_ref_counter(new_shared_kv_node)
        master_req.shared_kv_node = new_shared_kv_node
        assert (
            new_shared_kv_node.node_prefix_total_len == master_req.cur_kv_len
        ), f"shared len: {new_shared_kv_node.node_prefix_total_len} cur_kv_len {master_req.cur_kv_len}"

        share_node, kv_len, value = self.radix_cache.match_prefix(key, update_refs=False)
        assert share_node == new_shared_kv_node and kv_len == master_req.cur_kv_len
        self.req_to_token_indexs[master_req.req_idx][0 : master_req.cur_kv_len] = value
        return

    def share_request_prefix(self, master_req, slave_req):
        self.radix_cache.dec_node_ref_counter(slave_req.shared_kv_node)
        self.radix_cache.add_node_ref_counter(master_req.shared_kv_node)

        kv_len = master_req.cur_kv_len

        self.req_to_token_indexs[slave_req.req_idx][0:kv_len] = self.req_to_token_indexs[master_req.req_idx][0:kv_len]
        # torch.cuda.current_stream().synchronize()
        slave_req.shared_kv_node = master_req.shared_kv_node
        slave_req.cur_kv_len = kv_len
        # slave 在 prefill 阶段没有独立分配 KV，共享完成后需要同步其已持有的 KV 长度。
        slave_req.hold_kv_len = kv_len

    def restore_peer_state(self, req_idx, restored_len, peer_state, peer_req_idx):
        return

    def get_cpu_cache_match_page_num(self, page_indexes, page_lens):
        return len(page_indexes)

    def load_cpu_cache_to_gpu(self, mem_indexes, page_indexes, req, page_indexes_cpu=None) -> None:
        """Enqueue KV and any live request-state restoration on the current stream."""
        self.mem_manager.operator.load_cpu_cache_to_gpu(mem_indexes, page_indexes, self.cpu_cache_client, req)

    def offload_gpu_kv_to_cpu_cache(
        self,
        mem_indexes,
        page_indexes,
        page_readies,
        req,
        page_indexes_cpu=None,
        page_readies_cpu=None,
    ):
        return self.mem_manager.operator.offload_gpu_kv_to_cpu_cache(
            mem_indexes, page_indexes, page_readies, self.cpu_cache_client, req
        )


class HybridCacheManager(CacheManager):
    """Coordinate KV prefix reuse and hybrid attention state checkpoints."""

    def __init__(
        self,
        args,
        max_request_num,
        max_sequence_length,
        linear_state_manager=None,
        indexer_state_manager=None,
        *,
        state_manager=None,
        device="cuda",
    ):
        super().__init__(args, max_request_num, max_sequence_length, device=device)
        self.state_manager = state_manager if state_manager is not None else linear_state_manager
        self.linear_state_manager = linear_state_manager
        self.indexer_state_manager = indexer_state_manager
        self.small_page_buffers = None
        self.big_page_buffers = None

    @property
    def has_state_cache(self):
        return self.state_manager is not None

    def bind_mem_manager(self, mem_manager):
        if self.mem_manager is mem_manager:
            return
        if self.mem_manager is not None:
            raise ValueError("HybridCacheManager is already bound to a different memory manager")
        if self.has_state_cache:
            args = self.args
            big_page_tokens = args.linear_att_page_block_num * args.linear_att_hash_page_size
            # The last two slots are reserved for CPU-cache tail staging, never allocated to requests.
            self.big_page_buffers = self.state_manager.create_checkpoint_pool(
                triton.cdiv(mem_manager.size, big_page_tokens) + 2,
                keep_num=2,
            )
            self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = self.big_page_buffers.size - 2
            self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = self.big_page_buffers.size - 1
            self.cpu_cache_transfer = self.state_manager.create_cpu_transfer(mem_manager, self.big_page_buffers)
        self.mem_manager = mem_manager

    def create_state_transfer(self):
        if not self.has_state_cache:
            return None
        return self.state_manager.create_state_transfer(self.indexer_state_manager)

    def validate_radix_cache(self, radix_cache):
        super().validate_radix_cache(radix_cache)
        if radix_cache is not None and self.has_state_cache:
            if (
                radix_cache.big_page_buffers is not self.big_page_buffers
                or radix_cache.small_page_buffers is not self.small_page_buffers
            ):
                raise ValueError("Radix cache and request cache manager must share checkpoint pools")

    def create_small_page_cache_manager(self, size):
        if self.small_page_buffers is not None:
            if self.small_page_buffers.size != size:
                raise ValueError("Cannot replace an initialized checkpoint pool")
            return self.small_page_buffers
        self.small_page_buffers = self.state_manager.create_checkpoint_pool(size)
        return self.small_page_buffers

    def reset(self, req_idx):
        if self.state_manager is not None:
            self.state_manager.reset(req_idx)
        if self.indexer_state_manager is not None:
            self.indexer_state_manager.reset(req_idx)

    def restore_state(self, req_idx, state_cache_manager, buffer_idx, restored_len):
        assert state_cache_manager.has_checkpoint(buffer_idx, restored_len)
        self.state_manager.restore_checkpoint(req_idx, restored_len, state_cache_manager, buffer_idx)
        if self.indexer_state_manager is not None:
            self.indexer_state_manager.reset(req_idx)

    def restore_peer_state(self, req_idx, restored_len, peer_state, peer_req_idx):
        self.state_manager.restore_peer_state(req_idx, restored_len, peer_state, peer_req_idx)
        if self.indexer_state_manager is not None:
            self.indexer_state_manager.state.buffer[:, req_idx].copy_(
                peer_state.req_to_indexer_tail.buffer[:, peer_req_idx], non_blocking=True
            )

    def save_state(self, req_idx, buffer_idx, state_cache_manager, checkpoint_len):
        self.state_manager.save_state(req_idx, buffer_idx, state_cache_manager, checkpoint_len)
        state_cache_manager.checkpoint_lens[buffer_idx] = checkpoint_len

    def save_big_page_states(self, b_req_idx, req_indexes, buffer_indexes, checkpoint_lens):
        self.state_manager.save_checkpoints(
            b_req_idx, req_indexes, buffer_indexes, self.big_page_buffers, checkpoint_lens
        )
        for slot, length in zip(buffer_indexes, checkpoint_lens):
            self.big_page_buffers.checkpoint_lens[slot] = length

    def _restore_shared_prefix(
        self, req_idx, token_indexes, cached_indexes, restored_len, checkpoint_pool, checkpoint_slot
    ):
        token_indexes[:restored_len] = cached_indexes[:restored_len]
        self.restore_state(req_idx, checkpoint_pool, checkpoint_slot, restored_len)

    def restore_prefix(self, req):
        radix_cache = self.radix_cache
        enable_prompt_cache = (not req.sampling_param.disable_prompt_cache) and radix_cache is not None
        block_hashs = req.shm_req.hybrid_token_hash_list.get_all()
        hash_page_size = self.args.linear_att_hash_page_size
        match_tokens = max(0, min(len(block_hashs) * hash_page_size, req.get_cur_total_len() - 1))
        match_tokens = (match_tokens // hash_page_size) * hash_page_size
        block_hashs = block_hashs[: match_tokens // hash_page_size]
        assert len(block_hashs) == req.shm_req.hybrid_token_hash_list.size
        big_page_token_num = hash_page_size * self.args.linear_att_page_block_num
        big_page_is_disable = big_page_token_num > self.args.max_req_total_len
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
                token_indexes = self.req_to_token_indexs[req.req_idx]
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
                        assert need_tokens % self.args.page_size == 0
                        radix_cache.free_radix_cache_to_get_enough_token(need_token_num=need_tokens)
                        tail_mems = self.mem_manager.alloc(need_size=need_tokens)
                        token_indexes[:big_page_len] = value_tensor[:big_page_len]
                        token_indexes[big_page_len:matched_len] = tail_mems
                        self.mem_manager.operator.copy_mem_to_mem(value_tensor[big_page_len:matched_len], tail_mems)
                        req.prompt_selected_logprobs.copy_capture_slots_if_needed(
                            source_indexes=value_tensor[big_page_len:matched_len], destination_indexes=tail_mems
                        )
                        self.restore_state(
                            req.req_idx, self.small_page_buffers, share_node.small_page_buffer_idx, matched_len
                        )
                        restored_kv_len = matched_len
                        retained_node = radix_cache.deref_to_first_big_page_node(node=share_node, require_state=False)
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
        assert req.hold_kv_len % self.args.page_size == 0
        if req.cur_kv_len == 0:
            self.reset(req.req_idx)

    def save_state_cache(self, b_req_idx, reqs):
        """Snapshot after forward, before advancing req.cur_kv_len from the chunk start."""
        radix_cache, args = self.radix_cache, self.args
        if radix_cache is None:
            return
        # Request-state snapshot at a big-page boundary.
        big_page_token_num = args.linear_att_hash_page_size * args.linear_att_page_block_num
        big_page_buffer_ids, checkpoint_lens, req_rows = [], [], []
        for row, req in enumerate(reqs):
            chunk_end = req.get_chuncked_input_token_len()
            first_boundary = (req.cur_kv_len // big_page_token_num + 1) * big_page_token_num
            for length in range(first_boundary, min(chunk_end, req.hybrid_cache_len) + 1, big_page_token_num):
                slot = self.big_page_buffers.alloc_one_state_cache()
                assert slot is not None and length not in req.hybrid_len_to_big_page_id
                req.hybrid_len_to_big_page_id[length] = slot
                if self.state_manager.capture_intermediate_checkpoints or length == chunk_end:
                    big_page_buffer_ids.append(slot)
                    checkpoint_lens.append(length)
                    req_rows.append(row)
        if big_page_buffer_ids:
            selected_rows = torch.tensor(req_rows, dtype=torch.int64, device=b_req_idx.device)
            self.save_big_page_states(
                b_req_idx=b_req_idx[selected_rows],
                req_indexes=[reqs[row].req_idx for row in req_rows],
                buffer_indexes=big_page_buffer_ids,
                checkpoint_lens=checkpoint_lens,
            )

        assert (
            not args.disable_chunked_prefill or self.state_manager.capture_intermediate_checkpoints
        ), "chunked prefill must be enabled when only chunk-end state can be captured"

        # Request-state snapshot at the final small-page boundary.
        for req in reqs:
            # 判断本次prefill 完以后 kv 的长度是否到达 hybrid checkpoint 的存储边界。
            if req.cur_kv_len < req.hybrid_cache_len <= req.get_chuncked_input_token_len() and (
                self.state_manager.capture_intermediate_checkpoints
                or req.get_chuncked_input_token_len() == req.hybrid_cache_len
            ):
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
                            checkpoint_len=req.hybrid_cache_len,
                        )
        return

    def get_cacheable_prefix_len(self, req):
        """Peer runtime restores permit continuation, but do not supply historical checkpoints."""
        args = self.args
        page_size = args.linear_att_hash_page_size * args.linear_att_page_block_num
        shared_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
        end = min(req.cur_kv_len, req.hybrid_cache_len)
        cacheable_len = 0
        node = req.shared_kv_node
        while node is not None and node.node_prefix_total_len:
            length = node.node_prefix_total_len
            pool, slot = (
                (self.big_page_buffers, node.big_page_buffer_idx)
                if length % page_size == 0
                else (self.small_page_buffers, node.small_page_buffer_idx)
            )
            if pool.has_checkpoint(slot, length):
                cacheable_len = length
                break
            node = node.parent
        for boundary in range((shared_len // page_size + 1) * page_size, end + 1, page_size):
            if boundary not in req.hybrid_len_to_big_page_id:
                return cacheable_len
            slot = req.hybrid_len_to_big_page_id[boundary]
            if self.big_page_buffers.has_checkpoint(slot, boundary):
                cacheable_len = boundary
        if self.small_page_buffers.has_checkpoint(req.tail_small_page_buffer_id, end):
            cacheable_len = end
        return cacheable_len

    def _release_request_cache(self, req) -> list[torch.Tensor]:
        """Publish reusable cache and release unpublished checkpoints and held radix refs.

        Returns borrowed request-index slices for the caller to batch-free. The caller
        resets request lengths and must consume these slices before reusing the request
        row. Call only after pending CPU offload reads have completed. Published KV and
        checkpoints belong to radix; returned KV indexes still need allocator release.
        """
        radix_cache, args = self.radix_cache, self.args
        publish_cache = radix_cache is not None and CacheTier.GPU in req.cache_tiers
        token_indexes = self.req_to_token_indexs[req.req_idx]
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
            publish_len = min(publish_len, self.get_cacheable_prefix_len(req))
            if publish_len <= shared_len:
                publish_len = None

        if publish_len is not None:
            discarded = [length for length in req.hybrid_len_to_big_page_id if length > publish_len]
            if discarded:
                self.big_page_buffers.free_state_cache(
                    [req.hybrid_len_to_big_page_id.pop(length) for length in discarded]
                )
            if req.tail_small_page_buffer_id is not None and publish_len < req.hybrid_cache_len:
                self.small_page_buffers.free_state_cache([req.tail_small_page_buffer_id])
                req.tail_small_page_buffer_id = None
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

    def get_cpu_cache_match_page_num(self, page_indexes, page_lens):
        lengths = self.cpu_cache_transfer.get_checkpoint_lens(page_indexes, self.cpu_cache_client)
        for index in range(len(lengths) - 1, -1, -1):
            if lengths[index] == page_lens[index]:
                return index + 1
        return 0

    def load_cpu_cache_to_gpu(self, mem_indexes, page_indexes, req, page_indexes_cpu=None) -> None:
        """Enqueue KV and any live request-state restoration on the current stream."""
        cpu_cache_client = self.cpu_cache_client
        args = self.args
        page_size = args.cpu_cache_token_page_size
        assert page_size == args.linear_att_hash_page_size * args.linear_att_page_block_num
        assert len(mem_indexes) % args.linear_att_hash_page_size == 0
        assert triton.cdiv(len(mem_indexes), page_size) == len(page_indexes)
        assert clamp_mm_cache_len(req.cur_kv_len, req.mm_input_spans) == req.cur_kv_len
        if page_indexes_cpu is None:
            page_indexes_cpu = page_indexes.cpu().tolist()
        checkpoint_lens = self.cpu_cache_transfer.get_checkpoint_lens(page_indexes_cpu, cpu_cache_client)
        assert checkpoint_lens[-1] == req.cur_kv_len
        page_count = len(mem_indexes) // page_size
        end = req.cur_kv_len // page_size * page_size
        slots = self.big_page_buffers.alloc_state_cache(page_count)
        assert slots is not None, "Insufficient checkpoint slots for CPU cache restoration"
        first_end = end - (page_count - 1) * page_size
        for index, slot in enumerate(slots):
            length = first_end + index * page_size
            req.hybrid_len_to_big_page_id[length] = slot
            if checkpoint_lens[index] == length:
                self.big_page_buffers.checkpoint_lens[slot] = length
        if len(mem_indexes) % page_size:
            slots.append(self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID)
            self.big_page_buffers.checkpoint_lens[slots[-1]] = req.cur_kv_len
        self.cpu_cache_transfer.load(mem_indexes, page_indexes, cpu_cache_client, slots)
        self.restore_state(req.req_idx, self.big_page_buffers, slots[-1], req.cur_kv_len)

    def offload_gpu_kv_to_cpu_cache(
        self,
        mem_indexes,
        page_indexes,
        page_readies,
        req,
        page_indexes_cpu=None,
        page_readies_cpu=None,
    ):
        cpu_cache_client, radix_cache = self.cpu_cache_client, self.radix_cache
        page_size = self.args.cpu_cache_token_page_size
        end = len(mem_indexes) // page_size * page_size
        slots = [] if radix_cache is None else radix_cache.get_big_page_ids_by_node(req.shared_kv_node)
        slots = slots[: end // page_size]
        for length in range((len(slots) + 1) * page_size, end + 1, page_size):
            slots.append(req.hybrid_len_to_big_page_id[length])
        tail_state = None
        if len(mem_indexes) % page_size:
            assert req.tail_small_page_buffer_id is not None
            tail_state = self.small_page_buffers.get_state_cache(req.tail_small_page_buffer_id)
            slots.append(self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID)
            self.big_page_buffers.checkpoint_lens[slots[-1]] = self.small_page_buffers.checkpoint_lens[
                req.tail_small_page_buffer_id
            ]
        self.cpu_cache_transfer.offload(
            mem_indexes,
            page_indexes,
            page_readies,
            cpu_cache_client,
            slots,
            tail_state,
            page_indexes_cpu=page_indexes_cpu,
            page_readies_cpu=page_readies_cpu,
        )
