import threading
import torch.distributed as dist
import torch
import dataclasses
import bisect
from functools import lru_cache
from typing import Optional, List, Deque
from collections import deque
from lightllm.server.multi_level_kv_cache import CacheTier
from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
from lightllm.utils.config_utils import is_linear_att_mixed_model
from lightllm.utils.envs_utils import get_env_start_args
from ..infer_batch import InferReq
from lightllm.utils.dist_utils import create_new_group_for_current_dp
from lightllm.common.basemodel.triton_kernel.kv_cache_offload import offload_gpu_kv_to_cpu, load_cpu_kv_to_gpu
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MultiLevelKvCacheModule(object):
    def __init__(self, backend):
        self.args = get_env_start_args()
        from .base_backend import ModeBackend

        self.backend: ModeBackend = backend
        self.gloo_group = create_new_group_for_current_dp("gloo")
        self.filter_group = create_new_group_for_current_dp("gloo")
        self.init_sync_group = create_new_group_for_current_dp("nccl")
        dist.barrier(group=self.init_sync_group)
        self.offload_sync_group = create_new_group_for_current_dp("nccl")
        dist.barrier(group=self.offload_sync_group)
        self.offload_sync_tensor = torch.empty((1,), dtype=torch.int32, device="cuda")

        self.page_index_buffer = torch.empty((1024 * 1024 * 4,), dtype=torch.int32, device="cuda")
        self.page_ready_buffer = torch.empty((1024 * 1024 * 4,), dtype=torch.bool, device="cuda")

        self.cpu_cache_handle_queue: Deque[TransTask] = deque()
        self.cpu_cache_client = CpuKvCacheClient(only_create_meta_data=False, init_shm_data=False)
        self.linear_state_client = None
        if backend.is_linear_att_mixed_model:
            from lightllm.server.multi_level_kv_cache.linear_state_cache import CpuLinearStateCacheClient

            self.linear_state_client = CpuLinearStateCacheClient(self.cpu_cache_client, False, False)

    @lru_cache()
    def need_sync_compute_stream(self) -> bool:
        """
        fa3 在 offload 和 load kv cache 的时候，需要等待计算流完成，否则可能会概率崩溃。
        """

        model = self.backend.model
        att_backends = [
            model.prefill_att_backend,
            model.decode_att_backend,
            model.prefill_att_backend1,
            model.decode_att_backend1,
        ]
        for att_backend in att_backends:
            if att_backend is not None and "fa3" in att_backend.__class__.__name__.lower():
                logger.info("MultiLevelKvCacheModule: need sync compute stream for fa3 backend.")
                return True
        logger.info("MultiLevelKvCacheModule: no need sync compute stream.")
        return False

    def load_cpu_cache_to_reqs(self, reqs: List[InferReq]):
        idle_token_num = g_infer_context.get_can_alloc_token_num()
        all_page_list = []
        all_state_pages = []
        is_master_in_dp = self.backend.is_master_in_dp
        for req in reqs:
            page_list = req.shm_req.cpu_cache_match_page_indexes.get_all()
            state_page = req.shm_req.linear_att_cpu_state_page if self.linear_state_client is not None else -1
            if state_page >= 0:
                all_state_pages.append(state_page)
            # 需要返回 prompt logprobs 的请求不应加载 cpu cache：
            # 命中后会复用缓存 kv、跳过推理，拿不到对应 logprobs。
            # match 侧通常已跳过；这里仍要 deref 已 match 的 page，避免引用泄漏。
            if req.sampling_param.shm_param.prompt_logprobs >= 0 or req.sampling_param.disable_prompt_cache:
                if is_master_in_dp:
                    req.shm_req.cpu_prompt_cache_len = 0
                all_page_list.extend(page_list)
                continue

            page_len_list = req.shm_req.token_hash_page_len_list.get_all()
            original_pages = page_list
            use_local_state = False
            if self.linear_state_client is not None and self.backend.linear_att_checkpoint_cache is None:
                page_list = []
                page_len_list = []
            if self.linear_state_client is not None and page_list:
                coverage = page_len_list[len(page_list) - 1]
                remote_length = (
                    self.linear_state_client.page_items.get_item_by_index(state_page).token_count
                    if state_page >= 0
                    else 0
                )
                cache = self.backend.linear_att_checkpoint_cache
                with cache.lock:
                    # Leave one input token to compute logits, even for an
                    # exact physical-page hit with a retained prompt-end state.
                    candidates = cache.index.matching_lengths(
                        req.get_input_token_ids(), min(coverage, req.shm_req.input_len - 1)
                    )
                    gathered = [None] * self.backend.dp_world_size
                    dist.all_gather_object(gathered, candidates, group=self.gloo_group)
                    common = candidates.intersection(*gathered)
                    local_length = max(common, default=0)
                    resume_length = max(local_length, remote_length)
                    req.linear_checkpoint_demand = max(
                        req.linear_checkpoint_demand,
                        coverage // cache.policy.hash_page_size * cache.policy.hash_page_size,
                    )
                    if local_length > remote_length and 0 < local_length - req.cur_kv_len <= idle_token_num:
                        checkpoint = cache.restore(
                            req.get_input_token_ids(), local_length, req_manager=self.backend.model.req_manager, req=req
                        )
                        assert checkpoint.token_count == local_length
                        use_local_state = True
                page_len_list = list(
                    range(self.args.cpu_cache_token_page_size, resume_length, self.args.cpu_cache_token_page_size)
                )
                if resume_length:
                    page_len_list.append(resume_length)
                page_list = page_list[: len(page_len_list)]
                if self.backend.is_master_in_dp:
                    req.shm_req.disk_prompt_cache_len = min(
                        req.shm_req.disk_prompt_cache_len, max(0, resume_length - req.cur_kv_len)
                    )
            page_len_start_list = [0] + page_len_list
            assert len(page_list) <= len(page_len_list)

            if page_list:
                match_tokens = page_len_list[len(page_list) - 1]
            else:
                match_tokens = 0

            # 更新命中的 cpu kv cache 长度, 减去radix cache和disk cache的部分.
            if is_master_in_dp:
                req.shm_req.cpu_prompt_cache_len = max(
                    0, match_tokens - req.cur_kv_len - req.shm_req.disk_prompt_cache_len
                )

            need_token_num = match_tokens - req.cur_kv_len
            if self.linear_state_client is not None and need_token_num > idle_token_num and is_master_in_dp:
                req.shm_req.cpu_prompt_cache_len = 0
                req.shm_req.disk_prompt_cache_len = 0
            # 多匹配了一定数量的token同时请求长度大于一定的长度，才进行复制操作，不然操作效率不高，代价过高
            if need_token_num > 0 and (
                self.linear_state_client is not None or (need_token_num >= 128 and req.shm_req.input_len >= 256)
            ):
                if need_token_num <= idle_token_num:
                    if self.backend.radix_cache is not None:
                        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(need_token_num=need_token_num)

                    # 计算需要加载的页面（只加载未匹配的部分）
                    ready_page_num = bisect.bisect_right(page_len_list, req.cur_kv_len)
                    assert ready_page_num <= len(page_list)
                    need_pages = page_list[ready_page_num:]  # 只取需要的页面

                    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(need_size=need_token_num)

                    if self.need_sync_compute_stream():
                        # TODO fa3 现在必须使用同步模式, 未来需要移除
                        torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
                        # g_infer_context.get_overlap_stream().synchronize()

                    mem_manager = self.backend.model.mem_manager
                    req_manager = self.backend.model.req_manager

                    mem_indexes_cuda = mem_indexes.cuda(non_blocking=True)
                    page_indexes_cuda = torch.tensor(need_pages, dtype=torch.int32, device="cpu").cuda(
                        non_blocking=True
                    )
                    # Physical KV transfer starts at the page boundary. The
                    # operator masks the unused tail; state restores separately.
                    _start = page_len_start_list[ready_page_num]

                    _end = req.cur_kv_len
                    assert 0 <= _start <= _end, f"invalid pad range [{_start}, {_end}]"
                    mem_indexes_cuda = torch.cat(
                        [req_manager.req_to_token_indexs[req.req_idx, _start:_end], mem_indexes_cuda]
                    )

                    assert (
                        len(mem_indexes_cuda) == page_len_list[len(page_list) - 1] - page_len_start_list[ready_page_num]
                    )

                    # 更新 req 状态。
                    idle_token_num -= need_token_num
                    g_infer_context.req_manager.req_to_token_indexs[
                        req.req_idx, req.cur_kv_len : (req.cur_kv_len + need_token_num)
                    ] = mem_indexes
                    req.cur_kv_len = req.cur_kv_len + need_token_num

                    mem_manager.operator.load_cpu_cache_to_gpu(
                        mem_indexes=mem_indexes_cuda,
                        page_indexes=page_indexes_cuda,
                        cpu_cache_client=self.cpu_cache_client,
                        req=req,
                    )
                    if self.linear_state_client is not None and not use_local_state:
                        assert state_page >= 0
                        conv, ssm = self.linear_state_client.state_views(state_page, self.backend.rank_in_dp)
                        req_manager.restore_linear_att_state(req, conv, ssm)

                torch.cuda.current_stream().synchronize()

                if self.backend.is_master_in_dp:
                    req.shm_req.shm_cur_kv_len = req.cur_kv_len

            all_page_list.extend(original_pages)

        dist.barrier(group=self.init_sync_group)

        if self.backend.is_master_in_dp:
            if self.linear_state_client is not None:
                self.linear_state_client.lock.acquire_sleep1ms()
            self.cpu_cache_client.lock.acquire_sleep1ms()
            self.cpu_cache_client.deref_pages(page_list=all_page_list)
            self.cpu_cache_client.lock.release()
            if self.linear_state_client is not None:
                self.linear_state_client.deref_pages(all_state_pages)
                self.linear_state_client.lock.release()
        for req in reqs:
            if self.linear_state_client is not None:
                req.shm_req.linear_att_cpu_state_page = -1
        return

    def offload_finished_reqs_to_cpu_cache(self, finished_reqs: List[InferReq]) -> List[InferReq]:
        """
        将满足cpu kv cache 卸载条件的请求进行处理, 并返回真的满足退出条件的请求list。
        """
        # 如果开启了cpu cache，将达到finished状态的请求开启将gpu kv cache 卸载到 cpu cache中的操作。
        # 当 kv cache 卸载完成后，才会进行请求的真实退出操作。
        true_finished_reqs = []
        cpu_stream = g_infer_context.get_cpu_kv_cache_stream()
        for req in finished_reqs:
            # 只有 group_req_id 和 request_id 相同的请求才会被卸载到 cpu cache 中。
            # 这个限制是为了兼容 diverse 模式下的请求处理, 只有主请求才 offload kv 到 cpu
            # cache 中
            if req.shm_req.group_req_id != req.shm_req.request_id:
                true_finished_reqs.append(req)
                continue

            # 过滤不适合进行 kv 卸载到 cpu cache 的请求。
            if g_infer_context.is_linear_att_mixed_model:
                offload_limit_size = 1
            else:
                offload_limit_size = self.args.cpu_cache_token_page_size

            if req.cur_kv_len < offload_limit_size or req.shm_req.input_len <= offload_limit_size:
                true_finished_reqs.append(req)
                continue

            # 如果请求已经完成了 cpu cache 的任务，则满足了退出条件
            if req.cpu_cache_task_status.is_finished():
                true_finished_reqs.append(req)
                continue

            # 如果请求已经发起过卸载任务且正在卸载过程中，则在当前轮不进行处理
            if req.cpu_cache_task_status.is_running():
                continue

            assert req.cpu_cache_task_status.is_not_started()

            if self.need_sync_compute_stream():
                # TODO fa3 现在必须使用同步模式, 未来需要移除, 必须等待 overlap stream 上的计算任务完成，不然会崩溃
                g_infer_context.get_overlap_stream().synchronize()

            # 发起将请求的 kv cache 卸载到 cpu cache 中的任务
            trans_task = self._start_kv_cache_offload_task(req=req, cpu_kv_cache_stream=cpu_stream)

            # 根据是否成功创建了卸载任务，决定是否将请求加入到处理队列中
            if trans_task is not None:
                self.cpu_cache_handle_queue.append(trans_task)
            else:
                true_finished_reqs.append(req)

        if self.need_sync_compute_stream():
            # TODO fa3 现在必须使用同步模式, 未来需要移除
            cpu_stream.synchronize()

        return true_finished_reqs

    def _start_kv_cache_offload_task(
        self, req: InferReq, cpu_kv_cache_stream: torch.cuda.Stream
    ) -> Optional["TransTask"]:
        assert CacheTier.CPU in req.cache_tiers
        disk_offload_enable = CacheTier.DISK in req.cache_tiers
        with torch.cuda.stream(cpu_kv_cache_stream):
            if self.linear_state_client is not None:
                from lightllm.common.linear_att_cache_manager.checkpoints import prefix_hashes

                # Include committed output, including a partial physical page.
                length = min(req.cur_kv_len, req.linear_output_cache_len or req.cur_kv_len)
                page_len_list = list(
                    range(self.args.cpu_cache_token_page_size, length, self.args.cpu_cache_token_page_size)
                ) + [length]
                hashes = prefix_hashes(req.get_input_token_ids(), page_len_list)
                token_hash_list = [hashes[n] for n in page_len_list]
            else:
                token_hash_list = req.shm_req.token_hash_list.get_all()
                page_len_list = req.shm_req.token_hash_page_len_list.get_all()
            assert len(token_hash_list) == len(page_len_list)

            if self.backend.is_master_in_dp:
                find_index = bisect.bisect_right(page_len_list, req.cur_kv_len)
                move_block_size = find_index

                if move_block_size == 0:
                    dist.broadcast_object_list([0], group=self.gloo_group, group_src=0)
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None

                try:
                    if self.linear_state_client is not None:
                        self.linear_state_client.lock.acquire_sleep1ms()
                    self.cpu_cache_client.lock.acquire_sleep1ms()
                    allocator = (
                        self.linear_state_client.allocate_kv_pages
                        if self.linear_state_client is not None
                        else self.cpu_cache_client.allocate_pages
                    )
                    page_list, ready_list = allocator(
                        token_hash_list[:move_block_size],
                        disk_offload_enable=disk_offload_enable,
                    )
                finally:
                    self.cpu_cache_client.lock.release()
                    if self.linear_state_client is not None:
                        self.linear_state_client.lock.release()

                item_size = len(page_list)
                if item_size == 0:
                    dist.broadcast_object_list([0], group=self.gloo_group, group_src=0)
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None

                broadcast_data = {"item_size": item_size, "page_list": page_list, "ready_list": ready_list}
                dist.broadcast_object_list([broadcast_data], group=self.gloo_group, group_src=0)
            else:
                recv_list = [None]
                dist.broadcast_object_list(recv_list, group=self.gloo_group, group_src=0)
                if isinstance(recv_list[0], int) and recv_list[0] == 0:
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None
                broadcast_data = recv_list[0]
                item_size = broadcast_data["item_size"]
                page_list = broadcast_data["page_list"]
                ready_list = broadcast_data["ready_list"]

            page_indexes = torch.tensor(page_list, dtype=torch.int32, device="cpu", pin_memory=True)
            page_readies = torch.tensor(ready_list, dtype=torch.bool, device="cpu", pin_memory=True)
            assert len(page_indexes) <= self.page_index_buffer.shape[0]
            cuda_page_indexes = self.page_index_buffer[: len(page_indexes)]
            cuda_page_readies = self.page_ready_buffer[: len(page_readies)]
            cuda_page_indexes.copy_(page_indexes, non_blocking=True)
            cuda_page_readies.copy_(page_readies, non_blocking=True)

            move_token_num = page_len_list[item_size - 1]
            assert req.cur_kv_len >= move_token_num
            token_indexes = self.backend.model.req_manager.req_to_token_indexs[req.req_idx, 0:move_token_num]

            mem_manager = self.backend.model.mem_manager

            mem_manager.operator.offload_gpu_kv_to_cpu_cache(
                mem_indexes=token_indexes,
                page_indexes=cuda_page_indexes,
                page_readies=cuda_page_readies,
                cpu_cache_client=self.cpu_cache_client,
                req=req,
            )
            state_pages = self._offload_linear_checkpoints(req, page_list)

            # 这个操作只是为了在offload 对应的cuda stream中，同步标记下对应的kv cache offload 操作已经完成，
            if self.backend.dp_world_size > 1:
                dist.all_reduce(self.offload_sync_tensor, op=dist.ReduceOp.MAX, group=self.offload_sync_group)

            sync_event = torch.cuda.Event()
            sync_event.record()
            req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.RUNNING
            trans_task = TransTask(
                move_token_num=move_token_num,
                page_indexes=page_indexes,
                page_readies=page_readies,
                req_obj=req,
                sync_event=sync_event,
                state_pages=state_pages,
            )

        return trans_task

    def _offload_linear_checkpoints(self, req, kv_pages):
        if self.linear_state_client is None:
            return []
        from lightllm.common.linear_att_cache_manager.checkpoints import prefix_hashes

        cache = self.backend.linear_att_checkpoint_cache
        if cache is None:
            return []
        length = min(req.cur_kv_len, req.linear_output_cache_len or req.cur_kv_len)
        with cache.lock:
            lengths = {n for n, _ in cache.index.entries if n <= length}
            hashes = prefix_hashes(req.get_input_token_ids(), lengths)
            candidates = [(n, h) for n, h in hashes.items() if (n, h) in cache.index.entries]
            # Overlap threads may evict local snapshots in a different order.
            # Only publish snapshots for which every TP shard is still owned.
            gathered = [None] * self.backend.dp_world_size
            dist.all_gather_object(gathered, candidates, group=self.gloo_group)
            common = set(candidates)
            for rank_candidates in gathered:
                common.intersection_update(rank_candidates)
            descriptors = []
            client = self.linear_state_client
            if self.backend.is_master_in_dp:
                client.lock.acquire_sleep1ms()
                self.cpu_cache_client.lock.acquire_sleep1ms()
                try:
                    for n, h in sorted(common):
                        tail_index = (n - 1) // self.args.cpu_cache_token_page_size
                        if any(page < 0 for page in kv_pages[: tail_index + 1]):
                            continue
                        page_idx = client.allocate_checkpoint(h, n, kv_pages[tail_index])
                        if page_idx is not None:
                            descriptors.append((n, h, page_idx))
                finally:
                    self.cpu_cache_client.lock.release()
                    client.lock.release()
            data = [descriptors]
            dist.broadcast_object_list(data, group=self.gloo_group, group_src=0)
            descriptors = data[0]
            for n, h, page_idx in descriptors:
                source = cache.index.entries[(n, h)]
                conv, ssm = cache.buffers.get_state_cache(source.slot)
                dst_conv, dst_ssm = client.state_views(page_idx, self.backend.rank_in_dp)
                dst_conv.copy_(conv)
                dst_ssm.copy_(ssm)
            return [page_idx for _, _, page_idx in descriptors]

    def update_cpu_cache_task_states(self):
        if self.backend.is_master_in_dp:
            trans_ok_tasks = []
            while len(self.cpu_cache_handle_queue) != 0:
                task: TransTask = self.cpu_cache_handle_queue.popleft()
                if task.sync_event.query():
                    trans_ok_tasks.append(task)
                else:
                    self.cpu_cache_handle_queue.appendleft(task)
                    break
            item_size = len(trans_ok_tasks)
            dist.broadcast_object_list([item_size], group=self.filter_group, group_src=0)
        else:
            recv_list = [None]
            dist.broadcast_object_list(recv_list, group=self.filter_group, group_src=0)
            item_size = recv_list[0]
            trans_ok_tasks: List[TransTask] = [self.cpu_cache_handle_queue.popleft() for _ in range(item_size)]

        if item_size > 0:
            page_array_list = [task.page_indexes.tolist() for task in trans_ok_tasks]
            move_token_nums = [task.move_token_num for task in trans_ok_tasks]
            if self.backend.is_master_in_dp:
                if self.linear_state_client is not None:
                    self.linear_state_client.lock.acquire_sleep1ms()
                self.cpu_cache_client.lock.acquire_sleep1ms()
                # 分组update，避免不同请求的page交叉，导致disk cache hash不一致
                for task, pages, move_token_num in zip(trans_ok_tasks, page_array_list, move_token_nums):
                    self.cpu_cache_client.update_pages_status_to_ready(
                        page_list=pages,
                        deref=True,
                        disk_offload_enable=CacheTier.DISK in task.req_obj.cache_tiers,
                        token_num_in_page_list=move_token_num,
                    )
                    if self.linear_state_client is not None:
                        self.linear_state_client.update_pages_status_to_ready(task.state_pages)
                self.cpu_cache_client.lock.release()
                if self.linear_state_client is not None:
                    self.linear_state_client.lock.release()
            for task in trans_ok_tasks:
                task.req_obj.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
        return


@dataclasses.dataclass
class TransTask:
    move_token_num: int
    page_indexes: torch.Tensor
    page_readies: torch.Tensor
    req_obj: InferReq
    sync_event: torch.cuda.Event
    state_pages: List[int] = dataclasses.field(default_factory=list)
