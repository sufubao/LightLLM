import uuid
import numpy as np
from typing import Tuple
from ...batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue


class PDDecodeQueue(BaseQueue):
    def __init__(self, args, router, dp_index, dp_size_in_node) -> None:
        super().__init__(args, router, dp_index, dp_size_in_node)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req: Req, estimated_peak_token_num: int, batch_req_num: int) -> Tuple[bool, int, int]:
        # 与 batch 中尚未进入 decode 的请求使用相同的容量估算，直接累加 a_len + b_len。
        # get_tuple_tokens 统一处理输出长度估算，并计入分页对齐、MTP 和异步退出所需的余量，
        # 确保新请求准入时也为这些额外的 KV 占用预留容量。
        a_len, b_len = req.get_tuple_tokens(self.is_busy(), self.router.router_statics.ema_req_out_len)
        estimated_peak_token_num = estimated_peak_token_num + (a_len + b_len)
        ok_token_num = estimated_peak_token_num < self.max_total_tokens
        batch_req_num += 1
        ok_req_num = batch_req_num <= self.running_max_req_size

        if ok_token_num and ok_req_num:
            self.router.shared_token_load.set_estimated_peak_token_count(estimated_peak_token_num, self.dp_index)
            self.router.shared_token_load.set_dynamic_max_load(
                estimated_peak_token_num / self.max_total_tokens,
                self.dp_index,
            )
            return True, estimated_peak_token_num, batch_req_num
        else:
            return False, None, None

    def _caclu_batch_estimated_peak_token_num(self, batch: Batch):
        is_busy = self.is_busy()
        estimated_peak_token_num = 0
        owned_peak_token_num = 0
        decoding_req_list = []
        big_page_tokens = self.args.linear_att_hash_page_size * self.args.linear_att_page_block_num
        # PD Decode 启动时禁用 hybrid 大页，命中的 KV 全部保持 radix 引用。
        # 若未来启用大页，小页恢复可能复制私有 KV，继续使用原来的估算。
        use_cache = not self.args.disable_dynamic_prompt_cache and big_page_tokens > self.args.max_req_total_len
        if batch is not None:
            for req in batch.reqs:
                if req.sample_params.suggested_dp_index != self.dp_index:
                    continue
                a_len, b_len = req.get_tuple_tokens(is_busy, self.router.router_statics.ema_req_out_len)
                cache_len = min(req.prompt_cache_len, a_len) if use_cache else 0
                if req.is_infer_decode():
                    decoding_req_list.append((a_len, b_len, cache_len))
                else:
                    # 尚未初始化的请求 cache_len 为零，仍预留完整输入及输出。
                    estimated_peak_token_num += a_len + b_len
                    owned_peak_token_num += a_len + b_len - cache_len

        if decoding_req_list:
            decoding_req_list.sort(key=lambda x: -x[1])
            tokens = np.array(decoding_req_list)
            left_out_len_array = tokens[:, 1]
            size_array = np.arange(1, len(decoding_req_list) + 1)
            future_tokens = left_out_len_array * size_array
            estimated_peak_token_num += (future_tokens + np.cumsum(tokens[:, 0])).max()
            owned_peak_token_num += (future_tokens + np.cumsum(tokens[:, 0] - tokens[:, 2])).max()

        if use_cache and batch is not None:
            # 活跃前缀按 radix 的物理引用量计一次；请求结束后仍保留全部当前引用量，
            # 不假定共享节点会随单个请求退出而释放。读取在请求命中长度之后进行。
            refed_tokens = self.router.radix_cache_client.get_refed_tokens_num(self.dp_index)
            return min(estimated_peak_token_num, owned_peak_token_num + refed_tokens)
        return estimated_peak_token_num

    # @calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch: Batch):
        # 即使调度容量已满，也要先清理 abort 请求，避免继续占用共享请求槽位。
        self.filter_aborted_reqs()
        if not self.waiting_req_list:
            return None

        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        exist_req_num = self.get_batch_dp_req_size(current_batch)
        if exist_req_num >= self.running_max_req_size:
            return None

        estimated_peak_token_num = self._caclu_batch_estimated_peak_token_num(current_batch)
        batch_req_num = exist_req_num

        can_run_list = []
        consumed_req_count = 0

        waiting_queue = self.waiting_req_list

        for req in waiting_queue:
            ok_insert, estimated_peak_token_num, batch_req_num = self._can_add_new_req(
                req=req, estimated_peak_token_num=estimated_peak_token_num, batch_req_num=batch_req_num
            )
            if ok_insert:
                consumed_req_count += 1
                can_run_list.append(req)
            else:
                break
        new_batch = None
        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().int, can_run_list, dp_size_in_node=self.dp_size_in_node)
        self.waiting_req_list = self.waiting_req_list[consumed_req_count:]
        return new_batch

    def _calcu_batch_token_load_batch_not_none(self, current_batch: Batch):
        estimated_peak_token_num = self._caclu_batch_estimated_peak_token_num(current_batch)

        return (estimated_peak_token_num, estimated_peak_token_num / self.max_total_tokens)
