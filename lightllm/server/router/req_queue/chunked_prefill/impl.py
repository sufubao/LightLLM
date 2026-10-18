import uuid
import numpy as np
from ...batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue


class ChunkedPrefillQueue(BaseQueue):
    def _init_cache_list(self, current_batch: Batch, is_busy):
        if current_batch is not None:
            self.cache_len_list = [
                req.get_tuple_tokens(is_busy, self.router.router_statics.ema_req_out_len)
                for req in current_batch.reqs
                if req.sample_params.suggested_dp_index == self.dp_index
            ]
        else:
            self.cache_len_list = []
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req: Req, is_busy):
        self.cache_len_list.append(
            req.get_tuple_tokens(is_busy, self.router.router_statics.ema_req_out_len)
        )  # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        reserved = 0
        if getattr(self, "tool_call_admission_enabled", False):
            reservation_reader = getattr(self.router, "get_tool_call_reserved_tokens", None)
            if reservation_reader is not None:
                reserved = max(0, min(self.max_total_tokens, int(reservation_reader(self.dp_index))))
        # Idle tool prefixes are physically reclaimable, but admitting unrelated
        # work against all their space defeats the retention preference before
        # the continuation can reach inference. Account for that waiting space.
        capacity = self.max_total_tokens - reserved
        # Always allow a first feasible request to borrow the reservation. This
        # prevents head-of-line deadlock for a large request or a full budget.
        if (
            len(self.cache_len_list) == 1
            or getattr(req, "_tool_call_cache_hit", False)
            or getattr(getattr(req, "sample_params", None), "high_priority_request", False)
        ):
            capacity = self.max_total_tokens
        ok_token_num = need_max_token_num < capacity

        ok_req_num = len(self.cache_len_list) <= self.running_max_req_size

        if ok_token_num and ok_req_num:
            self.router.shared_token_load.set_estimated_peak_token_count(need_max_token_num, self.dp_index)
            self.router.shared_token_load.set_dynamic_max_load(
                need_max_token_num / self.max_total_tokens,
                self.dp_index,
            )
            return True
        else:
            return False

    # @calculate_time(show=True, min_cost_ms=10)
    def _prioritize_tool_continuations(self):
        self._tool_bypass_head = None
        self._tool_bypass_ids = set()
        if not getattr(self, "tool_call_admission_enabled", False):
            return
        reader = getattr(self.router, "get_tool_call_prefix_hints", None)
        hints = reader(self.dp_index) if reader is not None else None
        if hints is None:
            return
        limit = max(64, self.running_max_req_size * 2)
        head, tail = self.waiting_req_list[:limit], self.waiting_req_list[limit:]
        urgent, normal, matched = [], [], []
        for req in head:
            req._tool_call_cache_hit = False
            if req.sample_params.high_priority_request:
                urgent.append(req)
                continue
            if hints.by_length and not getattr(req.sample_params, "disable_prompt_cache", False):
                if not hasattr(req, "shm_prompt_ids"):
                    req.link_prompt_ids_shm_array()
                memo = getattr(req, "_tool_call_prefix_memo", None)
                if memo is None:
                    req._tool_call_prefix_memo = memo = {}
                req._tool_call_cache_hit = hints.matches(req.get_prompt_ids_numpy(), memo)
            normal.append(req)
            if req._tool_call_cache_hit:
                matched.append(req)
            elif self._tool_bypass_head is None:
                self._tool_bypass_head = req
        oldest = self._tool_bypass_head
        # At most eight successful continuation admissions can bypass the oldest
        # ordinary waiter. Explicit caller priority always remains ahead.
        quota = max(0, 8 - getattr(oldest, "_tool_call_priority_skips", 0)) if oldest is not None else len(matched)
        promoted = matched[:quota]
        promoted_ids = {id(req) for req in promoted}
        if oldest is not None:
            position = next(i for i, req in enumerate(normal) if req is oldest)
            self._tool_bypass_ids = {id(req) for req in normal[position + 1 :] if id(req) in promoted_ids}
        self.waiting_req_list = urgent + promoted + [req for req in normal if id(req) not in promoted_ids] + tail

    def generate_new_batch(self, current_batch: Batch):
        # 即使调度容量已满，也要先清理 abort 请求，避免继续占用共享请求槽位。
        self.filter_aborted_reqs()
        if not self.waiting_req_list:
            return None

        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        exist_req_num = self.get_batch_dp_req_size(current_batch)
        if exist_req_num >= self.running_max_req_size:
            return None

        original_waiting_queue = self.waiting_req_list
        self._prioritize_tool_continuations()

        is_busy = self.is_busy()

        self._init_cache_list(current_batch, is_busy)
        can_run_list = []
        consumed_req_count = 0

        waiting_queue = self.waiting_req_list

        for req in waiting_queue:
            ok_insert = self._can_add_new_req(req, is_busy)
            if ok_insert:
                consumed_req_count += 1
                can_run_list.append(req)
                if id(req) in self._tool_bypass_ids and self._tool_bypass_head is not None:
                    oldest = self._tool_bypass_head
                    oldest._tool_call_priority_skips = getattr(oldest, "_tool_call_priority_skips", 0) + 1
            else:
                break
        new_batch = None
        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().int, can_run_list, dp_size_in_node=self.dp_size_in_node)
        if getattr(self, "tool_call_admission_enabled", False):
            admitted = {id(req) for req in can_run_list}
            # Restore FIFO for requests that did not run. Repeated failed probes
            # must not permanently move continuations ahead without counting a
            # successful bypass against the oldest ordinary waiter.
            self.waiting_req_list = [req for req in original_waiting_queue if id(req) not in admitted]
        else:
            self.waiting_req_list = self.waiting_req_list[consumed_req_count:]
        return new_batch

    def _calcu_batch_token_load_batch_not_none(self, current_batch: Batch):
        is_busy = self.is_busy()
        self._init_cache_list(current_batch, is_busy)
        if len(self.cache_len_list) != 0:
            self.cache_len_list.sort(key=lambda x: -x[1])
            left_out_len_array = np.array([e[1] for e in self.cache_len_list])
            has_run_len_array = np.array([e[0] for e in self.cache_len_list])
            cum_run_len_array = np.cumsum(has_run_len_array)
            size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
            need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        else:
            need_max_token_num = 0

        return (need_max_token_num, need_max_token_num / self.max_total_tokens)
