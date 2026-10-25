from typing import List

import torch

from lightllm.utils.log_utils import init_logger


logger = init_logger("lightllm.common.req_manager")


class _ReqNode:
    def __init__(self, index):
        self.index = index
        self.next: "_ReqNode" = None


class _ReqLinkedList:
    def __init__(self, max_request_num):
        self.nodes = [_ReqNode(i) for i in range(max_request_num)]
        self.marks = [0 for _ in range(max_request_num)]
        self.root_node = _ReqNode(-1)
        for i in range(0, max_request_num - 1):
            self.nodes[i].next = self.nodes[i + 1]
        self.root_node.next = self.nodes[0]
        self.can_alloc_size = max_request_num
        return

    def alloc(self):
        if self.root_node.next is None:
            logger.warning("alloc req index fail")
            return None
        get_node = self.root_node.next
        self.root_node.next = self.root_node.next.next
        assert self.marks[get_node.index] == 0
        self.marks[get_node.index] = 1
        self.can_alloc_size -= 1
        return get_node.index

    def free(self, index):
        assert self.marks[index] == 1
        node = self.nodes[index]
        node.next = self.root_node.next
        self.root_node.next = node
        self.marks[index] = 0
        self.can_alloc_size += 1
        return

    def is_all_free(self):
        return self.can_alloc_size == len(self.marks)


class ReqManager:
    def __init__(self, max_request_num, cache_manager):
        from .req_sampling_params import ReqSamplingParamsManager

        assert cache_manager.req_to_token_indexs.shape[0] == max_request_num + 1
        self.req_list = _ReqLinkedList(max_request_num)
        self.cache_manager = cache_manager
        self.req_to_mtp_state_index = (
            torch.zeros((max_request_num + 1,), dtype=torch.int32, device=cache_manager.req_to_token_indexs.device)
            if cache_manager.args.mtp_step > 0 and cache_manager.state_manager is not None
            else None
        )
        if cache_manager.linear_state_manager is not None:
            cache_manager.linear_state_manager.bind_mtp_state_index(self.req_to_mtp_state_index)
        self.req_sampling_params_manager = ReqSamplingParamsManager(max_request_num)
        self.max_request_num = max_request_num
        self.HOLD_REQUEST_ID = max_request_num

    def init_hold_request_indexs(self, hold_token_indexes):
        assert (
            self.req_list.is_all_free()
        ), "hold request indexes can only be initialized when all requests are released"
        self.cache_manager.init_hold_request_indexs(hold_token_indexes)

    def alloc_slot(self):
        return self.req_list.alloc()

    def release_requests(self, reqs):
        self.cache_manager.release_request_caches(reqs)
        self._release_slots([req.req_idx for req in reqs])

    def _release_slots(self, req_indexes: List[int]):
        for req_index in req_indexes:
            self.req_list.free(req_index)
        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")

    def reset_slots(self):
        self.req_list = _ReqLinkedList(self.max_request_num)

    def commit_accepted_state(self, b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width):
        if self.cache_manager.state_manager is None:
            return
        self.cache_manager.state_manager.commit_accepted_state(
            req_to_mtp_state_index=self.req_to_mtp_state_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=verify_width,
        )
