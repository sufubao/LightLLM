from typing import TYPE_CHECKING, List

import torch

from lightllm.common.state_cache_manager import LinearAttCacheConfig, LinearAttCacheManager
from lightllm.common.linear_state import LinearStateManager
from lightllm.utils.envs_utils import get_env_start_args

from .hybrid_base import HybridAttentionReqManager


if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ReqManagerForMamba(HybridAttentionReqManager):
    def __init__(self, max_request_num, max_sequence_length, mem_manager, linear_config: LinearAttCacheConfig):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        args = get_env_start_args()
        self.linear_state_manager = LinearStateManager(max_request_num, linear_config, args)
        self.mtp_step = self.linear_state_manager.mtp_step
        if self.mtp_step > 0:
            assert args.enable_prefill_decode_mixed is False
        self.big_page_token_num = args.linear_att_page_block_num * args.linear_att_hash_page_size
        self.linear_config = linear_config

    def init_hybrid_attention_state(self, req: "InferReq"):
        self.linear_state_manager.reset(req.req_idx)

    def create_small_page_cache_manager(self, size: int):
        self.small_page_buffers = LinearAttCacheManager(size=size, linear_config=self.linear_config)
        return self.small_page_buffers

    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int]):
        self.linear_state_manager.save_checkpoints(b_req_idx, req_indexes, buffer_indexes, self.big_page_buffers)

    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager: LinearAttCacheManager):
        self.linear_state_manager.copy_state_to(req_idx, *state_cache_manager.get_state_cache(buffer_idx))

    def restore_state(self, req: "InferReq", state_cache_manager: LinearAttCacheManager, buffer_idx: int):
        self.linear_state_manager.restore(req.req_idx, *state_cache_manager.get_state_cache(buffer_idx))
