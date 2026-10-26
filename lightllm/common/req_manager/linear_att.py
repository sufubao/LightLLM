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
        self.req_to_conv_state = self.linear_state_manager.conv_state
        self.req_to_ssm_state = self.linear_state_manager.ssm_state
        self.req_to_mtp_state_index = self.linear_state_manager.mtp_state_index

    def init_hybrid_attention_state(self, req: "InferReq"):
        conv_index = req.req_idx
        ssm_start = req.req_idx * (self.mtp_step + 1)
        self.req_to_conv_state.buffer[:, conv_index, ...].fill_(0)
        # #17: zero the FULL (mtp_step + 1)-row SSM block, not just canonical row +0, so a future
        # first-step verify reading offset>0 after fresh init never hits a never-written row (NaN).
        self.req_to_ssm_state.buffer[:, ssm_start : ssm_start + (self.mtp_step + 1), ...].fill_(0)
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return

    def create_small_page_cache_manager(self, size: int):
        self.small_page_buffers = LinearAttCacheManager(size=size, linear_config=self.linear_config)
        return self.small_page_buffers

    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int]):
        from lightllm.common.basemodel.triton_kernel.linear_att_copy import copy_linear_att_state_to_kv_buffer

        buffer_indexes = torch.tensor(buffer_indexes, dtype=torch.int32, device="cpu").cuda(non_blocking=True)
        state_cache_manager = self.big_page_buffers
        copy_linear_att_state_to_kv_buffer(
            b_req_idx=b_req_idx,
            big_page_buffer_ids=buffer_indexes,
            gpu_conv_state=self.req_to_conv_state.buffer,
            gpu_ssm_state=self.req_to_ssm_state.buffer,
            cpu_kv_conv_state=state_cache_manager.conv_state_cache.buffer,
            cpu_kv_ssm_state=state_cache_manager.ssm_state_cache.buffer,
            mtp_step=self.mtp_step,
        )
        return

    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager: LinearAttCacheManager):
        # checkpoint 只保存标准 conv 窗口和请求的基准 SSM 状态，不包含 MTP 扩展运行态。
        conv_cache_width = self.linear_config.get_conv_state_shape()[-1]
        gpu_conv_state = self.req_to_conv_state.buffer[:, req_idx, ..., :conv_cache_width]
        gpu_ssm_state = self.req_to_ssm_state.buffer[:, req_idx * (self.mtp_step + 1), ...]
        dst_conv_state, dst_ssm_state = state_cache_manager.get_state_cache(buffer_idx=buffer_idx)
        dst_conv_state.copy_(gpu_conv_state, non_blocking=True)
        dst_ssm_state.copy_(gpu_ssm_state, non_blocking=True)

    def get_mamba_cache(self, layer_idx_in_all: int):
        assert (
            0 <= layer_idx_in_all < self.linear_config.all_layer_num
        ), f"invalid transformer layer index {layer_idx_in_all}"
        layer_idx_in_linear = layer_idx_in_all - (layer_idx_in_all // self.linear_config.full_attention_interval)
        conv_states = self.req_to_conv_state.buffer[layer_idx_in_linear]
        ssm_states = self.req_to_ssm_state.buffer[layer_idx_in_linear]
        return conv_states, ssm_states

    def update_mtp_state(self, b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width):
        from lightllm.common.basemodel.triton_kernel.mtp_utils import linear_att_mtp_state_index_update

        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=self.req_to_mtp_state_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=verify_width,
        )

    def restore_state(self, req: "InferReq", state_cache_manager: LinearAttCacheManager, buffer_idx: int):
        conv_state, ssm_state = state_cache_manager.get_state_cache(buffer_idx=buffer_idx)
        conv_dest = req.req_idx
        ssm_dest = req.req_idx * (self.mtp_step + 1)
        conv_cache_width = conv_state.shape[-1]
        self.req_to_conv_state.buffer[:, conv_dest, ..., :conv_cache_width] = conv_state
        self.req_to_ssm_state.buffer[:, ssm_dest, ...] = ssm_state
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return
