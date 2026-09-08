import torch
import triton

from .normal import NormalMemOperator
from lightllm.utils.envs_utils import get_env_start_args


class LinearAttMemOperator(NormalMemOperator):
    """Transfer full-attention KV only; recurrent snapshots have their own store."""

    def copy_kv_to_mem_manager(self, layer_index, mem_index, kv):
        layer_index = self.mem_manager.linear_config.get_full_att_kv_layer_index(layer_index)
        return super().copy_kv_to_mem_manager(layer_index, mem_index, kv)

    def _pad_page_indexes(self, mem_indexes):
        page_size = get_env_start_args().cpu_cache_token_page_size
        size = triton.cdiv(len(mem_indexes), page_size) * page_size
        if size == len(mem_indexes):
            return mem_indexes
        padded = torch.full((size,), -1, dtype=mem_indexes.dtype, device=mem_indexes.device)
        padded[: len(mem_indexes)] = mem_indexes
        return padded

    def load_cpu_cache_to_gpu(self, mem_indexes, page_indexes, cpu_cache_client, req):
        return super().load_cpu_cache_to_gpu(self._pad_page_indexes(mem_indexes), page_indexes, cpu_cache_client, req)

    def offload_gpu_kv_to_cpu_cache(self, mem_indexes, page_indexes, page_readies, cpu_cache_client, req):
        return super().offload_gpu_kv_to_cpu_cache(
            self._pad_page_indexes(mem_indexes), page_indexes, page_readies, cpu_cache_client, req
        )

    def copy_mem_to_mem(self, src_mem_index, dst_mem_index):
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        copy_kv_buffer_to_kv_buffer(
            src_mem_index.cuda(non_blocking=True), dst_mem_index.cuda(non_blocking=True), self.mem_manager.kv_buffer
        )

    def copy_kv_from_other_dp_ranks(self, *args, **kwargs):
        # This optional path transfers KV without a recurrent state. Keep
        # the previous unsupported behavior rather than silently resuming at
        # mismatched positions. Shared CPU checkpoint reuse supports DP.
        raise NotImplementedError("Hybrid DP prompt-cache fetch requires a matching recurrent-state transfer")
