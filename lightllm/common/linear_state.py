import torch

from lightllm.common.state_cache_manager import LayerCache


class LinearStateManager:
    """Own GPU convolution/SSM buffers and per-request MTP cursors.

    Buffers live with the model and are borrowed by InferStateInfo. Each
    request uses one convolution row and `mtp_step + 1` contiguous SSM rows;
    the extra request slot is reserved for padded inputs.

    The lifecycle coordinator decides when to initialize, save or restore
    a request. This manager keeps its buffers and MTP cursor consistent.
    """

    def __init__(self, max_request_num, linear_config, args):
        self.linear_config = linear_config
        self.mtp_step = args.mtp_step
        self.mtp_state_index = (
            torch.zeros((max_request_num + 1,), dtype=torch.int32, device="cuda") if self.mtp_step > 0 else None
        )
        self.conv_state = LayerCache(
            size=max_request_num + 1,
            dtype=linear_config.conv_state_dtype,
            shape=linear_config.get_mtp_conv_state_shape(self.mtp_step),
            layer_num=linear_config.linear_layer_num,
            device="cuda",
        )
        self.ssm_state = LayerCache(
            size=(max_request_num + 1) * (self.mtp_step + 1),
            dtype=linear_config.ssm_state_dtype,
            shape=linear_config.get_ssm_state_shape(),
            layer_num=linear_config.linear_layer_num,
            device="cuda",
        )

    def reset(self, req_idx):
        """Clear convolution state, all SSM slots and the cursor before slot reuse."""
        ssm_start = req_idx * (self.mtp_step + 1)
        self.conv_state.buffer[:, req_idx].zero_()
        self.ssm_state.buffer[:, ssm_start : ssm_start + self.mtp_step + 1].zero_()
        if self.mtp_state_index is not None:
            self.mtp_state_index[req_idx] = 0

    def get_layer_buffers(self, layer_idx_in_all):
        """Return writable buffer views for a linear-attention transformer layer."""
        assert 0 <= layer_idx_in_all < self.linear_config.all_layer_num
        layer_idx = layer_idx_in_all - layer_idx_in_all // self.linear_config.full_attention_interval
        return self.conv_state.buffer[layer_idx], self.ssm_state.buffer[layer_idx]

    def copy_state_to(self, req_idx, conv_dst, ssm_dst):
        """Copy the base convolution window and canonical SSM row to a checkpoint.

        Speculative MTP state is excluded. Copies are non-blocking; the
        caller must synchronize before consuming the destination on the CPU.
        """
        conv_width = self.linear_config.get_conv_state_shape()[-1]
        conv_dst.copy_(self.conv_state.buffer[:, req_idx, ..., :conv_width], non_blocking=True)
        ssm_dst.copy_(self.ssm_state.buffer[:, req_idx * (self.mtp_step + 1)], non_blocking=True)

    def save_checkpoints(self, b_req_idx, req_indexes, buffer_indexes, state_cache_manager):
        """Copy base states to CPU checkpoint slots; a destination of -1 is skipped.

        The copy kernel runs on the current CUDA stream. The caller must
        synchronize before publishing or reading the checkpoints on the CPU.
        """
        from lightllm.common.basemodel.triton_kernel.linear_att_copy import copy_linear_att_state_to_kv_buffer

        buffer_indexes = torch.tensor(buffer_indexes, dtype=torch.int32, device="cpu").cuda(non_blocking=True)
        copy_linear_att_state_to_kv_buffer(
            b_req_idx=b_req_idx,
            big_page_buffer_ids=buffer_indexes,
            gpu_conv_state=self.conv_state.buffer,
            gpu_ssm_state=self.ssm_state.buffer,
            cpu_kv_conv_state=state_cache_manager.conv_state_cache.buffer,
            cpu_kv_ssm_state=state_cache_manager.ssm_state_cache.buffer,
            mtp_step=self.mtp_step,
        )

    def restore(self, req_idx, conv_state, ssm_state):
        """Restore base state and reset the MTP cursor to zero.

        Speculative SSM rows and the convolution tail are left unchanged.
        """
        self.conv_state.buffer[:, req_idx, ..., : conv_state.shape[-1]] = conv_state
        self.ssm_state.buffer[:, req_idx * (self.mtp_step + 1)] = ssm_state
        if self.mtp_state_index is not None:
            self.mtp_state_index[req_idx] = 0

    def commit_accepted_state(self, b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width):
        """Advance request cursors to the SSM slots selected by MTP verification."""
        from lightllm.common.basemodel.triton_kernel.mtp_utils import linear_att_mtp_state_index_update

        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=self.mtp_state_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=verify_width,
        )
