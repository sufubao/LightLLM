import torch

from lightllm.common.mtp_workspace import (
    build_compact_mtp_ssm_indices,
    build_contiguous_mtp_ssm_indices,
)
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Qwen3NextInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.gate_value = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        mtp_step = self.runtime_mtp_step
        self.use_mtp_workspace = False
        memory_aware_mtp = getattr(model.req_manager, "memory_aware_mtp", False)

        is_mtp_draft_model = getattr(model, "is_mtp_draft_model", False)
        if is_mtp_draft_model:
            return

        # prefill 模式下
        if self.is_prefill:
            self.b_conv_buffer_idx = self.b_req_idx
            self.b_ssm_buffer_idx = self.b_req_idx if memory_aware_mtp else self.b_req_idx * (mtp_step + 1)
            return

        # decode 模式下
        if mtp_step == 0:
            # 非mtp模式下，不需要额外状态
            self.b_conv_buffer_idx = self.b_req_idx
            self.b_ssm_buffer_idx = self.b_req_idx
            return

        if mtp_step > 0:
            # mtp 模式下
            batch_size = self.batch_size
            att_batch_size = batch_size // (mtp_step + 1)
            assert batch_size % (mtp_step + 1) == 0

            # shape 为 [att_batch_size + 1]
            self.b1_mtp_cu_q_seq_len = torch.arange(
                0,
                batch_size + 1,
                mtp_step + 1,
                dtype=torch.int32,
                device=self.b_req_idx.device,
            )
            # shape 为 [att_batch_size]
            self.b_canonical_req_idx = self.b_req_idx.view(att_batch_size, mtp_step + 1)[:, 0].contiguous()
            if memory_aware_mtp:
                assert self.b_mtp_workspace_idx is not None
                assert self.b_mtp_workspace_idx.shape[0] == att_batch_size
                self.b_conv_buffer_idx = self.b_mtp_workspace_idx
                if self.use_contiguous_mtp_ssm_workspace:
                    self.b_ssm_buffer_idx = build_contiguous_mtp_ssm_indices(
                        workspace_idx=self.b_mtp_workspace_idx,
                        canonical_size=model.req_manager.max_request_num + 1,
                        mtp_step=mtp_step,
                    )
                else:
                    self.b_ssm_buffer_idx = build_compact_mtp_ssm_indices(
                        canonical_req_idx=self.b_canonical_req_idx,
                        workspace_idx=self.b_mtp_workspace_idx,
                        canonical_size=model.req_manager.max_request_num + 1,
                        mtp_step=mtp_step,
                    )
            else:
                self.b_conv_buffer_idx = self.b_canonical_req_idx
                self.b_ssm_buffer_idx = (self.b_conv_buffer_idx * (mtp_step + 1)).view(
                    att_batch_size, 1
                ) + torch.arange(
                    mtp_step + 1,
                    device=self.b_req_idx.device,
                    dtype=self.b_req_idx.dtype,
                ).view(
                    1, mtp_step + 1
                )
            # shape 为 [att_batch_size]
            # 上一步接受的数量，用于linear att 的decode mtp 算子定位正确的conv 和 ssm信息的起点。
            if memory_aware_mtp:
                self.b_num_accepted_tokens = model.req_manager.req_to_mtp_state_index[self.b_canonical_req_idx] + 1
                self.use_mtp_workspace = True
            else:
                self.b_num_accepted_tokens = model.req_manager.req_to_mtp_state_index[self.b_conv_buffer_idx] + 1
            return
        return
