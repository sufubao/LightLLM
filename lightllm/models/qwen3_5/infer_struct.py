import torch
from typing import List

from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo
from lightllm.utils.envs_utils import get_env_start_args


class Qwen35InferStateInfo(Qwen2VLInferStateInfo):
    def __init__(self):
        super().__init__()
        self.gate_value = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self.b_att_seq_len = self.b_seq_len
        mtp_step = get_env_start_args().mtp_step

        self.b_buffer_idx = self.b_req_idx * (mtp_step + 1) + self.b_mtp_index
        # conv buffer is now ONE widened slot per request (indexed by req_idx),
        # dropping the *(S+1) + mtp_index addressing used by the SSM block.
        self.b_conv_buffer_idx = self.b_req_idx
        # MTP verify batch: decode-mode, S+1 expanded, and gated on the
        # per-real-request accept tensor that decode_mtp threads in. Gating on
        # b_num_accepted_tokens (vs only b_mtp_index, which is set for any decode)
        # distinguishes the main-model verify forward from draft/plain decode.
        self.is_mtp_verify = (
            (mtp_step > 0)
            and (not self.is_prefill)
            and (self.b_mtp_index is not None)
            and (self.b_num_accepted_tokens is not None)
        )
        self.b_gdn_verify_cu_seqlens = None
        self.b_ssm_index_rows = None
        # b_num_accepted_tokens is threaded onto the infer_state from ModelInput by
        # _create_inferstate (mirrors b_mtp_index) BEFORE this runs; nothing to do here.
        if self.is_mtp_verify:
            step = mtp_step + 1
            n_real = self.b_req_idx.shape[0] // step
            self.b_gdn_verify_cu_seqlens = torch.arange(
                0, (n_real + 1) * step, step, dtype=torch.int32, device=self.b_req_idx.device
            )
            req_first = self.b_req_idx.view(n_real, step)[:, 0]
            base = (req_first * step).view(n_real, 1)
            self.b_ssm_index_rows = base + torch.arange(step, device=base.device, dtype=base.dtype).view(1, step)
            assert self.b_ssm_index_rows.shape == (n_real, step)
            # The spec conv kernel is per-SEQUENCE (one program per real request),
            # indexed by conv_state_indices[idx_seq] with idx_seq in [0, n_real),
            # aligned 1:1 with b_gdn_verify_cu_seqlens / b_num_accepted_tokens. The
            # default b_conv_buffer_idx = b_req_idx has the expanded length n_real*step,
            # which launches n_real*step conv programs and reads num_accepted/
            # query_start_loc out of bounds for idx_seq >= n_real, corrupting the
            # committed conv slot. Narrow it to one widened conv slot per request.
            self.b_conv_buffer_idx = req_first
        return
