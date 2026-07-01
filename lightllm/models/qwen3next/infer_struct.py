import torch

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.utils.envs_utils import get_env_start_args


class Qwen3NextInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.gate_value = None
        # MTP verify 阶段每个真实请求接受的 token 数。
        self.b_num_accepted_tokens: torch.Tensor = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self.init_normal_extra_state()
        self.init_mtp_extra_state(model)
        return

    def init_normal_extra_state(self):
        self.b_att_seq_len = self.b_seq_len
        self.b_buffer_idx = self.b_req_idx
        self.b_conv_buffer_idx = self.b_req_idx
        self.is_mtp_verify = False
        return

    def init_mtp_extra_state(self, model):
        mtp_step = get_env_start_args().mtp_step
        if mtp_step <= 0:
            return
        self.b_buffer_idx = self.b_req_idx * (mtp_step + 1) + self.b_mtp_index
        self.is_mtp_verify = (not self.is_prefill) and (self.b_mtp_index is not None)
        if self.is_mtp_verify:
            step = mtp_step + 1
            att_batch_size = self.b_req_idx.shape[0] // step
            self.b_gdn_verify_cu_seqlens = torch.arange(
                0, (att_batch_size + 1) * step, step, dtype=torch.int32, device=self.b_req_idx.device
            )
            req_first = self.b_req_idx.view(att_batch_size, step)[:, 0]
            base = (req_first * step).view(att_batch_size, 1)
            self.b_ssm_index_rows = base + torch.arange(step, device=base.device, dtype=base.dtype).view(1, step)
            assert self.b_ssm_index_rows.shape == (att_batch_size, step)
            self.b_conv_buffer_idx = req_first
            self.b_num_accepted_tokens = model.req_manager.req_to_accept_len[req_first]
        return
