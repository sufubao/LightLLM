from types import SimpleNamespace

import torch

import lightllm.common.basemodel.infer_struct  # noqa: F401
from lightllm.common.req_manager import ReqSamplingParamsManager


def test_init_req_sampling_params_clears_reused_mtp_draft_tokens():
    manager = ReqSamplingParamsManager.__new__(ReqSamplingParamsManager)
    manager.penalty_counter_mode = "cpu_counter"
    manager.req_to_next_token_ids = torch.full((3, 4), 777, dtype=torch.int64)
    manager.req_to_next_token_scores = None
    manager.req_to_presence_penalty = torch.zeros(3)
    manager.req_to_frequency_penalty = torch.zeros(3)
    manager.req_to_repetition_penalty = torch.zeros(3)
    manager.req_to_exponential_decay_length_penalty = torch.zeros(3)

    shm_param = SimpleNamespace(
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (0, 1.0)),
        input_penalty=False,
    )
    req = SimpleNamespace(
        req_idx=1,
        sampling_param=SimpleNamespace(shm_param=shm_param),
        shm_req=SimpleNamespace(get_prompt_ids=lambda: []),
        get_last_gen_token=lambda: 42,
    )

    manager.init_req_sampling_params(req)

    assert manager.req_to_next_token_ids[1].tolist() == [42, 0, 0, 0]
