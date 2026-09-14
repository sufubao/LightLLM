from typing import List, Tuple

import torch

from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.generic_post_process import (
    _random_sample,
    _top_p_top_k_sample,
)
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


def sample_vocab_candidates(
    logits: torch.Tensor,
    logits_token_ids: torch.Tensor,
    reqs: List[InferReq],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample candidate logits and translate candidate columns to global token IDs."""
    logits_width = logits.shape[-1]
    temperatures = []
    top_ps = []
    top_ks = []
    is_all_greedy = True
    skip_top_k = True
    skip_top_p = True
    exist_req_use_random_seed = False

    for req_obj in reqs:
        shm_param = req_obj.sampling_param.shm_param
        temperatures.append(shm_param.temperature)
        top_ps.append(shm_param.top_p)
        top_k = min(shm_param.top_k, logits_width)
        top_ks.append(top_k)
        is_all_greedy = is_all_greedy and top_k == 1
        skip_top_k = skip_top_k and top_k == logits_width
        skip_top_p = skip_top_p and shm_param.top_p == 1.0
        exist_req_use_random_seed = exist_req_use_random_seed or req_obj.generator is not None

    b_temperatures = g_pin_mem_manager.gen_from_list(key="temperatures", data=temperatures, dtype=torch.float32).cuda(
        non_blocking=True
    )
    b_top_ps = g_pin_mem_manager.gen_from_list(key="top_ps", data=top_ps, dtype=torch.float32).cuda(non_blocking=True)
    b_top_ks = g_pin_mem_manager.gen_from_list(key="top_ks", data=top_ks, dtype=torch.int32).cuda(non_blocking=True)

    logits.div_(b_temperatures.view((-1, 1)))
    probs = torch.softmax(logits, dim=-1)

    if is_all_greedy:
        candidate_indices = torch.argmax(logits, dim=-1)
        candidate_probs = probs.gather(1, candidate_indices.view(-1, 1))
        candidate_logprobs = torch.log(candidate_probs).view(-1)
    elif skip_top_k and skip_top_p:
        candidate_indices = _random_sample(probs, reqs, exist_req_use_random_seed)
        candidate_probs = probs.gather(1, candidate_indices.view(-1, 1))
        candidate_logprobs = torch.log(candidate_probs).view(-1)
    else:
        candidate_indices, candidate_logprobs = _top_p_top_k_sample(
            reqs,
            probs,
            b_top_ps,
            b_top_ks,
            exist_req_use_random_seed,
        )

    next_token_ids = logits_token_ids.gather(1, candidate_indices.long().view(-1, 1))
    return next_token_ids.view(-1), candidate_logprobs.view(-1)
