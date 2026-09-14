import math
from types import SimpleNamespace

import torch

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend import vocab_candidate_post_process


def test_draft_candidates_map_global_ids_and_approximate_confidence():
    backend = ModeBackend.__new__(ModeBackend)
    output = ModelOutput(
        logits=torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 2.0], [4.0, 4.0, 3.0]]),
        logits_token_ids=torch.tensor([[10, 110, 210], [20, 120, 220], [30, 130, 230]]),
    )
    # The last row ties across shards and must retain the lower global ID.
    expected_ids = torch.tensor([10, 120, 30])
    expected_probs = torch.tensor(
        [1 / (1 + math.exp(-1) + math.exp(-2)), 1 / (1 + math.exp(-1) + math.exp(-3)), 1 / (2 + math.exp(-1))]
    )
    greedy_ids = backend._gen_argmax_token_ids(output)
    token_ids, probs = backend._gen_argmax_token_ids_and_prob(output)
    torch.testing.assert_close(greedy_ids, expected_ids)
    torch.testing.assert_close(token_ids, expected_ids)
    torch.testing.assert_close(probs, expected_probs)

    rows = torch.tensor([2, 0])
    selected_output = ModelOutput(
        logits=output.logits.index_select(0, rows),
        logits_token_ids=output.logits_token_ids.index_select(0, rows),
    )
    selected_ids, selected_probs = backend._gen_argmax_token_ids_and_prob(selected_output)
    torch.testing.assert_close(selected_ids, expected_ids[rows])
    torch.testing.assert_close(selected_probs, expected_probs[rows])

    # Simulate the next graph replay overwriting model-owned output storage.
    output.logits.zero_()
    output.logits_token_ids.fill_(999)
    torch.testing.assert_close(greedy_ids, expected_ids)
    torch.testing.assert_close(token_ids, expected_ids)
    torch.testing.assert_close(probs, expected_probs)


def test_dense_draft_output_keeps_full_vocabulary_probability():
    backend = ModeBackend.__new__(ModeBackend)
    output = ModelOutput(logits=torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, math.log(3), 0.0, 0.0]]))
    token_ids, probs = backend._gen_argmax_token_ids_and_prob(output)
    torch.testing.assert_close(token_ids, torch.tensor([0, 1]))
    torch.testing.assert_close(probs, torch.tensor([0.25, 0.5]))


def test_target_candidate_sampling_maps_columns_outside_generic_sample(monkeypatch):
    class FakePinnedTensor:
        def __init__(self, data, dtype):
            self.tensor = torch.tensor(data, dtype=dtype)

        def cuda(self, non_blocking=True):
            return self.tensor

    class FakePinMemoryManager:
        def gen_from_list(self, key, data, dtype):
            return FakePinnedTensor(data, dtype)

    monkeypatch.setattr(vocab_candidate_post_process, "g_pin_mem_manager", FakePinMemoryManager())
    reqs = [
        SimpleNamespace(
            sampling_param=SimpleNamespace(shm_param=SimpleNamespace(temperature=1.0, top_p=1.0, top_k=1)),
            generator=None,
        ),
        SimpleNamespace(
            sampling_param=SimpleNamespace(shm_param=SimpleNamespace(temperature=1.0, top_p=1.0, top_k=1)),
            generator=None,
        ),
    ]
    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])
    token_ids = torch.tensor([[10, 20, 30], [40, 50, 60]])

    sampled_ids, sampled_logprobs = vocab_candidate_post_process.sample_vocab_candidates(
        logits.clone(), token_ids, reqs
    )

    torch.testing.assert_close(sampled_ids, torch.tensor([20, 60]))
    expected_logprobs = torch.log_softmax(logits, dim=-1).gather(1, torch.tensor([[1], [2]])).view(-1)
    torch.testing.assert_close(sampled_logprobs, expected_logprobs)
