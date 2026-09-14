from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import PostLayerOutput
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector
from lightllm.models.llama.layer_infer import post_layer_infer as llama_post
from lightllm.models.gemma4.layer_infer.post_layer_infer import Gemma4PostLayerInfer


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_output_head_returns_candidates_without_mutating_state(monkeypatch, draft, enabled) -> None:
    head = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    head.tp_world_size_ = 1
    head.alloc_tensor = lambda shape, dtype, **kwargs: torch.empty(shape, dtype=dtype)
    head._norm = lambda hidden, state, weight: hidden
    dense = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    class Weight:
        vocab_size = 4
        tp_vocab_start_id = 0

        def __call__(self, **kwargs) -> torch.Tensor:
            return dense

    values, ids = dense.T[:, :2], torch.tensor([[0, 1]] * 3)
    monkeypatch.setattr(llama_post, "vocab_parallel_candidates", lambda **kwargs: (values, ids))
    args = SimpleNamespace(target_vocab_topk_sampling=None, draft_vocab_topk_sampling=None)
    setattr(args, "draft_vocab_topk_sampling" if draft else "target_vocab_topk_sampling", 16 if enabled else None)
    monkeypatch.setattr(llama_post, "get_env_start_args", lambda: args)
    state = SimpleNamespace(is_draft_model=draft, dist_group=None)
    weight = SimpleNamespace(lm_head_weight_=Weight())
    output = head._lm_head_and_gather(torch.ones(3, 2), 3, weight, state)
    assert not hasattr(state, "logits_token_ids")
    torch.testing.assert_close(output.logits, values if enabled else dense.T)
    assert output.logits_token_ids is (ids if enabled else None)
    prompt = head._lm_head_and_gather(torch.ones(3, 2), 3, weight, state, allow_vocab_candidates=False)
    torch.testing.assert_close(prompt.logits, dense.T)
    assert prompt.logits_token_ids is None


def test_prefill_prompt_logits_do_not_replace_candidate_output() -> None:
    head = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    state = SimpleNamespace(prompt_logics=torch.ones(5, 2))
    head._slice_get_last_input = lambda *args: (torch.ones(1, 2), 1)
    candidates = PostLayerOutput(torch.ones(1, 2), torch.tensor([[10, 20]]))
    prompt = torch.ones(5, 32)
    head._lm_head_and_gather = lambda *args, **kwargs: (
        candidates if kwargs.get("allow_vocab_candidates", True) else PostLayerOutput(prompt)
    )
    assert head.token_forward(None, state, None) is candidates
    assert state.prompt_logics is prompt


def test_model_output_and_unpadding_preserve_candidate_mapping() -> None:
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    logits = torch.randn(3, 2)
    ids = torch.tensor([[10, 20], [30, 40], [50, 60]])
    state = SimpleNamespace(hidden_collector=NoopHiddenCollector(), prompt_logics=None)
    output = model._create_model_output(PostLayerOutput(logits, ids), state)
    unpadded = model._create_unpad_decode_model_output(output, 2)
    torch.testing.assert_close(unpadded.logits, logits[:2])
    torch.testing.assert_close(unpadded.logits_token_ids, ids[:2])


def test_gemma_softcap_preserves_candidate_ids(monkeypatch) -> None:
    ids = torch.tensor([[10, 20]])
    logits = torch.tensor([[1.0, 4.0]])
    monkeypatch.setattr(llama_post.LlamaPostLayerInfer, "token_forward", lambda *args: PostLayerOutput(logits, ids))
    head = Gemma4PostLayerInfer.__new__(Gemma4PostLayerInfer)
    head.final_logit_softcapping = 2.0
    state = SimpleNamespace(prompt_logics=None)
    output = head.token_forward(None, state, None)
    torch.testing.assert_close(output.logits, torch.tanh(logits / 2) * 2)
    assert output.logits_token_ids is ids


def test_overlap_keeps_each_microbatch_mapping() -> None:
    head = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    outputs = [PostLayerOutput(torch.ones(1, 2), torch.tensor([[10, 20]])), PostLayerOutput(torch.ones(1, 3))]
    head.token_forward = lambda hidden, state, layer_weight: outputs[state]
    output0, output1 = head.overlap_tpsp_token_forward(None, None, 0, 1, None)
    assert output0 is outputs[0]
    assert output1 is outputs[1]
