from types import SimpleNamespace

import torch

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.hidden_collector import FinalHiddenCollector, FinalHiddenOutput
from lightllm.common.basemodel.layer_weights.meta_weights import NoTpGEMMANormWeight
from lightllm.models.llama.layer_infer import post_layer_infer as llama_post
from lightllm.models.qwen3_5_mtp.layer_infer.pre_layer_infer import Qwen3_5MTPPreLayerInfer


def test_qwen35_main_to_draft_two_steps(monkeypatch) -> None:
    def norm_weight(values):
        norm = NoTpGEMMANormWeight.__new__(NoTpGEMMANormWeight)
        norm.dim = 3
        norm.data_type_ = torch.float32
        norm.weight_name = "norm.weight"
        norm.weight = torch.empty(3)
        norm.load_hf_weights({"norm.weight": torch.tensor(values)})
        norm._forward = norm._native_forward
        return norm

    main_norm = norm_weight([0.5, -0.25, 1.0])
    draft_norm = norm_weight([-0.5, 1.0, 0.25])
    enorm = norm_weight([0.0, 0.25, -0.25])
    hnorm = norm_weight([0.25, -0.5, 0.5])
    projection = torch.arange(1, 16, dtype=torch.float32).reshape(5, 3) / 10
    fusion = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
    )

    class LMHead:
        vocab_size = 5

        def __call__(self, input, alloc_func):
            return projection @ input

    main_weight = SimpleNamespace(final_norm_weight_=main_norm, lm_head_weight_=LMHead())
    draft_weight = SimpleNamespace(
        final_norm_weight_=draft_norm,
        lm_head_weight_=LMHead(),
        enorm_weight_=enorm,
        hnorm_weight_=hnorm,
        eh_proj_weight_=SimpleNamespace(mm=lambda hidden: hidden @ fusion.T),
    )
    post = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    post.eps_ = 1e-6
    post.tp_world_size_ = 1
    post.alloc_tensor = lambda shape, dtype, **kwargs: torch.empty(shape, dtype=dtype)
    pre = Qwen3_5MTPPreLayerInfer.__new__(Qwen3_5MTPPreLayerInfer)
    pre.eps_ = post.eps_
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.layers_num = 0
    model.post_infer = post
    post._tpsp_allgather = lambda input, infer_state: input

    def forward(hidden, weight):
        model.pre_post_weight = weight
        model.pre_infer = SimpleNamespace(
            token_forward=lambda *args: hidden,
            _tpsp_sp_split=lambda input, infer_state: input,
        )
        return model._token_forward(state)

    state = SimpleNamespace(
        is_prefill=False,
        is_cuda_graph=False,
        input_ids=torch.ones(2, dtype=torch.long),
        is_draft_model=False,
        batch_size=2,
        prompt_logics=None,
        hidden_collector=FinalHiddenCollector(output=FinalHiddenOutput.POST_NORM),
    )
    monkeypatch.setattr(
        llama_post,
        "get_env_start_args",
        lambda: SimpleNamespace(target_vocab_topk_sampling=None, draft_vocab_topk_sampling=None),
    )

    def reference_norm(hidden, weight):
        return torch.nn.functional.rms_norm(hidden, (3,), weight.weight, eps=post.eps_)

    hidden = torch.tensor([[1.0, 2.0, -3.0], [4.0, -2.0, 1.0]])
    expected = reference_norm(hidden, main_norm)
    output = forward(hidden, main_weight)
    torch.testing.assert_close(output.mtp_collector.spec_hidden, expected)
    torch.testing.assert_close(output.logits, expected @ projection.T)
    assert state.hidden_collector.final_hidden is None

    state.is_draft_model = True
    for step in range(2):
        embedding = torch.tensor([[2.0, -1.0, 3.0], [-3.0, 2.0, 1.0]]) + step
        state.mtp_draft_input_hiddens = output.mtp_collector.spec_hidden
        expected_fused = (
            torch.cat(
                (reference_norm(embedding, enorm), reference_norm(expected, hnorm)),
                dim=-1,
            )
            @ fusion.T
        )
        fused = pre._mtp_fuse(embedding, state, draft_weight)
        torch.testing.assert_close(fused, expected_fused)
        expected = reference_norm(expected_fused, draft_norm)
        output = forward(fused, draft_weight)
        torch.testing.assert_close(output.mtp_collector.spec_hidden, expected)
        torch.testing.assert_close(output.logits, expected @ projection.T)
        assert state.hidden_collector.final_hidden is None
        assert not torch.allclose(expected, reference_norm(expected_fused, main_norm))
