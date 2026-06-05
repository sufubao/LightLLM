from types import SimpleNamespace

import torch

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelInput


def test_mtp_decode_cuda_graph_warmup_uses_verify_layout():
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    graph = CudaGraph.__new__(CudaGraph)
    graph.mtp_step = 2
    graph.graph_max_len_in_batch = 128

    class FakeMemManager:
        HOLD_TOKEN_MEMINDEX = -1

        def alloc(self, size):
            return torch.arange(size, dtype=torch.int32)

    model = SimpleNamespace(
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=99),
        mem_manager=FakeMemManager(),
        _gen_special_model_input=lambda token_num: {"mtp_draft_input_hiddens": None},
    )

    model_input = graph._build_warmup_decode_model_input(model, batch_size=6, device="cpu")

    assert model_input.batch_size == 6
    assert model_input.b_mtp_index.tolist() == [0, 1, 2, 0, 1, 2]
    assert model_input.b_seq_len.tolist() == [2, 3, 4, 2, 3, 4]
    assert model_input.b_num_accepted_tokens.tolist() == [1, 1]
    assert model_input.total_token_num == 18


def test_mtp_decode_cuda_graph_warmup_supports_normal_layout_for_draft():
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    graph = CudaGraph.__new__(CudaGraph)
    graph.mtp_step = 2
    graph.graph_max_len_in_batch = 128

    class FakeMemManager:
        HOLD_TOKEN_MEMINDEX = -1

        def alloc(self, size):
            return torch.arange(size, dtype=torch.int32)

    model = SimpleNamespace(
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=99),
        mem_manager=FakeMemManager(),
        _gen_special_model_input=lambda token_num: {"mtp_draft_input_hiddens": torch.full((token_num, 4), 3.0)},
    )

    model_input = graph._build_warmup_decode_model_input(
        model,
        batch_size=5,
        device="cpu",
        is_mtp_verify_decode=False,
    )

    assert model_input.batch_size == 5
    assert model_input.b_mtp_index.tolist() == [0, 0, 0, 0, 0]
    assert model_input.b_seq_len.tolist() == [2, 2, 2, 2, 2]
    assert model_input.b_num_accepted_tokens is None
    assert model_input.total_token_num == 10
    assert model_input.mtp_draft_input_hiddens.shape == (5, 4)


def test_mtp_decode_cuda_graph_keys_verify_and_normal_layouts():
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    graph = CudaGraph.__new__(CudaGraph)
    graph.mtp_step = 2
    graph.graph = {}
    graph.normal_cuda_graph_batch_sizes = [1, 2, 4, 8]
    graph.mtp_verify_cuda_graph_batch_sizes = [3, 6, 9, 12]
    graph.cuda_graph_batch_sizes = graph.mtp_verify_cuda_graph_batch_sizes

    verify_state = SimpleNamespace(
        input_ids=torch.ones(6, dtype=torch.int64),
        b_num_accepted_tokens=torch.ones(2, dtype=torch.int32),
    )
    normal_state = SimpleNamespace(
        input_ids=torch.ones(6, dtype=torch.int64),
        b_num_accepted_tokens=None,
    )

    assert graph._decode_graph_key(verify_state) == (6, True)
    assert graph._decode_graph_key(normal_state) == (6, False)
    assert graph.find_closest_graph_batch_size(5, is_mtp_verify_decode=True) == 6
    assert graph.find_closest_graph_batch_size(5, is_mtp_verify_decode=False) == 8

    graph.graph[(6, True)] = "verify graph"
    assert graph.need_capture(6, is_mtp_verify_decode=True) is False
    assert graph.need_capture(5, is_mtp_verify_decode=False) is True


def test_mtp_decode_cuda_graph_warmup_layouts_split_main_and_draft_models():
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    class Qwen3_5MOETpPartModel:
        pass

    class Qwen3_5MoeMTPModel:
        pass

    graph = CudaGraph.__new__(CudaGraph)
    graph.mtp_step = 2
    graph.normal_cuda_graph_batch_sizes = [1, 2, 4, 8]
    graph.mtp_verify_cuda_graph_batch_sizes = [3, 6, 9]

    assert list(graph._iter_warmup_graph_layouts(Qwen3_5MOETpPartModel())) == [(True, [3, 6, 9])]
    assert list(graph._iter_warmup_graph_layouts(Qwen3_5MoeMTPModel())) == [(False, [1, 2, 4, 8])]


def test_mtp_decode_warmup_layout_marks_qwen3next_verify(monkeypatch):
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for gen_decode_params")

    import lightllm.models.qwen3next.infer_struct as infer_struct_mod
    from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo

    monkeypatch.setattr(infer_struct_mod, "get_env_start_args", lambda: SimpleNamespace(mtp_step=2))

    state = Qwen3NextInferStateInfo()
    state.is_prefill = False
    state.b_req_idx = torch.tensor([5, 5, 5, 6, 6, 6], dtype=torch.int32, device="cuda")
    state.b_mtp_index = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32, device="cuda")
    state.b_seq_len = torch.tensor([2, 3, 4, 2, 3, 4], dtype=torch.int32, device="cuda")
    state.b_num_accepted_tokens = torch.tensor([1, 2], dtype=torch.int32, device="cuda")

    model = SimpleNamespace(
        _cos_cached=torch.zeros((16, 4), dtype=torch.float32, device="cuda"),
        _sin_cached=torch.zeros((16, 4), dtype=torch.float32, device="cuda"),
    )

    state.init_some_extra_state(model)

    assert state.is_mtp_verify is True
    assert state.b_gdn_verify_cu_seqlens.tolist() == [0, 3, 6]
    assert state.b_conv_buffer_idx.tolist() == [5, 6]
    assert state.b_ssm_index_rows.tolist() == [[15, 16, 17], [18, 19, 20]]


def test_mtp_decode_padding_preserves_verify_groups(monkeypatch):
    import lightllm.common.basemodel.basemodel as basemodel_mod

    monkeypatch.setattr(basemodel_mod, "enable_diverse_mode_gqa_decode_fast_kernel", lambda: False)

    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args = SimpleNamespace(mtp_step=2)
    model.req_manager = SimpleNamespace(HOLD_REQUEST_ID=99)
    model.mem_manager = SimpleNamespace(HOLD_TOKEN_MEMINDEX=-1)

    model_input = ModelInput(
        batch_size=3,
        total_token_num=12,
        max_q_seq_len=1,
        max_kv_seq_len=4,
        input_ids=torch.tensor([10, 11, 12], dtype=torch.int32),
        mem_indexes=torch.tensor([20, 21, 22], dtype=torch.int32),
        b_req_idx=torch.tensor([7, 7, 7], dtype=torch.int32),
        b_mtp_index=torch.tensor([0, 1, 2], dtype=torch.int32),
        b_seq_len=torch.tensor([2, 3, 4], dtype=torch.int32),
        b_num_accepted_tokens=torch.tensor([2], dtype=torch.int32),
        is_prefill=False,
        multimodal_params=[{"images": [], "audios": []} for _ in range(3)],
    )

    padded = model._create_padded_decode_model_input(model_input, new_batch_size=6)

    assert padded.batch_size == 6
    assert padded.b_req_idx.tolist() == [7, 7, 7, 99, 99, 99]
    assert padded.b_mtp_index.tolist() == [0, 1, 2, 0, 1, 2]
    assert padded.b_seq_len.tolist() == [2, 3, 4, 2, 3, 4]
    assert padded.b_num_accepted_tokens.tolist() == [2, 1]
    assert padded.mem_indexes.tolist() == [20, 21, 22, -1, -1, -1]
    assert len(padded.multimodal_params) == 6


def test_qwen3next_hybrid_mtp_keeps_decode_cuda_graph_enabled(monkeypatch):
    import lightllm.models.qwen3next.model as qwen3next_model
    from lightllm.models.qwen3next.model import Qwen3NextTpPartModel

    monkeypatch.setattr(qwen3next_model, "get_env_start_args", lambda: SimpleNamespace(mtp_step=2))

    called = {}

    def fake_base_init_cudagraph(self):
        called["disable_cudagraph"] = self.disable_cudagraph
        self.graph = "captured"

    monkeypatch.setattr(TpPartBaseModel, "_init_cudagraph", fake_base_init_cudagraph)

    model = Qwen3NextTpPartModel.__new__(Qwen3NextTpPartModel)
    model.disable_cudagraph = False

    Qwen3NextTpPartModel._init_cudagraph(model)

    assert called["disable_cudagraph"] is False
    assert model.disable_cudagraph is False
    assert model.graph == "captured"


def test_fa3_decode_uses_normal_layout_for_narrowed_mtp_draft(monkeypatch):
    import lightllm.common.basemodel.attention.fa3.fp as fa3_fp
    from lightllm.common.basemodel.attention.fa3.fp import Fa3DecodeAttState

    monkeypatch.setattr(fa3_fp, "get_env_start_args", lambda: SimpleNamespace(mtp_step=2))

    copied = {}

    def fake_page_table_copy(page_table, req_to_token_indexs, b_req_idx):
        copied["page_table_shape"] = tuple(page_table.shape)
        copied["b_req_idx"] = b_req_idx.clone()

    monkeypatch.setattr(fa3_fp, "page_table_copy", fake_page_table_copy)

    model = SimpleNamespace(
        graph_max_batch_size=16,
        graph_max_len_in_batch=32,
        req_manager=SimpleNamespace(req_to_token_indexs=torch.empty((8, 32), dtype=torch.int32)),
    )
    backend = SimpleNamespace(
        model=model,
        get_page_table_buffer=lambda: [torch.empty(16 * 32, dtype=torch.int32)],
    )
    infer_state = SimpleNamespace(
        batch_size=2,
        max_kv_seq_len=16,
        input_ids=torch.ones(2, dtype=torch.int64),
        b_seq_len=torch.tensor([5, 7], dtype=torch.int32),
        b1_cu_q_seq_len=torch.tensor([0, 1, 2], dtype=torch.int32),
        b1_cu_kv_seq_len=torch.tensor([0, 5, 12], dtype=torch.int32),
        b_req_idx=torch.tensor([3, 4], dtype=torch.int32),
        b_num_accepted_tokens=None,
        microbatch_index=0,
    )

    state = Fa3DecodeAttState(backend=backend, infer_state=infer_state)
    state.init_state()

    assert state.decode_max_q_seq_len == 1
    assert state.b_att_seq_len.tolist() == [5, 7]
    assert copied["page_table_shape"] == (2, 16)
    assert copied["b_req_idx"].tolist() == [3, 4]


def test_build_eagle_accepted_draft_input_narrows_to_accepted_rows():
    from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
    from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import (
        ChunkedPrefillBackend,
    )

    backend = ChunkedPrefillBackend.__new__(ChunkedPrefillBackend)
    backend.mtp_step = 2

    main_input = ModelInput(
        batch_size=6,
        total_token_num=27,
        max_q_seq_len=1,
        max_kv_seq_len=9,
        input_ids=torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int64),
        mem_indexes=torch.tensor([100, 101, 102, 200, 201, 202], dtype=torch.int32),
        b_req_idx=torch.tensor([3, 3, 3, 4, 4, 4], dtype=torch.int32),
        b_mtp_index=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32),
        b_seq_len=torch.tensor([5, 6, 7, 6, 7, 8], dtype=torch.int32),
        b_num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32),
        is_prefill=False,
        multimodal_params=[
            {"row": 0},
            {"row": 1},
            {"row": 2},
            {"row": 3},
            {"row": 4},
            {"row": 5},
        ],
    )
    hidden = torch.arange(6 * 4, dtype=torch.float32).view(6, 4)
    main_output = ModelOutput(logits=torch.empty(6, 8), mtp_main_output_hiddens=hidden)
    next_token_ids = torch.tensor([110, 111, 112, 220, 221, 222], dtype=torch.int64)
    b_req_mtp_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    mtp_accept_len = torch.tensor([2, 3], dtype=torch.int32)

    (draft_input, accepted_next_tokens, accepted_req_idx,) = backend._build_eagle_accepted_draft_input(
        main_model_input=main_input,
        main_model_output=main_output,
        next_token_ids=next_token_ids,
        mtp_accept_len=mtp_accept_len,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
    )

    assert draft_input.batch_size == 2
    assert draft_input.input_ids.tolist() == [111, 222]
    assert draft_input.b_req_idx.tolist() == [3, 4]
    assert draft_input.b_mtp_index.tolist() == [1, 2]
    assert draft_input.b_seq_len.tolist() == [6, 8]
    assert draft_input.mem_indexes.tolist() == [101, 202]
    assert draft_input.b_num_accepted_tokens is None
    assert draft_input.multimodal_params == [{"row": 1}, {"row": 5}]
    assert accepted_next_tokens.tolist() == [111, 222]
    assert accepted_req_idx.tolist() == [3, 4]
    torch.testing.assert_close(draft_input.mtp_draft_input_hiddens, hidden[[1, 5]])


def test_eagle_draft_decode_uses_narrowed_hidden_on_first_step(monkeypatch):
    import lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl as chunked_impl
    from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
    from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import (
        ChunkedPrefillBackend,
    )

    class FakeMemManager:
        HOLD_TOKEN_MEMINDEX = -1

        def alloc(self, need_size):
            return torch.arange(300, 300 + need_size, dtype=torch.int32)

    req_to_next_token_ids = torch.empty((8, 3), dtype=torch.int64)
    monkeypatch.setattr(
        chunked_impl,
        "g_infer_context",
        SimpleNamespace(
            radix_cache=None,
            req_manager=SimpleNamespace(
                mem_manager=FakeMemManager(),
                req_sampling_params_manager=SimpleNamespace(req_to_next_token_ids=req_to_next_token_ids),
            ),
        ),
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)

    backend = ChunkedPrefillBackend.__new__(ChunkedPrefillBackend)
    backend.mtp_step = 2
    backend.num_mtp_models = 1

    seen_hiddens = []

    class FakeDraftModel:
        def forward(self, model_input):
            seen_hiddens.append(model_input.mtp_draft_input_hiddens.clone())
            logits = torch.zeros((model_input.batch_size, 8), dtype=torch.float32)
            return ModelOutput(
                logits=logits,
                mtp_main_output_hiddens=model_input.mtp_draft_input_hiddens + 100,
            )

    backend.draft_models = [FakeDraftModel()]

    scattered = {}

    def fake_scatter(accepted_req_idx, all_next_token_ids):
        scattered["accepted_req_idx"] = accepted_req_idx.clone()
        scattered["all_next_token_ids"] = all_next_token_ids.clone()

    backend._scatter_accepted_next_token_ids = fake_scatter

    main_input = ModelInput(
        batch_size=6,
        total_token_num=27,
        max_q_seq_len=1,
        max_kv_seq_len=9,
        input_ids=torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int64),
        mem_indexes=torch.tensor([100, 101, 102, 200, 201, 202], dtype=torch.int32),
        b_req_idx=torch.tensor([3, 3, 3, 4, 4, 4], dtype=torch.int32),
        b_mtp_index=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32),
        b_seq_len=torch.tensor([5, 6, 7, 6, 7, 8], dtype=torch.int32),
        b_num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32),
        is_prefill=False,
        multimodal_params=[{"images": [], "audios": []} for _ in range(6)],
    )
    hidden = torch.arange(6 * 4, dtype=torch.float32).view(6, 4)
    main_output = ModelOutput(logits=torch.empty(6, 8), mtp_main_output_hiddens=hidden)
    next_token_ids = torch.tensor([110, 111, 112, 220, 221, 222], dtype=torch.int64)
    b_req_mtp_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    mtp_accept_len = torch.tensor([2, 3], dtype=torch.int32)

    returned_mem = backend._draft_decode_eagle(
        main_model_input=main_input,
        main_model_output=main_output,
        next_token_ids=next_token_ids,
        mtp_accept_len=mtp_accept_len,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
    )

    assert returned_mem.tolist() == [300, 301, 302, 303]
    torch.testing.assert_close(seen_hiddens[0], hidden[[1, 5]])
    torch.testing.assert_close(seen_hiddens[1], hidden[[1, 5]] + 100)
    assert scattered["accepted_req_idx"].tolist() == [3, 4]
    assert scattered["all_next_token_ids"].shape == (2, 3)
