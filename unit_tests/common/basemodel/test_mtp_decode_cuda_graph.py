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


def test_mtp_decode_cuda_graph_warmup_builds_normal_layout_when_not_verify():
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


def test_mtp_decode_cuda_graph_keys_distinguish_verify_and_normal():
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    graph = CudaGraph.__new__(CudaGraph)
    graph.mtp_step = 2
    graph.graph = {}
    graph.cuda_graph_batch_sizes = [3, 6, 9, 12]

    verify_state = SimpleNamespace(
        input_ids=torch.ones(6, dtype=torch.int64),
        b_num_accepted_tokens=torch.ones(2, dtype=torch.int32),
    )
    normal_state = SimpleNamespace(
        input_ids=torch.ones(6, dtype=torch.int64),
        b_num_accepted_tokens=None,
    )

    # Same batch size, but the verify and normal decodes get distinct graph keys.
    assert graph._decode_graph_key(verify_state) == (6, True)
    assert graph._decode_graph_key(normal_state) == (6, False)
    assert graph.find_closest_graph_batch_size(5) == 6

    # A captured verify graph does not satisfy a normal-graph capture need at the same batch size.
    graph.graph[(6, True)] = "verify graph"
    assert graph.need_capture(6, is_mtp_verify_decode=True) is False
    assert graph.need_capture(6, is_mtp_verify_decode=False) is True


def test_mtp_decode_cuda_graph_warmup_layouts_use_verify_for_main_and_draft():
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    class Qwen3_5MOETpPartModel:
        pass

    class Qwen3_5MoeMTPModel:
        is_mtp_draft_model = True

    graph = CudaGraph.__new__(CudaGraph)
    graph.mtp_step = 2
    graph.cuda_graph_batch_sizes = [3, 6, 9]

    # Under MTP both the main verify forward and the pure-full-attention draft forward run the
    # (mtp_step+1)-grouped verify decode layout (the draft reuses the main model_input and keeps
    # b_num_accepted_tokens), so both warm up the verify graph key over the same batch-size set.
    assert list(graph._iter_warmup_graph_layouts(Qwen3_5MOETpPartModel())) == [(True, [3, 6, 9])]
    assert list(graph._iter_warmup_graph_layouts(Qwen3_5MoeMTPModel())) == [(True, [3, 6, 9])]

    # A non-MTP model (mtp_step == 0) warms up the normal layout instead.
    graph.mtp_step = 0
    assert list(graph._iter_warmup_graph_layouts(Qwen3_5MOETpPartModel())) == [(False, [3, 6, 9])]


def test_mtp_decode_warmup_layout_marks_qwen3next_verify(monkeypatch):
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for gen_decode_params")

    import lightllm.common.basemodel.mtp_verify_extra_state as mtp_verify_extra_state_mod
    from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo

    monkeypatch.setattr(mtp_verify_extra_state_mod, "get_env_start_args", lambda: SimpleNamespace(mtp_step=2))

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


def test_fa3_decode_uses_normal_layout_when_no_accept_tensor(monkeypatch):
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
