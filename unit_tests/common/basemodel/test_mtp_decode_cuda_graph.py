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


def test_mtp_decode_cuda_graph_warmup_builds_normal_layout_for_non_mtp():
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    # A non-MTP model (mtp_step == 0) has a single, ungrouped decode layout.
    graph = CudaGraph.__new__(CudaGraph)
    graph.mtp_step = 0
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

    model_input = graph._build_warmup_decode_model_input(model, batch_size=5, device="cpu")

    assert model_input.batch_size == 5
    assert model_input.b_mtp_index.tolist() == [0, 0, 0, 0, 0]
    assert model_input.b_seq_len.tolist() == [2, 2, 2, 2, 2]
    assert model_input.b_num_accepted_tokens is None
    assert model_input.total_token_num == 10
    assert model_input.mtp_draft_input_hiddens.shape == (5, 4)


def test_mtp_decode_cuda_graph_key_is_batch_size():
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    # Under MTP there is a single (mtp_step+1)-grouped decode layout, so the graph is keyed by
    # batch size alone — no verify/normal distinction in the key.
    graph = CudaGraph.__new__(CudaGraph)
    graph.mtp_step = 2
    graph.graph = {}
    graph.cuda_graph_batch_sizes = [3, 6, 9, 12]

    state = SimpleNamespace(input_ids=torch.ones(6, dtype=torch.int64))
    assert graph._decode_graph_key(state) == 6
    assert graph.find_closest_graph_batch_size(5) == 6

    graph.graph[6] = "decode graph"
    assert graph.need_capture(6) is False
    assert graph.need_capture(3) is True


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
