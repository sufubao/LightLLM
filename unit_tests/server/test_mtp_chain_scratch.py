from types import SimpleNamespace

import torch

from lightllm.common.basemodel.triton_kernel import gather_token_id
from lightllm.server.router.model_infer.mode_backend.chunked_prefill import impl
from lightllm.server.router.model_infer.mode_backend.chunked_prefill import mtp_fused_decode_graph
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.mtp_fused_decode_graph import (
    MTPFusedDecodeGraph,
)


def test_gather_token_reuses_caller_output(monkeypatch):
    calls = {}

    class Kernel:
        def __getitem__(self, _grid):
            return lambda **kwargs: calls.update(kwargs)

    monkeypatch.setattr(gather_token_id, "_fwd_kernel_gather", Kernel())
    output = torch.empty(4, dtype=torch.int64)

    result = gather_token_id.gather_token(
        torch.empty((2, 4), dtype=torch.int64),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        out=output,
    )

    assert result is output
    assert calls["output"] is output


def test_mtp_chain_scratch_is_reserved_once(monkeypatch):
    allocated_sizes = []
    scratch_gpu = object()
    scratch_cpu = SimpleNamespace(cuda=lambda: scratch_gpu)
    mem_manager = SimpleNamespace(alloc=lambda size: allocated_sizes.append(size) or scratch_cpu)
    monkeypatch.setattr(impl.g_infer_context, "req_manager", SimpleNamespace(mem_manager=mem_manager))

    backend = SimpleNamespace(
        is_mtp_eagle=True,
        mtp_step=3,
        model=SimpleNamespace(req_manager=SimpleNamespace(max_request_num=7)),
    )

    impl.ChunkedPrefillBackend._init_mtp_chain_scratch(backend)

    assert allocated_sizes == [7 * (3 + 1) * (3 - 1)]
    assert backend.mtp_chain_scratch is scratch_gpu


def test_mtp_draft_chain_uses_scratch_and_restores_verify_mapping(monkeypatch):
    batch_size = 4
    verify_mem_indexes = torch.arange(10, 10 + batch_size, dtype=torch.int32)
    scratch = torch.arange(20, 20 + batch_size * 2, dtype=torch.int32)
    original_seq_len = torch.arange(5, 5 + batch_size, dtype=torch.int32)
    forward_mem_indexes = []
    restored = []

    class DraftModel:
        def forward(self, model_input):
            forward_mem_indexes.append(model_input.mem_indexes.clone())
            return SimpleNamespace(
                logits=torch.zeros(batch_size, 1),
                mtp_main_output_hiddens=torch.zeros(batch_size, 1),
            )

    monkeypatch.setattr(
        impl,
        "copy_kv_index_to_req",
        lambda req_to_token, req_idx, seq_len, mem_indexes: restored.append(
            (req_to_token, req_idx.clone(), seq_len.clone(), mem_indexes.clone())
        ),
    )
    monkeypatch.setattr(impl, "mtp_scatter_next_token_ids", lambda **_kwargs: None)

    req_to_token_indexs = object()
    backend = SimpleNamespace(
        mtp_step=3,
        num_mtp_models=1,
        mtp_chain_scratch=scratch,
        draft_models=[DraftModel()],
        model=SimpleNamespace(
            req_manager=SimpleNamespace(
                req_to_token_indexs=req_to_token_indexs,
                req_sampling_params_manager=SimpleNamespace(req_to_next_token_ids=object()),
            )
        ),
        _gen_argmax_token_ids=lambda _output: torch.zeros(batch_size, dtype=torch.int64),
    )
    model_input = SimpleNamespace(
        batch_size=batch_size,
        mem_indexes=verify_mem_indexes,
        b_seq_len=original_seq_len.clone(),
        max_kv_seq_len=20,
        b_req_idx=torch.arange(batch_size, dtype=torch.int32),
    )
    model_output = SimpleNamespace(mtp_main_output_hiddens=torch.zeros(batch_size, 1))

    result = impl.ChunkedPrefillBackend._draft_decode_eagle(
        backend,
        main_model_input=model_input,
        main_model_output=model_output,
        next_token_ids=torch.zeros(batch_size, dtype=torch.int64),
        mtp_accept_len=torch.ones(1, dtype=torch.int32),
        b_req_mtp_start_loc=torch.zeros(1, dtype=torch.int32),
    )

    assert result is None
    assert len(forward_mem_indexes) == 3
    assert torch.equal(forward_mem_indexes[0], verify_mem_indexes)
    assert torch.equal(forward_mem_indexes[1], scratch[:batch_size])
    assert torch.equal(forward_mem_indexes[2], scratch[batch_size:])
    assert torch.equal(model_input.b_seq_len, original_seq_len)
    assert model_input.mem_indexes is verify_mem_indexes
    assert len(restored) == 1
    assert restored[0][0] is req_to_token_indexs
    assert torch.equal(restored[0][2], original_seq_len)
    assert torch.equal(restored[0][3], verify_mem_indexes)


def _make_graph_req(**overrides):
    shm_param = SimpleNamespace(
        exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (1, 1.0)),
        min_new_tokens=1,
    )
    values = dict(
        mtp_step=3,
        sampling_param=SimpleNamespace(shm_param=shm_param, invalid_token_ids=[]),
        generator=None,
        need_out_token_id_statistics=False,
        shm_req=SimpleNamespace(input_len=8),
        get_cur_total_len=lambda: 9,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_mtp_fused_graph_runtime_guards():
    graph = object.__new__(MTPFusedDecodeGraph)
    graph.cuda_graph_batch_sizes = [4]
    graph.graph_max_len_in_batch = 128
    graph.mtp_step = 3
    graph.mtp_size = 4
    graph.backend = SimpleNamespace(decode_mask_func=None)

    assert graph.can_run([_make_graph_req()], max_kv_seq_len=16, batch_size=4)
    assert not graph.can_run([_make_graph_req(mtp_step=2)], max_kv_seq_len=16, batch_size=4)
    assert not graph.can_run([_make_graph_req(generator=object())], max_kv_seq_len=16, batch_size=4)
    assert not graph.can_run([_make_graph_req()], max_kv_seq_len=126, batch_size=4)


def test_mtp_fused_graph_is_disabled_for_rl():
    backend = object.__new__(impl.ChunkedPrefillBackend)
    backend.disable_cudagraph = False
    backend.is_mtp_eagle = True
    backend.num_mtp_models = 1
    backend.classed_req_no_decode = False
    backend.decode_mask_func = None
    backend.enable_decode_microbatch_overlap = False
    backend.args = SimpleNamespace(dp=1, enable_rl=True)

    backend._init_mtp_fused_graph()

    assert backend.mtp_fused_graph is None


def test_decode_mtp_dispatches_to_fused_graph(monkeypatch):
    model_input = SimpleNamespace(batch_size=4, max_kv_seq_len=32)
    run_reqs = [object()] * 4
    calls = []

    monkeypatch.setattr(impl, "prepare_decode_inputs", lambda _decode_reqs: (model_input, run_reqs))
    backend = SimpleNamespace(
        mtp_fused_graph=SimpleNamespace(can_run=lambda **_kwargs: True),
        _decode_mtp_fused=lambda **kwargs: calls.append(kwargs),
    )
    decode_reqs = [object()]
    event_pack = object()

    impl.ChunkedPrefillBackend.decode_mtp(backend, event_pack=event_pack, decode_reqs=decode_reqs)

    assert len(calls) == 1
    assert calls[0] == {
        "event_pack": event_pack,
        "model_input": model_input,
        "run_reqs": run_reqs,
        "decode_reqs": decode_reqs,
    }


def test_mtp_fused_forward_keeps_outputs_referenced(monkeypatch):
    infer_state = SimpleNamespace(
        is_cuda_graph=False,
        b_req_idx=object(),
        b_seq_len=object(),
        mem_index=object(),
        init_some_extra_state=lambda _model: None,
        init_att_state=lambda: None,
    )
    model = SimpleNamespace(
        req_manager=SimpleNamespace(req_to_token_indexs=object()),
        _create_inferstate=lambda _model_input: infer_state,
        _token_forward=lambda state: state.is_cuda_graph,
    )
    monkeypatch.setattr(mtp_fused_decode_graph, "copy_kv_index_to_req", lambda *_args: None)

    graph = object.__new__(MTPFusedDecodeGraph)

    assert graph._forward_in_body(model, object()) is False


def test_mtp_fused_resets_padding_linear_state():
    graph = object.__new__(MTPFusedDecodeGraph)
    graph.backend = SimpleNamespace(is_linear_att_mixed_model=True)
    graph.hold_req_idx = 3
    graph.req_manager = SimpleNamespace(req_to_mtp_state_index=torch.tensor([0, 0, 0, 4]))

    graph._reset_padding_linear_state()

    assert graph.req_manager.req_to_mtp_state_index.tolist() == [0, 0, 0, 0]


def test_mtp_fused_position_delta_clears_stale_rows():
    graph = object.__new__(MTPFusedDecodeGraph)
    graph.b_position_delta = torch.zeros(16, dtype=torch.int32)
    graph.b_position_delta_pin = torch.zeros(16, dtype=torch.int32)
    graph._position_delta_rows = 0

    graph.b_position_delta_pin[:8] = torch.tensor([0, 0, 0, 0, 0, 100, 0, 0])
    graph._flush_position_delta(has_delta=True, batch_size=8)
    assert graph.b_position_delta[5].item() == 100

    graph.b_position_delta_pin[:4].zero_()
    graph._flush_position_delta(has_delta=False, batch_size=4)
    graph.b_position_delta_pin[:8].zero_()
    graph._flush_position_delta(has_delta=False, batch_size=8)

    assert graph.b_position_delta[:8].eq(0).all()


def test_mtp_fused_draft_graph_uses_scratch_and_restores_mapping(monkeypatch):
    batch_size = 4
    verify_mem_indexes = torch.arange(10, 10 + batch_size, dtype=torch.int32)
    scratch = torch.arange(20, 20 + batch_size * 2, dtype=torch.int32)
    original_seq_len = torch.arange(5, 5 + batch_size, dtype=torch.int32)
    forward_mem_indexes = []
    restored = []

    graph = object.__new__(MTPFusedDecodeGraph)
    graph.mtp_step = 3
    graph.mtp_size = 4
    graph.mem_indexes = verify_mem_indexes
    graph.chain_scratch = scratch
    graph.b_req_idx = torch.arange(batch_size, dtype=torch.int32)
    graph.b_seq_len = original_seq_len.clone()
    graph.out_next_token_ids = torch.zeros(batch_size, dtype=torch.int64)
    graph.draft_model = object()
    graph.req_manager = SimpleNamespace(req_to_token_indexs=object())
    graph.sampling_manager = SimpleNamespace(req_to_next_token_ids=object())
    graph._build_model_input = lambda _batch_size: SimpleNamespace(
        input_ids=None,
        mem_indexes=verify_mem_indexes,
        mtp_draft_input_hiddens=None,
    )

    def forward(_model, model_input):
        forward_mem_indexes.append(model_input.mem_indexes.clone())
        return SimpleNamespace(
            logits=torch.zeros(batch_size, 1),
            mtp_main_output_hiddens=torch.zeros(batch_size, 1),
        )

    graph._forward_in_body = forward
    monkeypatch.setattr(
        mtp_fused_decode_graph,
        "copy_kv_index_to_req",
        lambda _req_to_token, _req_idx, seq_len, mem_indexes: restored.append((seq_len.clone(), mem_indexes.clone())),
    )
    monkeypatch.setattr(mtp_fused_decode_graph, "mtp_scatter_next_token_ids", lambda **_kwargs: None)

    graph._run_draft_body(
        batch_size,
        (
            torch.ones(1, dtype=torch.int32),
            torch.ones(batch_size, dtype=torch.int32),
            torch.zeros(batch_size, 1),
        ),
    )

    assert len(forward_mem_indexes) == 3
    assert torch.equal(forward_mem_indexes[0], verify_mem_indexes)
    assert torch.equal(forward_mem_indexes[1], scratch[:batch_size])
    assert torch.equal(forward_mem_indexes[2], scratch[batch_size:])
    assert torch.equal(restored[0][0], original_seq_len)
    assert torch.equal(restored[0][1], verify_mem_indexes)
