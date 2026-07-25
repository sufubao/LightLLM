from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention import FlashInferAttBackend, MlaFlashInferAttBackend
from lightllm.server.router.model_infer.mode_backend.chunked_prefill import impl
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.mtp_fused_decode_graph import (
    MTPFusedDecodeGraph,
)


def test_can_run_does_not_require_return_logprobs():
    graph = object.__new__(MTPFusedDecodeGraph)
    graph.cuda_graph_batch_sizes = [4]
    graph.graph_max_len_in_batch = 128
    graph.mtp_step = 3
    graph.backend = SimpleNamespace(decode_mask_func=None)

    shm_param = SimpleNamespace(
        exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (1, 1.0)),
        min_new_tokens=1,
    )
    req = SimpleNamespace(
        mtp_step=3,
        sampling_param=SimpleNamespace(shm_param=shm_param, invalid_token_ids=[]),
        generator=None,
        need_out_token_id_statistics=False,
        shm_req=SimpleNamespace(input_len=8),
        get_cur_total_len=lambda: 9,
    )

    assert graph.can_run(decode_reqs=[req], max_kv_seq_len=16, batch_size=4)


def test_fused_graph_resets_shared_padding_linear_state():
    graph = object.__new__(MTPFusedDecodeGraph)
    graph.backend = SimpleNamespace(is_linear_att_mixed_model=True)
    graph.hold_req_idx = 3
    graph.req_manager = SimpleNamespace(req_to_mtp_state_index=torch.tensor([0, 0, 0, 4]))

    graph._reset_padding_linear_state()

    assert graph.req_manager.req_to_mtp_state_index.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("proposal_step", "runtime_step", "expected"),
    [
        (4, 4, "mtp"),
        (4, 2, "mtp"),
        (2, 4, "transition"),
        (0, 1, "transition"),
    ],
)
def test_select_mtp_profile(proposal_step, runtime_step, expected):
    reqs = [SimpleNamespace(mtp_proposal_step=proposal_step) for _ in range(2)]

    assert impl.select_mtp_profile(reqs, runtime_mtp_step=runtime_step) == expected


def test_dynamic_mtp_can_dispatch_dense_plan():
    backend = object.__new__(impl.ChunkedPrefillBackend)
    backend.mtp_step = 4
    backend._last_mtp_profile = None
    backend._mtp_profile_counts = {"dense": 0, "transition": 0, "mtp": 0}
    backend._get_selected_runtime_mtp_step = lambda: 0
    calls = []
    backend._decode_transition_mtp_profile = lambda event_pack, decode_reqs, runtime_mtp_step: calls.append(
        ("decode", runtime_mtp_step, len(decode_reqs))
    )
    backend._mark_mtp_plan_step = lambda: calls.append(("mark",))

    backend.decode_mtp(
        event_pack=object(),
        decode_reqs=[
            SimpleNamespace(mtp_proposal_step=3),
            SimpleNamespace(mtp_proposal_step=0),
        ],
    )

    assert calls == [("decode", 0, 2), ("mark",)]
    assert backend._mtp_profile_counts["dense"] == 1


@pytest.mark.parametrize(
    ("classed_req_no_decode", "decode_mask_func"),
    [
        (True, None),
        (False, object()),
    ],
)
def test_fused_graph_is_disabled_when_backend_cannot_decode(monkeypatch, classed_req_no_decode, decode_mask_func):
    backend = object.__new__(impl.ChunkedPrefillBackend)
    backend.is_mtp_eagle = True
    backend.num_mtp_models = 1
    backend.classed_req_no_decode = classed_req_no_decode
    backend.decode_mask_func = decode_mask_func

    monkeypatch.setattr(impl, "get_env_start_args", lambda: SimpleNamespace(disable_cudagraph=False))
    monkeypatch.setattr(backend, "_init_mtp_chain_scratch", lambda: None)

    backend._init_mtp_fused_graph()

    assert backend.mtp_fused_graph is None


@pytest.mark.parametrize(
    ("model_index", "backend_type"),
    [
        (0, FlashInferAttBackend),
        (1, MlaFlashInferAttBackend),
    ],
)
def test_fused_graph_is_disabled_for_main_or_draft_flashinfer(monkeypatch, model_index, backend_type):
    backend = object.__new__(impl.ChunkedPrefillBackend)
    backend.is_mtp_eagle = True
    backend.num_mtp_models = 1
    backend.classed_req_no_decode = False
    backend.decode_mask_func = None
    backend.enable_decode_microbatch_overlap = False
    backend.args = SimpleNamespace(dp=1)

    models = [
        SimpleNamespace(decode_att_backend=object(), decode_att_backend1=None),
        SimpleNamespace(decode_att_backend=object(), decode_att_backend1=None),
    ]
    models[model_index].decode_att_backend = object.__new__(backend_type)
    backend.model = models[0]
    backend.draft_models = [models[1]]

    monkeypatch.setattr(impl, "get_env_start_args", lambda: SimpleNamespace(disable_cudagraph=False))
    monkeypatch.setattr(backend, "_init_mtp_chain_scratch", lambda: None)

    backend._init_mtp_fused_graph()

    assert backend.mtp_fused_graph is None
