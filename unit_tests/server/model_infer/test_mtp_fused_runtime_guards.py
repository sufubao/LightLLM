from types import SimpleNamespace

import pytest

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

    backend._init_mtp_fused_graph()

    assert backend.mtp_fused_graph is None
