from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.common.req_manager.linear_att import ReqManagerForMamba
from lightllm.server.core.objs import FinishStatus
from lightllm.server.multi_level_kv_cache import CacheTier
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.model_infer import infer_batch
from lightllm.server.router.model_infer.infer_batch import InferReq, InferenceContext
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_impl
from lightllm.utils import shm_utils


@pytest.mark.parametrize("transfer_failed", [False, True])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
def test_pd_hybrid_shared_kv_survives_completion_and_transfer_failure(monkeypatch, transfer_failed, device):
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: "test_pd_hybrid_kv_cache")
    cache = RadixCache(128, 0, page_size=4)
    table = torch.full((2, 16), -1, dtype=torch.int32, device=device)
    table[0, :12] = torch.arange(12, device=device)
    initialized = []
    context = InferenceContext()
    context.args = SimpleNamespace(page_size=4)
    context.is_hybrid_att_model = True
    context.use_hybrid_checkpoint_cache = False
    context.radix_cache = cache
    context.req_manager = SimpleNamespace(
        req_to_token_indexs=table,
        init_hybrid_attention_state=lambda req: initialized.append(req.req_idx),
    )
    monkeypatch.setattr(infer_batch, "g_infer_context", context)
    monkeypatch.setattr(decode_impl, "g_infer_context", context)

    def request(index):
        return SimpleNamespace(
            req_idx=index,
            req_id=index,
            args=context.args,
            cur_kv_len=10,
            hold_kv_len=12,
            cache_tiers=(CacheTier.GPU,),
            shared_kv_node=None,
            get_input_token_ids=lambda: np.arange(12),
            get_cur_total_len=lambda: 12,
            sampling_param=SimpleNamespace(disable_prompt_cache=False),
            shm_req=SimpleNamespace(
                shm_cur_kv_len=10, prompt_cache_len=0, shm_prompt_ids=SimpleNamespace(arr=np.arange(12))
            ),
        )

    first = request(0)
    released = []
    context.free_a_req_mem(released, first)
    assert torch.cat([x.cpu() for x in released]).tolist() == [8, 9, 10, 11]
    assert cache.get_tree_total_tokens_num() == 8

    second = request(1)
    second.cur_kv_len = second.hold_kv_len = 0
    InferReq._match_radix_cache(second)
    assert initialized == [1]
    assert second.cur_kv_len == second.hold_kv_len == second.shm_req.prompt_cache_len == 8
    assert table[1, :8].tolist() == list(range(8))
    assert cache.get_refed_tokens_num() == 8
    with pytest.raises(AssertionError):
        cache.evict(8, lambda indexes: pytest.fail("referenced KV was evicted"))

    second.cur_kv_len = 10
    second.hold_kv_len = 12
    table[1, 8:12] = torch.arange(16, 20, device=device)
    released = []
    if transfer_failed:
        second.pd_task_num = second.pd_task_failed_num = 1
        second.pd_task_success_num = 0
        second.infer_aborted = False
        second.finish_status = FinishStatus()
        context.requests_mapping = {second.req_id: second}
        backend = decode_impl.PDDecodeNode.__new__(decode_impl.PDDecodeNode)
        backend.is_master_in_dp = False
        backend.model = SimpleNamespace(
            req_manager=context.req_manager,
            mem_manager=SimpleNamespace(free=lambda indexes: released.append(indexes)),
        )
        assert backend._filter_not_ready_reqs([second.req_id]) == [second]
        assert second.cur_kv_len == second.hold_kv_len == 8
    context.free_a_req_mem(released, second)
    assert torch.cat([x.cpu() for x in released]).tolist() == [16, 17, 18, 19]
    assert cache.get_refed_tokens_num() == 0
    assert cache.get_tree_total_tokens_num() == 8
    evicted = []
    cache.evict(8, evicted.append)
    assert torch.cat(evicted).tolist() == list(range(8))


@pytest.mark.parametrize("run_mode,checkpoint_cache", [("prefill", True), ("normal", True), ("decode", False)])
def test_hybrid_checkpoint_cache_is_preserved_outside_pd_decode(monkeypatch, run_mode, checkpoint_cache):
    manager = ReqManagerForMamba.__new__(ReqManagerForMamba)
    manager.req_sampling_params_manager = SimpleNamespace()
    monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: SimpleNamespace(run_mode=run_mode))
    context = InferenceContext()
    context.register(SimpleNamespace(), manager, None, None, 32)
    assert context.is_hybrid_att_model
    assert context.use_hybrid_checkpoint_cache is checkpoint_cache
