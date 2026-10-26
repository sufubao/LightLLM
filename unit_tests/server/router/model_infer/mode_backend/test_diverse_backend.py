from types import SimpleNamespace

import torch

from lightllm.common.cache_manager import CacheManager

# 沿用生产代码的模块加载顺序，避免独立运行本测试文件时形成循环导入。
from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.diverse_backend.impl import DiversehBackend


class _FakeRadixCache:
    def __init__(self):
        self.dec_nodes = []
        self.add_nodes = []

    def dec_node_ref_counter(self, node):
        self.dec_nodes.append(node)

    def add_node_ref_counter(self, node):
        self.add_nodes.append(node)


def test_diverse_copy_skips_slaves_waiting_for_overlap_cleanup():
    active_slave = SimpleNamespace(filter_mark=False)
    filtered_slave = SimpleNamespace(filter_mark=True)
    master_req = SimpleNamespace(slave_reqs=[active_slave, filtered_slave])
    backend = DiversehBackend.__new__(DiversehBackend)

    batch_idx, run_reqs = backend._diverse_copy(
        master_reqs=[master_req],
        b_prefill_has_out=[True],
    )

    assert batch_idx == [0, 0]
    assert run_reqs == [master_req, active_slave]


def test_copy_master_kv_updates_slave_hold_kv_len():
    req_to_token_indexs = torch.full((2, 8), -1, dtype=torch.int32)
    req_to_token_indexs[0, :4] = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    old_shared_node = object()
    new_shared_node = object()
    master_req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=4,
        cur_output_len=1,
        shared_kv_node=new_shared_node,
    )
    slave_req = SimpleNamespace(
        req_idx=1,
        cur_kv_len=2,
        hold_kv_len=2,
        cur_output_len=0,
        shared_kv_node=old_shared_node,
        related_master_req=master_req,
        shm_req=SimpleNamespace(input_len=4),
        args=SimpleNamespace(page_size=1),
    )
    backend = DiversehBackend.__new__(DiversehBackend)
    cache_manager = CacheManager(SimpleNamespace(page_size=1), 1, 8, device="cpu")
    cache_manager.req_to_token_indexs = req_to_token_indexs
    backend.model = SimpleNamespace(
        req_manager=SimpleNamespace(cache_manager=cache_manager),
    )
    backend.radix_cache = cache_manager.radix_cache = _FakeRadixCache()
    backend.is_master_in_dp = False

    backend._copy_master_req_to_slave_req(slave_req)

    assert req_to_token_indexs[1, :4].tolist() == [10, 11, 12, 13]
    assert slave_req.shared_kv_node is new_shared_node
    assert slave_req.cur_kv_len == 4
    assert slave_req.hold_kv_len == 4
    assert InferReq._kv_cache_alloc_need(slave_req, target_kv_len=5) == 1
    assert backend.radix_cache.dec_nodes == [old_shared_node]
    assert backend.radix_cache.add_nodes == [new_shared_node]


def test_publish_share_and_release_preserves_cached_prefix(monkeypatch, tmp_path):
    from lightllm.common import cache_manager as cache_module
    from lightllm.server.multi_level_kv_cache import CacheTier
    from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
    from lightllm.utils import shm_utils

    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: tmp_path.name)
    monkeypatch.setattr(cache_module, "custom_cat", torch.cat)
    freed = []
    memory = SimpleNamespace(
        page_size=1, allocator=SimpleNamespace(can_use_mem_size=26), free=lambda indexes: freed.extend(indexes.tolist())
    )
    cache = CacheManager(SimpleNamespace(page_size=1), 2, 8, device="cpu")
    cache.bind_mem_manager(memory)
    radix = RadixCache(32, 0, mem_manager=memory, page_size=1)
    cache.bind_radix_cache(radix)
    tokens = torch.arange(1, 7)
    radix.insert(tokens[:4], torch.arange(100, 104))
    shared, _, _ = radix.match_prefix(tokens[:2], update_refs=True)
    old_slave, _, _ = radix.match_prefix(tokens[:2], update_refs=True)
    cache.req_to_token_indexs[0, :6] = torch.tensor([100, 101, 12, 13, 14, 15])
    master = SimpleNamespace(
        req_idx=0,
        cur_kv_len=6,
        hold_kv_len=6,
        shared_kv_node=shared,
        get_input_token_ids=lambda: tokens.tolist(),
        cache_tiers=(CacheTier.GPU,),
        shm_req=SimpleNamespace(shm_cur_kv_len=6),
    )
    slave = SimpleNamespace(
        req_idx=1,
        cur_kv_len=2,
        hold_kv_len=2,
        shared_kv_node=old_slave,
        get_input_token_ids=lambda: tokens.tolist(),
        cache_tiers=(CacheTier.GPU,),
        shm_req=SimpleNamespace(shm_cur_kv_len=2),
    )
    cache.publish_request_prefix(master)
    assert freed == [12, 13]
    cache.share_request_prefix(master, slave)
    assert cache.req_to_token_indexs[1, :6].tolist() == [100, 101, 102, 103, 14, 15]
    assert slave.hold_kv_len == 6 and slave.shared_kv_node is master.shared_kv_node
    cache.req_to_token_indexs[1, 6:8] = torch.tensor([20, 21])
    slave.hold_kv_len = 8
    cache.release_request_caches([master, slave])
    assert freed == [12, 13, 20, 21]
    assert radix.get_refed_tokens_num() == 0 and radix.get_tree_total_tokens_num() == 6
    assert master.shared_kv_node is slave.shared_kv_node is None
    cache.flush_cache()
    assert sorted(freed) == [12, 13, 14, 15, 20, 21, 100, 101, 102, 103]
