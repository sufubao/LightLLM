from lightllm.server.multi_level_kv_cache import CacheTier
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.cache_manager import HybridCacheManager


@pytest.mark.parametrize(
    "hit", ["big", "small", "small_only", "small_no_ancestor", "fallback", "fallback_no_ancestor", "cold", "existing"]
)
def test_restore_prefix(hit):
    events = []
    big = SimpleNamespace(node_prefix_total_len=8, big_page_buffer_idx=1, is_big_page_node=lambda: True)
    small = SimpleNamespace(node_prefix_total_len=12, small_page_buffer_idx=2, is_big_page_node=lambda: False)
    req = SimpleNamespace(
        req_idx=1,
        args=SimpleNamespace(
            linear_att_hash_page_size=4,
            linear_att_page_block_num=4 if hit == "small_only" else 2,
            max_req_total_len=12 if hit == "small_only" else 64,
            page_size=4,
        ),
        sampling_param=SimpleNamespace(disable_prompt_cache=False),
        shm_req=SimpleNamespace(
            hybrid_token_hash_list=SimpleNamespace(get_all=lambda: [1, 2, 3], size=3),
            shm_prompt_ids=SimpleNamespace(arr=list(range(13))),
            prompt_cache_len=0,
        ),
        cur_kv_len=0,
        get_cur_total_len=lambda: 13,
        shared_kv_node=None,
        tail_small_page_buffer_id=None,
        mm_input_spans=[],
        prompt_selected_logprobs=SimpleNamespace(
            copy_capture_slots_if_needed=lambda **kwargs: events.append("logprobs")
        ),
    )
    if hit == "existing":
        req.cur_kv_len = 8
        req.shared_kv_node = big
        req.shm_req.prompt_cache_len = 4
    if hit.endswith("no_ancestor"):
        small.node_prefix_total_len = 4
    source = None if hit == "cold" else big if hit == "big" else small

    def match(key, **kwargs):
        assert kwargs["update_refs"]
        assert key.tolist() == list(range(12))
        events.append("match")
        count = source.node_prefix_total_len if source is not None else 0
        return source, count, torch.arange(count, dtype=torch.int32)

    def restore(req_idx, conv, ssm):
        assert req_idx == req.req_idx
        expected_slot = 1 if hit in {"big", "fallback"} else 2
        assert conv == ssm == expected_slot
        if hit in {"small", "small_no_ancestor"}:
            assert req.shared_kv_node is None
        events.append("restore")

    manager = HybridCacheManager(
        SimpleNamespace(page_size=1),
        2,
        16,
        SimpleNamespace(
            restore_checkpoint=lambda req_idx, length, pool, slot: restore(req_idx, *pool.get_state_cache(slot)),
            reset=lambda req_idx: events.append("reset"),
            conv_state=None,
            ssm_state=None,
        ),
        SimpleNamespace(reset=lambda req_idx: events.append("tail"), state=None),
        device="cpu",
    )
    manager.small_page_buffers = SimpleNamespace(
        get_state_cache=lambda slot: (slot, slot), has_checkpoint=lambda slot, length: slot is not None
    )
    mem = SimpleNamespace(
        allocator=SimpleNamespace(can_use_mem_size=4 if hit.startswith("fallback") else 8),
        alloc=lambda need_size: torch.arange(20, 20 + need_size, dtype=torch.int32),
        operator=SimpleNamespace(copy_mem_to_mem=lambda src, dst: events.append("copy")),
    )
    manager.mem_manager = mem
    manager.big_page_buffers = SimpleNamespace(
        get_state_cache=lambda slot: (slot, slot), has_checkpoint=lambda slot, length: slot is not None
    )
    requests = SimpleNamespace(
        mem_manager=mem, cache_manager=SimpleNamespace(req_to_token_indexs=torch.full((2, 16), -1, dtype=torch.int32))
    )
    radix = SimpleNamespace(
        match_prefix=match,
        mem_manager=mem,
        get_tree_total_tokens_num=lambda: 12,
        get_refed_tokens_num=lambda: 12,
        free_radix_cache_to_get_enough_token=lambda **kwargs: events.append("reserve"),
        deref_to_first_big_page_node=lambda **kwargs: None if hit.endswith("no_ancestor") else big,
    )
    manager.args = req.args
    manager.req_to_token_indexs = requests.cache_manager.req_to_token_indexs
    manager.radix_cache = radix
    manager.restore_prefix(req)

    if hit == "existing":
        assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == 8
        assert req.shared_kv_node is big
        assert req.shm_req.prompt_cache_len == 4
        assert not events
        return
    length = (
        0
        if hit in {"cold", "fallback_no_ancestor"}
        else 8
        if hit in {"big", "fallback"}
        else 4
        if hit == "small_no_ancestor"
        else 12
    )
    assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == length
    assert req.shm_req.prompt_cache_len == length
    assert req.shared_kv_node is (
        None if hit in {"cold", "small_no_ancestor", "fallback_no_ancestor"} else small if hit == "small_only" else big
    )
    if hit == "small_no_ancestor":
        assert requests.cache_manager.req_to_token_indexs[1, :4].tolist() == list(range(20, 24))
        assert events == ["match", "reserve", "copy", "logprobs", "restore", "tail"]
    elif hit == "small":
        assert requests.cache_manager.req_to_token_indexs[1, :12].tolist() == list(range(8)) + list(range(20, 24))
        assert events == ["match", "reserve", "copy", "logprobs", "restore", "tail"]
    elif hit in {"cold", "fallback_no_ancestor"}:
        assert events == ["match", "reset", "tail"]
        assert (requests.cache_manager.req_to_token_indexs == -1).all()
    else:
        assert requests.cache_manager.req_to_token_indexs[1, :length].tolist() == list(range(length))
        assert events == ["match", "restore", "tail"]


def test_memory_binding_rejects_another_pool():
    manager = HybridCacheManager(SimpleNamespace(page_size=1), 2, 16, device="cpu")
    mem = SimpleNamespace()
    manager.bind_mem_manager(mem)
    manager.bind_mem_manager(mem)
    with pytest.raises(ValueError, match="different memory manager"):
        manager.bind_mem_manager(SimpleNamespace())
    assert manager.mem_manager is mem


@pytest.mark.parametrize("kv_type", ["None", "fp8kv_sph", "fp8kv_spt"])
def test_qwen_runtime_and_cache_share_config(monkeypatch, kv_type):
    from lightllm.models.qwen3next import model as qwen
    from lightllm.common.state_cache_manager import linear_att
    from lightllm.server.core.objs.start_args_type import StartArgs

    args = StartArgs(tp=4, dp=1, data_type="bfloat16", llm_kv_type=kv_type)
    monkeypatch.setattr(qwen, "get_env_start_args", lambda: args)
    model = object.__new__(qwen.Qwen3NextTpPartModel)
    model.args = args
    model.config = dict(
        n_layer=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=8,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
    )
    model.max_req_num = 2
    model.tp_world_size_ = 4
    model.num_kv_heads = 1
    model.data_type = torch.bfloat16
    model.max_total_token_num = 32
    model.mem_fraction = 0.9
    events = []

    def runtime(size, config, args):
        events.append("runtime")
        return SimpleNamespace(linear_config=config, conv_state=object(), ssm_state=object())

    def memory(**kwargs):
        events.append("memory")
        assert events == ["runtime", "memory"]
        return SimpleNamespace(
            **kwargs, kv_buffer=object(), set_kv_layer_mapping=lambda mapping: events.append(mapping)
        )

    monkeypatch.setattr(linear_att, "get_added_mtp_kv_layer_num", lambda: 1)
    monkeypatch.setattr(qwen, "LinearStateManager", runtime)
    for name in (
        "MemoryManager",
        "FP8StaticPerHeadQuantMemManager",
        "FP8StaticPerTensorQuantMemManager",
    ):
        monkeypatch.setattr(qwen, name, memory)
    create_cache = qwen.HybridCacheManager
    monkeypatch.setattr(qwen, "HybridCacheManager", lambda *args, **kwargs: create_cache(*args, **kwargs, device="cpu"))
    model.req_manager = SimpleNamespace(cache_manager=model._create_cache_manager(16))
    model._init_mem_manager()
    pools = []

    def checkpoint_pool(size, config, keep_num):
        pools.append((size, config, keep_num))
        return SimpleNamespace(size=size)

    monkeypatch.setattr(
        model.req_manager.cache_manager.state_manager,
        "create_checkpoint_pool",
        lambda size, keep_num=0: checkpoint_pool(size, model.linear_config, keep_num),
        raising=False,
    )
    monkeypatch.setattr(
        model.req_manager.cache_manager.state_manager, "create_cpu_transfer", lambda *args: None, raising=False
    )
    model.req_manager.cache_manager.bind_mem_manager(model.mem_manager)
    pool = model.req_manager.cache_manager.big_page_buffers
    model.req_manager.cache_manager.bind_mem_manager(model.mem_manager)
    assert model.req_manager.cache_manager.big_page_buffers is pool
    assert len(pools) == 1
    assert pools[0][1] is model.linear_config
    assert pools[0][2] == 2
    assert not hasattr(model.mem_manager, "big_page_buffers")
    assert not hasattr(model.mem_manager, "req_to_conv_state")
    config = model.linear_config
    assert config is model.req_manager.cache_manager.linear_state_manager.linear_config
    assert events[-1] == config.get_full_att_kv_layer_mapping()
    assert config.num_linear_k_heads == 1
    assert config.num_linear_v_heads == 2
    assert config.full_att_dtype == (torch.bfloat16 if kv_type == "None" else torch.uint8)
    assert config.get_full_att_kv_layer_index(8) == 2


@pytest.mark.parametrize("mismatch", [False, True])
def test_context_uses_the_request_managers_cache(mismatch):
    from lightllm.common.cache_manager import CacheManager
    from lightllm.server.router.model_infer.infer_batch import InferenceContext

    config = SimpleNamespace(page_size=1, output_constraint_mode="none")
    mem = SimpleNamespace()
    cache = CacheManager(config, 1, 4, device="cpu")
    cache.bind_mem_manager(mem)
    radix = SimpleNamespace(mem_manager=object() if mismatch else mem)
    if mismatch:
        with pytest.raises(ValueError, match="share the memory manager"):
            cache.bind_radix_cache(radix)
        return
    cache.bind_radix_cache(radix)
    requests = SimpleNamespace(cache_manager=cache, req_sampling_params_manager=None)
    context = InferenceContext()
    context.register(backend=SimpleNamespace(), req_manager=requests, shm_req_manager=None, vocab_size=1)
    assert context.req_manager.cache_manager is cache
    assert not hasattr(context, "cache_manager")


@pytest.mark.parametrize("pool", ["big_page_buffers", "small_page_buffers"])
def test_radix_cannot_use_a_different_checkpoint_pool(pool):
    cache_manager = HybridCacheManager(SimpleNamespace(page_size=1), 2, 16, SimpleNamespace(), device="cpu")
    cache_manager.mem_manager = object()
    cache_manager.big_page_buffers = object()
    cache_manager.small_page_buffers = object()
    radix = SimpleNamespace(
        mem_manager=cache_manager.mem_manager,
        big_page_buffers=cache_manager.big_page_buffers,
        small_page_buffers=cache_manager.small_page_buffers,
    )
    cache_manager.validate_radix_cache(radix)
    setattr(radix, pool, object())
    with pytest.raises(ValueError, match="checkpoint pools"):
        cache_manager.validate_radix_cache(radix)


def test_checkpoint_pool_cannot_be_replaced():
    cache_manager = HybridCacheManager(SimpleNamespace(page_size=1), 2, 16, device="cpu")
    pool = cache_manager.small_page_buffers = SimpleNamespace(size=8)
    assert cache_manager.create_small_page_cache_manager(8) is pool
    with pytest.raises(ValueError, match="replace"):
        cache_manager.create_small_page_cache_manager(16)


def test_full_attention_uses_the_cache_manager_for_restore_and_release():
    from lightllm.common.cache_manager import CacheManager

    cache_manager = CacheManager(SimpleNamespace(page_size=1), 2, 16, device="cpu")
    assert cache_manager.linear_state_manager is cache_manager.indexer_state_manager is None
    node = SimpleNamespace(node_prefix_total_len=4)
    references = []
    radix = SimpleNamespace(
        match_prefix=lambda *args, **kwargs: (node, 4, torch.tensor([10, 11, 12, 13])),
        insert=lambda key, value: (6, node),
        dec_node_ref_counter=references.append,
    )
    reqs = SimpleNamespace(cache_manager=SimpleNamespace(req_to_token_indexs=torch.arange(20, 28).reshape(1, 8)))
    req = SimpleNamespace(
        req_idx=0,
        args=SimpleNamespace(page_size=2),
        cur_kv_len=0,
        hold_kv_len=0,
        shared_kv_node=None,
        mm_input_spans=[],
        sampling_param=SimpleNamespace(disable_prompt_cache=False),
        shm_req=SimpleNamespace(shm_prompt_ids=SimpleNamespace(arr=list(range(9)))),
        get_cur_total_len=lambda: 9,
        get_input_token_ids=lambda: list(range(9)),
    )
    cache_manager.args = req.args
    cache_manager.req_to_token_indexs = reqs.cache_manager.req_to_token_indexs
    cache_manager.radix_cache = radix
    cache_manager.restore_prefix(req)
    assert req.cur_kv_len == req.hold_kv_len == req.shm_req.prompt_cache_len == 4
    req.cur_kv_len, req.hold_kv_len = 7, 8
    cache_manager.args = req.args
    cache_manager.req_to_token_indexs = reqs.cache_manager.req_to_token_indexs
    cache_manager.radix_cache = radix
    req.cache_tiers = (CacheTier.GPU,) if True else ()
    released = cache_manager._release_request_cache(req)
    assert [part.tolist() for part in released] == [[26, 27], [24, 25]]
    assert references == [node]
    assert req.shared_kv_node is None


@pytest.mark.parametrize("local_prefix", [0, 8])
def test_peer_runtime_without_intermediate_checkpoints_is_not_published(local_prefix):
    from unittest.mock import Mock

    cache_manager = HybridCacheManager(SimpleNamespace(page_size=1), 2, 16, device="cpu")
    cache_manager.big_page_buffers = SimpleNamespace(
        free_state_cache=Mock(), has_checkpoint=lambda slot, length: slot is not None
    )
    cache_manager.small_page_buffers = SimpleNamespace(
        free_state_cache=Mock(), has_checkpoint=lambda slot, length: slot is not None
    )
    shared = SimpleNamespace(node_prefix_total_len=local_prefix, big_page_buffer_idx=0) if local_prefix else None
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=28,
        hold_kv_len=28,
        shared_kv_node=shared,
        hybrid_cache_len=28,
        hybrid_len_to_big_page_id={24: 3},
        tail_small_page_buffer_id=7,
    )
    args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2)
    requests = SimpleNamespace(cache_manager=SimpleNamespace(req_to_token_indexs=torch.arange(28).reshape(1, 28)))
    radix = SimpleNamespace(insert=Mock(), dec_node_ref_counter=Mock())
    cache_manager.args = args
    assert cache_manager.get_cacheable_prefix_len(req) == local_prefix
    cache_manager.args = args
    cache_manager.req_to_token_indexs = requests.cache_manager.req_to_token_indexs
    cache_manager.radix_cache = radix
    req.cache_tiers = (CacheTier.GPU,) if True else ()
    freed = cache_manager._release_request_cache(req)
    assert torch.cat(freed).tolist() == list(range(local_prefix, 28))
    radix.insert.assert_not_called()
    cache_manager.big_page_buffers.free_state_cache.assert_called_once_with([3])
    cache_manager.small_page_buffers.free_state_cache.assert_called_once_with([7])
    assert not req.hybrid_len_to_big_page_id and req.tail_small_page_buffer_id is None


def test_release_keeps_only_the_checkpoint_complete_prefix():
    from unittest.mock import Mock

    cache_manager = HybridCacheManager(SimpleNamespace(page_size=1), 2, 16, device="cpu")
    cache_manager.big_page_buffers = SimpleNamespace(
        free_state_cache=Mock(), has_checkpoint=lambda slot, length: slot is not None
    )
    cache_manager.small_page_buffers = SimpleNamespace(
        free_state_cache=Mock(), has_checkpoint=lambda slot, length: slot is not None
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=28,
        hold_kv_len=28,
        shared_kv_node=None,
        hybrid_cache_len=28,
        hybrid_len_to_big_page_id={8: 2, 24: 3},
        tail_small_page_buffer_id=7,
        get_input_token_ids=lambda: list(range(28)),
        shm_req=SimpleNamespace(hybrid_token_hash_list=SimpleNamespace(get_all=lambda: list(range(7)))),
    )
    args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2)
    requests = SimpleNamespace(cache_manager=SimpleNamespace(req_to_token_indexs=torch.arange(28).reshape(1, 28)))

    def insert(key, value, **kwargs):
        assert len(key) == len(value) == 8
        assert kwargs["len_to_big_page_id"] == {8: 2}
        kwargs["len_to_big_page_id"].clear()
        return 0, None

    cache_manager.args = args
    assert cache_manager.get_cacheable_prefix_len(req) == 8
    cache_manager.args = args
    cache_manager.req_to_token_indexs = requests.cache_manager.req_to_token_indexs
    cache_manager.radix_cache = SimpleNamespace(insert=insert)
    req.cache_tiers = (CacheTier.GPU,) if True else ()
    freed = cache_manager._release_request_cache(req)
    assert torch.cat(freed).tolist() == list(range(8, 28))
    cache_manager.big_page_buffers.free_state_cache.assert_called_once_with([3])
    cache_manager.small_page_buffers.free_state_cache.assert_called_once_with([7])


def test_dp_restore_copies_linear_state_and_indexer_tail():
    from lightllm.common.linear_state import LinearStateManager

    state = LinearStateManager.__new__(LinearStateManager)
    state.linear_config = SimpleNamespace(get_conv_state_shape=lambda: (4, 3))
    state.mtp_step = 2
    state._req_to_mtp_state_index = torch.full((2,), 2, dtype=torch.int32)
    state.conv_state = SimpleNamespace(buffer=torch.zeros(1, 2, 4, 5))
    state.ssm_state = SimpleNamespace(buffer=torch.zeros(1, 6, 4, 4))
    indexer = SimpleNamespace(state=SimpleNamespace(buffer=torch.zeros(1, 2, 6, 4)))
    cache_manager = HybridCacheManager(SimpleNamespace(page_size=1), 2, 16, state, indexer, device="cpu")
    peer = SimpleNamespace(
        mtp_step=2,
        req_to_conv_state=SimpleNamespace(buffer=torch.randn(1, 2, 4, 5)),
        req_to_ssm_state=SimpleNamespace(buffer=torch.randn(1, 6, 4, 4)),
        req_to_indexer_tail=SimpleNamespace(buffer=torch.randn(1, 2, 6, 4)),
    )
    cache_manager.restore_peer_state(0, 11, peer, 1)
    torch.testing.assert_close(state.conv_state.buffer[:, 0, ..., :3], peer.req_to_conv_state.buffer[:, 1, ..., :3])
    torch.testing.assert_close(state.ssm_state.buffer[:, 0], peer.req_to_ssm_state.buffer[:, 3])
    torch.testing.assert_close(indexer.state.buffer[:, 0], peer.req_to_indexer_tail.buffer[:, 1])
    assert not state.conv_state.buffer[:, 0, ..., 3:].any()
    assert state._req_to_mtp_state_index.tolist() == [0, 2]


def test_checkpoint_lengths(monkeypatch):
    from unittest.mock import Mock
    from lightllm.common.state_cache_manager import StateCacheManager

    class Pool(StateCacheManager):
        def get_state_cache(self, buffer_idx):
            return buffer_idx

    state = SimpleNamespace(save_state=Mock(), restore_checkpoint=Mock())
    cache_manager = HybridCacheManager(SimpleNamespace(page_size=1), 2, 16, state_manager=state, device="cpu")
    pool = cache_manager.big_page_buffers = Pool(4, keep_num=1)
    cache_manager.small_page_buffers = Pool(1)
    slot = pool.alloc_one_state_cache()
    with pytest.raises(AssertionError):
        cache_manager.restore_state(0, pool, slot, 8)
    cache_manager.save_state(0, slot, pool, 8)
    with pytest.raises(AssertionError):
        cache_manager.restore_state(0, pool, slot, 16)
    cache_manager.restore_state(0, pool, slot, 8)
    state.restore_checkpoint.assert_called_once_with(0, 8, pool, slot)
    pool.free_state_cache([slot])
    assert not pool.has_checkpoint(slot, 8)

    args = SimpleNamespace(cpu_cache_token_page_size=8, linear_att_hash_page_size=4, linear_att_page_block_num=2)
    cache_manager.args = args
    cache_manager.cpu_cache_transfer = SimpleNamespace(
        load=Mock(), get_checkpoint_lens=lambda pages, client: [None, 16][: len(pages)]
    )
    cache_manager.cpu_cache_client = None
    assert cache_manager.get_cpu_cache_match_page_num([0], [8]) == 0
    cache_manager.cpu_cache_client = None
    assert cache_manager.get_cpu_cache_match_page_num([0, 1], [8, 16]) == 2
    req = SimpleNamespace(req_idx=0, cur_kv_len=16, mm_input_spans=[], hybrid_len_to_big_page_id={})
    cache_manager.cpu_cache_client = None
    cache_manager.load_cpu_cache_to_gpu(torch.arange(16), torch.arange(2), req)
    first, last = [req.hybrid_len_to_big_page_id[length] for length in (8, 16)]
    assert not pool.has_checkpoint(first, 8)
    assert pool.has_checkpoint(last, 16)
    state.restore_checkpoint.assert_called_with(0, 16, pool, last)
    req.hybrid_cache_len = req.cur_kv_len
    req.tail_small_page_buffer_id = req.shared_kv_node = None
    cache_manager.args = args
    assert cache_manager.get_cacheable_prefix_len(req) == 16
    req.cur_kv_len = 8
    cache_manager.args = args
    assert cache_manager.get_cacheable_prefix_len(req) == 0
    pool.clear_to_init_state()
    assert pool.checkpoint_lens == [None] * pool.size
