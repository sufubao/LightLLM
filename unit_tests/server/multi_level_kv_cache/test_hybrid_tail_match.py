import time
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.server.core.objs import Req
from lightllm.server.core.objs import req as req_impl
from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.server.multi_level_kv_cache.manager import MultiLevelKVCacheManager
from lightllm.utils.kv_cache_utils import compute_token_list_hash


@pytest.fixture
def cache_match(monkeypatch):
    args = SimpleNamespace(
        cpu_cache_token_page_size=1024,
        linear_att_hash_page_size=256,
        linear_att_page_block_num=4,
        diverse_mode=False,
    )
    monkeypatch.setattr(req_impl, "get_env_start_args", lambda: args)
    manager = MultiLevelKVCacheManager.__new__(MultiLevelKVCacheManager)
    manager.args = args
    manager.is_hybrid_att_model = True
    manager.only_cpu_cache_enable = True
    manager.cpu_cache_time_out = 10
    manager.send_to_router = Mock()
    manager.shm_req_manager = Mock()
    ready_pages = {}
    refs = Counter()

    def query(hash_key):
        page = ready_pages.get(hash_key)
        if page is not None:
            refs[page] += 1
        return page, page is not None

    manager.cpu_cache_client = Mock()
    manager.cpu_cache_client.query_one_page.side_effect = query
    manager.cpu_cache_client.check_allpages_ready.return_value = True

    def make_request(length, stored_ends):
        request = Req()
        request.input_len = length
        request.sample_params.prompt_logprobs = -1
        hashes = compute_token_list_hash(list(range(length)), args.linear_att_hash_page_size)
        request.hybrid_token_hash_list.fill(hashes)
        page_hashes, page_lens = request._calcu_hybrid_cpu_cache_page_len_list()
        request.token_hash_list.fill(page_hashes)
        request.token_hash_page_len_list.fill(page_lens)
        for page, end in enumerate(stored_ends):
            ready_pages[hashes[end // args.linear_att_hash_page_size - 1]] = page
        manager.shm_req_manager.get_req_obj_by_index.return_value = request
        return request

    def match(request):
        group = GroupReqIndexes(0, None, [0], time.time())
        manager._handle_group_req_multi_cache_match(group, time.time())
        manager.send_to_router.send_pyobj.assert_called_once()
        return request.cpu_cache_match_page_indexes.get_all()

    return manager, make_request, match, refs


@pytest.mark.parametrize(
    "length, stored_ends, expected_pages, tail_end",
    [
        (1025, [512, 768], [1], 768),
        (769, [512], [0], 512),
        (2049, [1024, 1280, 1792], [0, 2], 1792),
        (2049, [512, 2048], [0], 512),
        (2049, [1024, 2048, 1792], [0, 1], 0),
        (769, [768, 512], [0], 0),
        (2049, [1024], [0], 0),
        (257, [], [], 0),
        (256, [], [], 0),
    ],
)
def test_match_longest_ready_tail_after_full_pages(cache_match, length, stored_ends, expected_pages, tail_end):
    manager, make_request, match, refs = cache_match
    request = make_request(length, stored_ends)
    page_hashes = request.token_hash_list.get_all()
    page_lens = request.token_hash_page_len_list.get_all()

    assert match(request) == expected_pages
    assert request.cpu_cache_match_tail_len == tail_end
    assert refs == Counter(expected_pages)
    assert request.token_hash_list.get_all() == page_hashes
    assert request.token_hash_page_len_list.get_all() == page_lens
    lock = manager.cpu_cache_client.lock
    assert lock.acquire_sleep1ms.call_count == lock.release.call_count


def test_full_attention_does_not_match_hybrid_tail(cache_match):
    manager, make_request, match, refs = cache_match
    manager.is_hybrid_att_model = False
    request = make_request(1025, [768])
    assert match(request) == []
    assert not refs


def test_disk_pages_take_precedence_over_cpu_tail(cache_match):
    manager, make_request, match, refs = cache_match
    request = make_request(3073, [1024, 1792, 2560])
    manager.only_cpu_cache_enable = False
    manager._disk_cache_match = Mock(return_value=([0, 99], 1))

    assert match(request) == [0, 99, 2]
    assert request.cpu_cache_match_tail_len == 2560
    assert request.disk_prompt_cache_len == 1024
    assert refs == Counter([0, 2])


def test_prompt_logprobs_skips_tail_matching(cache_match):
    manager, make_request, match, refs = cache_match
    request = make_request(1025, [768])
    request.sample_params.prompt_logprobs = 0
    assert match(request) == []
    assert not refs
    manager.cpu_cache_client.query_one_page.assert_not_called()
