import time
from types import SimpleNamespace
from unittest.mock import Mock

from lightllm.server.multi_level_kv_cache.manager import MultiLevelKVCacheManager


def test_hybrid_lookup_attaches_token_storage_before_reading():
    attached = []

    def read_tokens():
        assert attached
        return [1, 2, 3]

    req = SimpleNamespace(
        is_aborted=False,
        input_len=3,
        sample_params=SimpleNamespace(prompt_logprobs=-1, disable_prompt_cache=False),
        link_prompt_ids_shm_array=lambda: attached.append(True),
        get_prompt_ids=read_tokens,
        token_hash_list=SimpleNamespace(get_all=lambda: []),
        cpu_cache_match_page_indexes=Mock(),
        token_hash_page_len_list=Mock(),
    )
    manager = MultiLevelKVCacheManager.__new__(MultiLevelKVCacheManager)
    manager.args = SimpleNamespace(diverse_mode=False, cpu_cache_token_page_size=8)
    manager.cpu_cache_time_out = 100
    manager.only_cpu_cache_enable = True
    manager.shm_req_manager = Mock()
    manager.shm_req_manager.get_req_obj_by_index.return_value = req
    manager.cpu_cache_client = Mock()
    manager.linear_state_client = Mock()
    manager.linear_state_client.match.return_value = (-1, [], [])
    manager.send_to_router = Mock()
    group = SimpleNamespace(shm_req_indexes=[0])
    manager._handle_group_req_multi_cache_match(group, time.time())
    manager.linear_state_client.match.assert_called_once_with([1, 2, 3], 2)
    manager.shm_req_manager.put_back_req_obj.assert_called_once_with(req)
    manager.send_to_router.send_pyobj.assert_called_once()
