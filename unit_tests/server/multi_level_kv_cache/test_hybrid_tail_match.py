from types import SimpleNamespace

from lightllm.server.multi_level_kv_cache.manager import MultiLevelKVCacheManager


class _HashList:
    def __init__(self, values):
        self.values = list(values)
        self.size = len(self.values)

    def get_all(self):
        return list(self.values)

    def fill(self, values):
        self.values = list(values)
        self.size = len(self.values)

    def clear(self):
        self.values = []
        self.size = 0


class _Lock:
    def acquire_sleep1ms(self):
        return None

    def release(self):
        return None


def _manager(*, only_cpu=True):
    manager = MultiLevelKVCacheManager.__new__(MultiLevelKVCacheManager)
    manager.args = SimpleNamespace(linear_att_page_block_num=4, linear_att_hash_page_size=4)
    manager.only_cpu_cache_enable = only_cpu
    manager.cpu_cache_time_out = 0.5
    manager.cpu_cache_client = SimpleNamespace(lock=_Lock())
    return manager


def _req():
    return SimpleNamespace(
        hybrid_token_hash_list=_HashList(range(1, 12)),
        token_hash_list=_HashList([4, 8, 11]),
        token_hash_page_len_list=_HashList([16, 32, 44]),
    )


def test_cpu_cache_reuses_previous_hybrid_tail_page():
    manager = _manager()
    queried = []

    def query(hash_value):
        queried.append(hash_value)
        return (9, True) if hash_value == 6 else (None, False)

    manager.cpu_cache_client.query_one_page = query
    req = _req()

    pages, disk_pages = manager._match_hybrid_tail(req, [4, 8, 11], [3])

    assert pages == [3, 9]
    assert disk_pages == 0
    assert queried == [7, 6]
    assert req.token_hash_list.get_all() == [4, 6]
    assert req.token_hash_page_len_list.get_all() == [16, 24]


def test_disk_tail_search_can_close_the_first_missing_disk_block():
    manager = _manager(only_cpu=False)
    manager.disk_cache_worker = SimpleNamespace(service=SimpleNamespace(_n=2))
    attempts = []

    def disk_match(hashes, pages):
        attempts.append(hashes)
        return ([7, 8], 2) if hashes == [4, 6] else (pages, 0)

    manager._disk_cache_match = disk_match
    req = _req()

    pages, disk_pages = manager._match_hybrid_tail(req, [4, 8, 11], [])

    assert pages == [7, 8]
    assert disk_pages == 2
    assert attempts == [[4, 7], [4, 6]]
    assert req.token_hash_list.get_all() == [4, 6]
    assert req.token_hash_page_len_list.get_all() == [16, 24]
