from contextlib import nullcontext
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.common import kv_transfer_resources as shared


def test_dp_export_without_pd_layout():
    mem = SimpleNamespace(
        kv_buffer=object(),
        dtype="bfloat16",
        kv_is_replicated=False,
        get_paged_kv_move_buffer_shape=Mock(side_effect=RuntimeError("PD layout unavailable")),
    )
    view = shared.KVTransferView.from_mem_manager(mem)
    assert view.kv_buffer is mem.kv_buffer
    assert view.page_shape is None
    mem.get_paged_kv_move_buffer_shape.assert_not_called()
    with pytest.raises(RuntimeError, match="PD layout unavailable"):
        shared.KVTransferView.from_mem_manager(mem, include_page_layout=True)


def test_pd_export_with_page_layout():
    mem = SimpleNamespace(
        kv_buffer=object(),
        dtype="bfloat16",
        kv_is_replicated=True,
        get_paged_kv_move_buffer_shape=Mock(return_value=(1, 1, 2, 1, 576)),
    )
    view = shared.KVTransferView.from_mem_manager(mem, include_page_layout=True)
    assert view.page_shape == (2, 1, 576)
    assert view.page_dtype == mem.dtype
    assert view.replicated
    mem.get_paged_kv_move_buffer_shape.assert_called_once_with(1, 1)


def test_shared_resource_exports_have_independent_consumer_slots(monkeypatch):
    segment = SimpleNamespace(name="test", buf=None, close=lambda: None)

    def shm(**kwargs):
        if kwargs["force_mode"] == "create":
            segment.size = kwargs["expected_size"]
            segment.buf = memoryview(bytearray(segment.size))
        return segment

    monkeypatch.setattr(shared, "create_or_link_shm", shm)
    monkeypatch.setattr(shared, "FileLock", lambda _: nullcontext())
    monkeypatch.setattr(shared, "get_service_shm_name", lambda name: name)
    monkeypatch.setattr(shared, "get_node_world_size", lambda: 2)
    monkeypatch.setattr(shared, "get_current_rank_in_node", lambda: 0)
    monkeypatch.setattr(shared, "kv_trans_use_p2p", lambda: False)
    monkeypatch.setattr(
        shared, "ForkingPickler", SimpleNamespace(dumps=lambda obj: memoryview(pickle.dumps(obj)), loads=pickle.loads)
    )
    resources = shared.KVTransferResources(SimpleNamespace(size=32), [[1, 2]], SimpleNamespace(mtp_step=2))
    resources.write_to_shm()
    for remaining in (3, 2, 1, 0):
        restored = shared.KVTransferResources.loads_from_shm(0)
        assert restored.kv.size == 32
        assert restored.req_to_token_indexes == [[1, 2]]
        assert restored.state_transfer.mtp_step == 2
        assert int.from_bytes(segment.buf[:4], "little") == remaining
    with pytest.raises(AssertionError):
        shared.KVTransferResources.loads_from_shm(0)
