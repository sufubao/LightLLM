import pytest
import torch

from lightllm.common.attention_state.full_att import GQAAttentionConfig, GQAFullAttentionState
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.execution.storage import GQAStateAdapter, GQAStateStore


def make_gqa_state():
    state = GQAFullAttentionState(
        GQAAttentionConfig(
            logical_layer_groups=((0, 2),),
            global_head_num=2,
            head_dim=4,
            dtype=torch.float32,
        ),
        page_size=2,
        device="cpu",
    )
    state.allocate_storage(token_capacity=4)
    return state


def make_unallocated_gqa_state():
    return GQAFullAttentionState(
        GQAAttentionConfig(
            logical_layer_groups=((0, 2),),
            global_head_num=2,
            head_dim=4,
            dtype=torch.float32,
        ),
        page_size=2,
        device="cpu",
    )


def test_gqa_adapter_exposes_existing_storage_as_shared_state_views():
    adapter = GQAStateAdapter(make_gqa_state())

    first = adapter.view_for(0)
    shared = adapter.view_for(2)

    assert first.state.key == ("main", "gqa.0")
    assert first.state.producer == 0
    assert first.state.consumers == (0, 2)
    assert first.format.fields == ("k", "v")
    assert first.fields[0].data_ptr() == shared.fields[0].data_ptr()
    assert first.fields[1].data_ptr() == shared.fields[1].data_ptr()


def test_gqa_adapter_rejects_unregistered_or_unallocated_layers():
    state = make_gqa_state()
    adapter = GQAStateAdapter(state)

    with pytest.raises(ValueError, match="not registered"):
        adapter.spec_for(1)

    state.k_storage = None
    with pytest.raises(RuntimeError, match="allocated"):
        adapter.view_for(0)


def test_gqa_state_store_owns_allocate_view_and_close_lifecycle():
    store = GQAStateStore(GQAStateAdapter(make_unallocated_gqa_state()))

    with pytest.raises(RuntimeError, match="allocated"):
        store.view_for(0)

    store.allocate_storage(token_capacity=4)
    view = store.view_for(2)
    assert store.is_allocated
    assert view.fields[0].shape[0] == 4 + store.adapter.state.page_size

    with pytest.raises(RuntimeError, match="already allocated"):
        store.allocate_storage(token_capacity=4)

    store.close()
    store.close()
    assert not store.is_allocated
    with pytest.raises(RuntimeError, match="allocated"):
        store.view_for(0)


def test_attention_state_manager_routes_gqa_storage_through_execution_store(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "execution-store-test")
    manager = AttentionStateManager(request_capacity=1)
    state = make_unallocated_gqa_state()
    manager.register_state("main", state)
    manager.finish_state_registration()

    manager.allocate_storage(token_capacity=4, page_size=2, big_page_tokens=1, device="cpu")

    store = manager.get_execution_store("main", "gqa", logical_layer_id=2)
    view = manager.get_execution_view("main", "gqa", logical_layer_id=0)
    assert isinstance(store, GQAStateStore)
    assert store.is_allocated
    assert view.state.key == ("main", "gqa.0")
    assert view.fields[0].data_ptr() == state.k_storage[0].data_ptr()

    manager.close_execution_stores()
    assert state.k_storage is None
