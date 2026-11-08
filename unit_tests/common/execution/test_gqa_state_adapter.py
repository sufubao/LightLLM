import pytest
import torch

from lightllm.common.attention_state.full_att import GQAAttentionConfig, GQAFullAttentionState
from lightllm.common.execution.storage import GQAStateAdapter


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
