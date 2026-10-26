from types import SimpleNamespace

import pytest
import torch

from lightllm.common.linear_state import LinearStateManager
from lightllm.common.indexer_state import IndexerStateManager


def test_restore_and_reset_keep_state_slots_consistent():
    state = LinearStateManager.__new__(LinearStateManager)
    state.mtp_step = 2
    state.mtp_state_index = torch.full((3,), 2, dtype=torch.int32)
    state.conv_state = SimpleNamespace(buffer=torch.ones(1, 3, 2, 4))
    state.ssm_state = SimpleNamespace(buffer=torch.ones(1, 9, 2, 2))
    conv, ssm = torch.full((1, 2, 2), 7.0), torch.full((1, 2, 2), 9.0)
    state.restore(1, conv, ssm)
    assert torch.equal(state.conv_state.buffer[:, 1, :, :2], conv)
    assert torch.equal(state.ssm_state.buffer[:, 3], ssm)
    assert state.mtp_state_index.tolist() == [2, 0, 2]
    state.reset(1)
    assert not state.conv_state.buffer[:, 1].any()
    assert not state.ssm_state.buffer[:, 3:6].any()
    assert state.ssm_state.buffer[:, :3].eq(1).all()
    assert state.ssm_state.buffer[:, 6:].eq(1).all()


def test_indexer_reset_is_request_local():
    config = SimpleNamespace(
        full_att_dtype=torch.float32,
        index_kpool=4,
        index_head_dim=2,
        get_full_att_kv_layer_num_with_draft_model=lambda: 2,
        get_full_att_kv_layer_index=lambda index: index,
    )
    state = IndexerStateManager.__new__(IndexerStateManager)
    state.linear_config = config
    state.state = SimpleNamespace(buffer=torch.empty(2, 3, 5, 4))
    state.state.buffer.fill_(3)
    state.reset(1)
    assert not state.state.buffer[:, 1].any()
    assert state.state.buffer[:, 0].eq(3).all()
    assert state.get_layer_buffer(1) is not None
