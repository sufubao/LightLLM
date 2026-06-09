import types
import torch

import lightllm.common.basemodel.mtp_verify_extra_state as mod


def _state(n_real, mtp_step, is_prefill=False, with_accept=True):
    step = mtp_step + 1
    s = types.SimpleNamespace()
    s.b_seq_len = torch.arange(1, n_real * step + 1, dtype=torch.int32)
    s.b_req_idx = torch.arange(n_real, dtype=torch.int32).repeat_interleave(step)
    s.b_mtp_index = torch.arange(step, dtype=torch.int32).repeat(n_real)
    s.is_prefill = is_prefill
    s.b_num_accepted_tokens = torch.ones(n_real, dtype=torch.int32) if with_accept else None
    return s


def test_verify_branch_sets_index_rows(monkeypatch):
    monkeypatch.setattr(mod, "get_env_start_args", lambda: types.SimpleNamespace(mtp_step=2))
    n_real, mtp_step = 3, 2
    step = mtp_step + 1
    s = _state(n_real, mtp_step)
    mod.init_mtp_verify_extra_state(s)
    assert s.is_mtp_verify is True
    assert s.b_ssm_index_rows.shape == (n_real, step)
    assert s.b_gdn_verify_cu_seqlens.tolist() == [0, 3, 6, 9]
    assert s.b_conv_buffer_idx.tolist() == [0, 1, 2]  # one widened conv slot per req


def test_non_verify_branch_no_index_rows(monkeypatch):
    monkeypatch.setattr(mod, "get_env_start_args", lambda: types.SimpleNamespace(mtp_step=2))
    s = _state(3, 2, with_accept=False)
    mod.init_mtp_verify_extra_state(s)
    assert s.is_mtp_verify is False
    assert s.b_ssm_index_rows is None
    assert s.b_gdn_verify_cu_seqlens is None
