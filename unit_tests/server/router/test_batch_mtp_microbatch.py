from types import SimpleNamespace

from lightllm.common.mtp_scheduler import MTPDecodeProfile, MTPPlanRule
from lightllm.server.router import batch as batch_module


class _Req:
    _next_id = 0

    def __init__(self, dp_index: int, need_tokens: int):
        self.request_id = self._next_id
        type(self)._next_id += 1
        self.sample_params = SimpleNamespace(suggested_dp_index=dp_index)
        self._need_tokens = need_tokens

    def get_decode_need_tokens(self):
        return self._need_tokens


def test_router_reservation_tracks_dynamic_mtp_step(monkeypatch):
    monkeypatch.setattr(
        batch_module,
        "get_env_start_args",
        lambda: SimpleNamespace(
            mtp_mode="eagle_with_att",
            dynamic_mtp=True,
            mtp_workspace_rows=8,
            max_mtp_step=4,
        ),
    )
    batch = batch_module.Batch(
        batch_id=1,
        reqs=[_Req(0, 10), _Req(0, 10)],
        dp_size_in_node=1,
    )

    assert batch.get_batch_decode_need_tokens() == [20]


def test_router_reservation_can_drop_at_step_boundary(monkeypatch):
    monkeypatch.setattr(
        batch_module,
        "get_env_start_args",
        lambda: SimpleNamespace(
            mtp_mode="eagle_with_att",
            dynamic_mtp=True,
            mtp_workspace_rows=8,
            max_mtp_step=4,
        ),
    )
    batch = batch_module.Batch(
        batch_id=1,
        reqs=[_Req(0, 10), _Req(0, 10), _Req(0, 10)],
        dp_size_in_node=1,
    )

    assert batch.get_batch_decode_need_tokens() == [18]


def test_router_reservation_uses_profiled_microbatch_plan(monkeypatch):
    profile = MTPDecodeProfile(
        rules=(
            MTPPlanRule(
                min_active_requests=3,
                max_active_requests=8,
                microbatch_size=2,
                mtp_step=3,
                estimated_output_tps=100,
            ),
        ),
        lease_steps=8,
    )
    monkeypatch.setattr(
        batch_module,
        "get_env_start_args",
        lambda: SimpleNamespace(
            mtp_mode="eagle_with_att",
            dynamic_mtp=True,
            mtp_workspace_rows=8,
            max_mtp_step=4,
            mtp_scheduler_profile="profile.json",
        ),
    )
    monkeypatch.setattr(batch_module, "load_mtp_decode_profile", lambda _: profile)
    batch = batch_module.Batch(
        batch_id=1,
        reqs=[_Req(0, 10) for _ in range(4)],
        dp_size_in_node=1,
    )

    assert batch.get_batch_decode_need_tokens() == [16]


def test_router_reservation_covers_delayed_profile_switch(monkeypatch):
    profile = MTPDecodeProfile(
        rules=(
            MTPPlanRule(
                min_active_requests=3,
                max_active_requests=8,
                microbatch_size=2,
                mtp_step=3,
                estimated_output_tps=100,
            ),
        ),
        lease_steps=0,
        plan_switch_steps=2,
    )
    monkeypatch.setattr(
        batch_module,
        "get_env_start_args",
        lambda: SimpleNamespace(
            mtp_mode="eagle_with_att",
            dynamic_mtp=True,
            mtp_workspace_rows=8,
            max_mtp_step=4,
            mtp_scheduler_profile="profile.json",
        ),
    )
    monkeypatch.setattr(batch_module, "load_mtp_decode_profile", lambda _: profile)
    batch = batch_module.Batch(
        batch_id=1,
        reqs=[_Req(0, 10) for _ in range(4)],
        dp_size_in_node=1,
    )

    assert batch.get_batch_decode_need_tokens() == [24]
