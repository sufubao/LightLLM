import json
from types import SimpleNamespace

from lightllm.common.mtp_scheduler import (
    MTPDecodePlanScheduler,
    get_uncovered_active_request_counts,
    get_mtp_plan_decode_token_num,
    load_mtp_decode_profile,
    select_mtp_decode_plan,
)


def _write_profile(
    tmp_path,
    plans,
    lease_steps=2,
    plan_switch_steps=2,
    min_switch_speedup=0.0,
):
    profile_path = tmp_path / "mtp_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "lease_steps": lease_steps,
                "plan_switch_steps": plan_switch_steps,
                "min_switch_speedup": min_switch_speedup,
                "plans": plans,
            }
        )
    )
    return load_mtp_decode_profile(str(profile_path))


def test_profile_selects_highest_throughput_feasible_plan(tmp_path):
    profile = _write_profile(
        tmp_path,
        plans=[
            {
                "min_active_requests": 33,
                "max_active_requests": 128,
                "microbatch_size": 32,
                "mtp_step": 3,
                "estimated_output_tps": 3800,
            },
            {
                "min_active_requests": 33,
                "max_active_requests": 128,
                "microbatch_size": 64,
                "mtp_step": 2,
                "estimated_output_tps": 3400,
            },
            {
                "min_active_requests": 33,
                "max_active_requests": 128,
                "microbatch_size": 64,
                "mtp_step": 4,
                "estimated_output_tps": 9999,
            },
        ],
    )

    plan = select_mtp_decode_plan(
        active_requests=64,
        workspace_rows=128,
        max_mtp_step=4,
        profile=profile,
    )

    assert (plan.logical_batch_size, plan.mtp_step) == (32, 3)
    assert plan.profiled
    assert get_mtp_plan_decode_token_num(plan) == 256


def test_profile_gap_falls_back_to_capacity_depth(tmp_path):
    profile = _write_profile(
        tmp_path,
        plans=[
            {
                "min_active_requests": 64,
                "max_active_requests": 128,
                "microbatch_size": 32,
                "mtp_step": 3,
                "estimated_output_tps": 3800,
            }
        ],
    )

    plan = select_mtp_decode_plan(
        active_requests=16,
        workspace_rows=128,
        max_mtp_step=4,
        profile=profile,
    )

    assert (plan.logical_batch_size, plan.mtp_step) == (16, 4)
    assert not plan.profiled


def test_profile_can_select_dense_decode_without_workspace(tmp_path):
    profile = _write_profile(
        tmp_path,
        plans=[
            {
                "min_active_requests": 64,
                "max_active_requests": 128,
                "microbatch_size": 128,
                "mtp_step": 0,
                "estimated_output_tps": 5000,
            }
        ],
    )

    plan = select_mtp_decode_plan(
        active_requests=128,
        workspace_rows=0,
        max_mtp_step=4,
        profile=profile,
    )

    assert (plan.logical_batch_size, plan.mtp_step) == (128, 0)
    assert plan.profiled
    assert get_mtp_plan_decode_token_num(plan) == 256


def test_scheduler_retains_dense_plan_across_profile_gap(tmp_path):
    profile = _write_profile(
        tmp_path,
        plans=[
            {
                "min_active_requests": 64,
                "max_active_requests": 128,
                "microbatch_size": 128,
                "mtp_step": 0,
                "estimated_output_tps": 5000,
            },
            {
                "min_active_requests": 1,
                "max_active_requests": 63,
                "microbatch_size": 32,
                "mtp_step": 3,
                "estimated_output_tps": 3000,
            },
        ],
        plan_switch_steps=2,
    )
    scheduler = MTPDecodePlanScheduler(
        workspace_rows=96,
        max_mtp_step=4,
        profile=profile,
    )
    reqs = [SimpleNamespace(req_id=i) for i in range(128)]

    _, plan = scheduler.select(reqs)
    assert plan.mtp_step == 0

    _, plan = scheduler.select(reqs[:63])
    assert (plan.logical_batch_size, plan.mtp_step) == (63, 0)


def test_dense_profile_allows_workspace_smaller_than_request_capacity(
    tmp_path,
):
    profile = _write_profile(
        tmp_path,
        plans=[
            {
                "min_active_requests": 1,
                "max_active_requests": 32,
                "microbatch_size": 32,
                "mtp_step": 3,
                "estimated_output_tps": 3000,
            },
            {
                "min_active_requests": 33,
                "max_active_requests": 128,
                "microbatch_size": 128,
                "mtp_step": 0,
                "estimated_output_tps": 5000,
            },
        ],
    )

    assert (
        get_uncovered_active_request_counts(
            profile=profile,
            running_max_req_size=128,
            workspace_rows=96,
            max_mtp_step=4,
        )
        == []
    )


def test_profile_coverage_reports_infeasible_mtp_ranges(tmp_path):
    profile = _write_profile(
        tmp_path,
        plans=[
            {
                "min_active_requests": 1,
                "max_active_requests": 128,
                "microbatch_size": 128,
                "mtp_step": 1,
                "estimated_output_tps": 4000,
            }
        ],
    )

    assert get_uncovered_active_request_counts(
        profile=profile,
        running_max_req_size=128,
        workspace_rows=96,
        max_mtp_step=4,
    ) == list(range(97, 129))


def test_scheduler_retains_then_rotates_profiled_microbatch(tmp_path):
    profile = _write_profile(
        tmp_path,
        plans=[
            {
                "min_active_requests": 1,
                "max_active_requests": 128,
                "microbatch_size": 2,
                "mtp_step": 3,
                "estimated_output_tps": 100,
            }
        ],
        lease_steps=2,
    )
    scheduler = MTPDecodePlanScheduler(
        workspace_rows=8,
        max_mtp_step=4,
        profile=profile,
    )
    reqs = [SimpleNamespace(req_id=i) for i in range(4)]

    selected, plan = scheduler.select(reqs)
    assert [req.req_id for req in selected] == [0, 1]
    assert plan.mtp_step == 3

    scheduler.mark_mtp_step()
    selected, _ = scheduler.select(reqs)
    assert [req.req_id for req in selected] == [0, 1]

    scheduler.mark_mtp_step()
    selected, _ = scheduler.select(reqs)
    assert [req.req_id for req in selected] == [2, 3]


def test_plan_hysteresis_ignores_one_step_boundary_flap(tmp_path):
    profile = _write_profile(
        tmp_path,
        plans=[
            {
                "min_active_requests": 1,
                "max_active_requests": 15,
                "microbatch_size": 15,
                "mtp_step": 3,
                "estimated_output_tps": 100,
            },
            {
                "min_active_requests": 16,
                "max_active_requests": 32,
                "microbatch_size": 32,
                "mtp_step": 4,
                "estimated_output_tps": 200,
            },
        ],
        plan_switch_steps=2,
    )
    scheduler = MTPDecodePlanScheduler(
        workspace_rows=128,
        max_mtp_step=4,
        profile=profile,
    )
    reqs = [SimpleNamespace(req_id=i) for i in range(16)]

    _, plan = scheduler.select(reqs)
    assert plan.mtp_step == 4

    _, plan = scheduler.select(reqs[:15])
    assert plan.mtp_step == 4
    scheduler.mark_mtp_step()

    _, plan = scheduler.select(reqs)
    assert plan.mtp_step == 4

    _, plan = scheduler.select(reqs[:15])
    assert plan.mtp_step == 4


def test_plan_hysteresis_ignores_small_profile_gain(tmp_path):
    profile = _write_profile(
        tmp_path,
        plans=[
            {
                "min_active_requests": 1,
                "max_active_requests": 15,
                "microbatch_size": 15,
                "mtp_step": 3,
                "estimated_output_tps": 104,
            },
            {
                "min_active_requests": 1,
                "max_active_requests": 15,
                "microbatch_size": 15,
                "mtp_step": 4,
                "estimated_output_tps": 100,
            },
            {
                "min_active_requests": 16,
                "max_active_requests": 32,
                "microbatch_size": 32,
                "mtp_step": 4,
                "estimated_output_tps": 200,
            },
        ],
        plan_switch_steps=2,
        min_switch_speedup=0.05,
    )
    scheduler = MTPDecodePlanScheduler(
        workspace_rows=128,
        max_mtp_step=4,
        profile=profile,
    )
    reqs = [SimpleNamespace(req_id=i) for i in range(16)]

    _, plan = scheduler.select(reqs)
    assert plan.mtp_step == 4

    for _ in range(10):
        scheduler.mark_mtp_step()
        _, plan = scheduler.select(reqs[:15])

    assert plan.mtp_step == 4


def test_scheduler_keeps_requests_when_only_depth_changes(tmp_path):
    profile = _write_profile(
        tmp_path,
        plans=[
            {
                "min_active_requests": 1,
                "max_active_requests": 2,
                "microbatch_size": 2,
                "mtp_step": 4,
                "estimated_output_tps": 100,
            },
            {
                "min_active_requests": 3,
                "max_active_requests": 4,
                "microbatch_size": 2,
                "mtp_step": 3,
                "estimated_output_tps": 200,
            },
        ],
    )
    scheduler = MTPDecodePlanScheduler(
        workspace_rows=8,
        max_mtp_step=4,
        profile=profile,
    )
    reqs = [SimpleNamespace(req_id=i) for i in range(4)]

    selected, plan = scheduler.select(reqs[:2])
    assert [req.req_id for req in selected] == [0, 1]
    assert plan.mtp_step == 4

    selected, plan = scheduler.select(reqs)
    assert [req.req_id for req in selected] == [0, 1]

    for _ in range(2):
        scheduler.mark_mtp_step()
        selected, plan = scheduler.select(reqs)

    assert [req.req_id for req in selected] == [0, 1]
    assert plan.mtp_step == 3


def test_zero_lease_keeps_owners_until_they_finish(tmp_path):
    profile = _write_profile(
        tmp_path,
        plans=[
            {
                "min_active_requests": 1,
                "max_active_requests": 128,
                "microbatch_size": 2,
                "mtp_step": 3,
                "estimated_output_tps": 100,
            }
        ],
        lease_steps=0,
    )
    scheduler = MTPDecodePlanScheduler(
        workspace_rows=8,
        max_mtp_step=4,
        profile=profile,
    )
    reqs = [SimpleNamespace(req_id=i) for i in range(4)]

    selected, _ = scheduler.select(reqs)
    assert [req.req_id for req in selected] == [0, 1]

    for _ in range(100):
        scheduler.mark_mtp_step()
    selected, _ = scheduler.select(reqs)
    assert [req.req_id for req in selected] == [0, 1]

    selected, _ = scheduler.select(reqs[1:])
    assert [req.req_id for req in selected] == [1, 2]
