import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from lightllm.common.mtp_workspace import select_runtime_mtp_step


@dataclass(frozen=True)
class MTPPlanRule:
    min_active_requests: int
    max_active_requests: int
    microbatch_size: int
    mtp_step: int
    estimated_output_tps: float

    def matches(
        self,
        active_requests: int,
        workspace_rows: int,
        max_mtp_step: int,
    ) -> bool:
        logical_batch_size = min(active_requests, self.microbatch_size)
        return (
            self.min_active_requests <= active_requests <= self.max_active_requests
            and self.mtp_step <= max_mtp_step
            and logical_batch_size * self.mtp_step <= workspace_rows
        )


@dataclass(frozen=True)
class MTPDecodePlan:
    logical_batch_size: int
    mtp_step: int
    estimated_output_tps: float
    profiled: bool


@dataclass(frozen=True)
class MTPDecodeProfile:
    rules: Tuple[MTPPlanRule, ...]
    lease_steps: int
    plan_switch_steps: int = 0
    min_switch_speedup: float = 0.0


def _parse_rule(raw_rule: dict) -> MTPPlanRule:
    rule = MTPPlanRule(
        min_active_requests=int(raw_rule["min_active_requests"]),
        max_active_requests=int(raw_rule["max_active_requests"]),
        microbatch_size=int(raw_rule["microbatch_size"]),
        mtp_step=int(raw_rule["mtp_step"]),
        estimated_output_tps=float(raw_rule["estimated_output_tps"]),
    )
    assert rule.min_active_requests > 0
    assert rule.max_active_requests >= rule.min_active_requests
    assert rule.microbatch_size > 0
    assert rule.mtp_step >= 0
    assert rule.estimated_output_tps > 0
    return rule


@lru_cache(maxsize=None)
def load_mtp_decode_profile(profile_path: str) -> MTPDecodeProfile:
    with open(profile_path, "r") as profile_file:
        raw_profile = json.load(profile_file)

    assert raw_profile.get("version") == 1, "unsupported MTP scheduler profile version"
    lease_steps = int(raw_profile.get("lease_steps", 0))
    assert lease_steps >= 0
    plan_switch_steps = int(raw_profile.get("plan_switch_steps", 0))
    assert plan_switch_steps >= 0
    min_switch_speedup = float(raw_profile.get("min_switch_speedup", 0.0))
    assert min_switch_speedup >= 0.0
    rules = tuple(_parse_rule(raw_rule) for raw_rule in raw_profile["plans"])
    assert rules, "MTP scheduler profile must contain at least one plan"
    return MTPDecodeProfile(
        rules=rules,
        lease_steps=lease_steps,
        plan_switch_steps=plan_switch_steps,
        min_switch_speedup=min_switch_speedup,
    )


def select_mtp_decode_plan(
    active_requests: int,
    workspace_rows: int,
    max_mtp_step: int,
    profile: Optional[MTPDecodeProfile],
) -> Optional[MTPDecodePlan]:
    if active_requests == 0:
        return None

    if profile is not None:
        matching_rules = [
            rule
            for rule in profile.rules
            if rule.matches(
                active_requests=active_requests,
                workspace_rows=workspace_rows,
                max_mtp_step=max_mtp_step,
            )
        ]
        if matching_rules:
            rule = max(
                matching_rules,
                key=lambda item: (
                    item.estimated_output_tps,
                    min(active_requests, item.microbatch_size),
                    -item.mtp_step,
                ),
            )
            return MTPDecodePlan(
                logical_batch_size=min(active_requests, rule.microbatch_size),
                mtp_step=rule.mtp_step,
                estimated_output_tps=rule.estimated_output_tps,
                profiled=True,
            )

    runtime_mtp_step = select_runtime_mtp_step(
        logical_batch_size=active_requests,
        workspace_rows=workspace_rows,
        max_mtp_step=max_mtp_step,
    )
    return MTPDecodePlan(
        logical_batch_size=active_requests,
        mtp_step=runtime_mtp_step,
        estimated_output_tps=0.0,
        profiled=False,
    )


def get_uncovered_active_request_counts(
    profile: MTPDecodeProfile,
    running_max_req_size: int,
    workspace_rows: int,
    max_mtp_step: int,
) -> List[int]:
    return [
        active_requests
        for active_requests in range(1, running_max_req_size + 1)
        if not any(
            rule.matches(
                active_requests=active_requests,
                workspace_rows=workspace_rows,
                max_mtp_step=max_mtp_step,
            )
            for rule in profile.rules
        )
    ]


def get_mtp_plan_decode_token_num(plan: Optional[MTPDecodePlan]) -> int:
    if plan is None:
        return 0
    return plan.logical_batch_size * (plan.mtp_step + 1) * 2


def get_hysteresis_mtp_decode_token_num(
    active_requests: int,
    workspace_rows: int,
    max_mtp_step: int,
) -> int:
    if active_requests == 0:
        return 0
    return max(
        min(active_requests, workspace_rows // mtp_step) * (mtp_step + 1) * 2 for mtp_step in range(1, max_mtp_step + 1)
    )


class MTPDecodePlanScheduler:
    def __init__(
        self,
        workspace_rows: int,
        max_mtp_step: int,
        profile: MTPDecodeProfile,
    ):
        self.workspace_rows = workspace_rows
        self.max_mtp_step = max_mtp_step
        self.profile = profile
        self.current_req_ids = []
        self.waiting_req_ids = []
        self.completed_lease_steps = 0
        self.current_plan = None
        self.pending_mtp_step = None
        self.completed_pending_steps = 0
        self.lock = threading.Lock()

    def _plan_for_retained_step(
        self,
        active_requests: int,
        mtp_step: int,
    ) -> MTPDecodePlan:
        matching_rules = [
            rule
            for rule in self.profile.rules
            if rule.mtp_step == mtp_step
            and rule.matches(
                active_requests=active_requests,
                workspace_rows=self.workspace_rows,
                max_mtp_step=self.max_mtp_step,
            )
        ]
        if matching_rules:
            rule = max(
                matching_rules,
                key=lambda item: (
                    item.estimated_output_tps,
                    min(active_requests, item.microbatch_size),
                ),
            )
            return MTPDecodePlan(
                logical_batch_size=min(active_requests, rule.microbatch_size),
                mtp_step=mtp_step,
                estimated_output_tps=rule.estimated_output_tps,
                profiled=True,
            )

        return MTPDecodePlan(
            logical_batch_size=(
                active_requests if mtp_step == 0 else min(active_requests, self.workspace_rows // mtp_step)
            ),
            mtp_step=mtp_step,
            estimated_output_tps=0.0,
            profiled=False,
        )

    def _stabilize_plan(
        self,
        active_requests: int,
        preferred_plan: MTPDecodePlan,
    ) -> MTPDecodePlan:
        if self.current_plan is None or preferred_plan.mtp_step == self.current_plan.mtp_step:
            self.pending_mtp_step = None
            self.completed_pending_steps = 0
            return preferred_plan

        retained_plan = self._plan_for_retained_step(
            active_requests=active_requests,
            mtp_step=self.current_plan.mtp_step,
        )
        if (
            retained_plan.profiled
            and preferred_plan.profiled
            and preferred_plan.estimated_output_tps
            <= retained_plan.estimated_output_tps * (1.0 + self.profile.min_switch_speedup)
        ):
            self.pending_mtp_step = None
            self.completed_pending_steps = 0
            return retained_plan

        if self.profile.plan_switch_steps == 0:
            self.pending_mtp_step = None
            self.completed_pending_steps = 0
            return preferred_plan

        if self.pending_mtp_step != preferred_plan.mtp_step:
            self.pending_mtp_step = preferred_plan.mtp_step
            self.completed_pending_steps = 0

        if self.completed_pending_steps >= self.profile.plan_switch_steps:
            self.pending_mtp_step = None
            self.completed_pending_steps = 0
            return preferred_plan

        return retained_plan

    def select(self, decode_candidates: Sequence) -> Tuple[List, Optional[MTPDecodePlan]]:
        with self.lock:
            plan = select_mtp_decode_plan(
                active_requests=len(decode_candidates),
                workspace_rows=self.workspace_rows,
                max_mtp_step=self.max_mtp_step,
                profile=self.profile,
            )
            if plan is None:
                self.current_req_ids = []
                self.waiting_req_ids = []
                self.completed_lease_steps = 0
                self.current_plan = None
                self.pending_mtp_step = None
                self.completed_pending_steps = 0
                return [], None
            plan = self._stabilize_plan(
                active_requests=len(decode_candidates),
                preferred_plan=plan,
            )

            req_by_id = {req.req_id: req for req in decode_candidates}
            candidate_ids = set(req_by_id)
            self.current_req_ids = [req_id for req_id in self.current_req_ids if req_id in candidate_ids]
            self.waiting_req_ids = [
                req_id
                for req_id in self.waiting_req_ids
                if (req_id in candidate_ids and req_id not in self.current_req_ids)
            ]

            known_ids = set(self.current_req_ids) | set(self.waiting_req_ids)
            self.waiting_req_ids.extend(req.req_id for req in decode_candidates if req.req_id not in known_ids)

            plan_shape = (plan.logical_batch_size, plan.mtp_step)
            current_shape = (
                None
                if self.current_plan is None
                else (
                    self.current_plan.logical_batch_size,
                    self.current_plan.mtp_step,
                )
            )
            if plan_shape != current_shape:
                overflow = self.current_req_ids[plan.logical_batch_size :]
                self.current_req_ids = self.current_req_ids[: plan.logical_batch_size]
                self.waiting_req_ids = overflow + self.waiting_req_ids
                self.completed_lease_steps = 0

            if self.profile.lease_steps > 0 and self.completed_lease_steps >= self.profile.lease_steps:
                if self.waiting_req_ids:
                    self.waiting_req_ids.extend(self.current_req_ids)
                    self.current_req_ids = []
                self.completed_lease_steps = 0

            while len(self.current_req_ids) < plan.logical_batch_size and self.waiting_req_ids:
                self.current_req_ids.append(self.waiting_req_ids.pop(0))

            self.current_plan = plan
            selected = [req_by_id[req_id] for req_id in self.current_req_ids]
            return selected, plan

    def mark_mtp_step(self):
        with self.lock:
            if self.current_plan is not None and self.profile.lease_steps > 0:
                self.completed_lease_steps += 1
            if self.pending_mtp_step is not None:
                self.completed_pending_steps += 1
