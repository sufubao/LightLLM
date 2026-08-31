import asyncio

import pytest

from lightllm.server.httpserver_for_pd_master.admission import (
    AdmissionPolicy,
    AdmissionPriority,
    AdmissionRequest,
    PDAdmissionController,
    SessionTracker,
)
from lightllm.utils.error_utils import ServerBusyError


def _request(
    priority=AdmissionPriority.COLD,
    session_key=None,
    decode_slots=1,
):
    return AdmissionRequest(
        session_key=session_key,
        priority=priority,
        decode_slots=decode_slots,
    )


def test_admission_waits_before_granting_more_than_decode_capacity():
    async def run():
        controller = PDAdmissionController(lambda: 1)
        first = await controller.acquire(_request())
        second_task = asyncio.create_task(controller.acquire(_request()))
        await asyncio.sleep(0)

        assert controller.active_slots == 1
        assert controller.queued_slots == 1
        assert second_task.done() is False

        first.release()
        second = await second_task
        assert controller.active_slots == 1
        assert controller.queued_slots == 0
        assert second.waited_seconds >= 0
        second.release()

    asyncio.run(run())


def test_admission_prioritizes_continuations_without_starving_lower_classes():
    async def run():
        controller = PDAdmissionController(lambda: 3)
        active = [await controller.acquire(_request()) for _ in range(3)]
        cold_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.COLD)))
        probable_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.PROBABLE_CACHE_HIT)))
        continuation_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.CONTINUATION)))
        await asyncio.sleep(0)

        active[0].release()
        continuation = await continuation_task
        assert probable_task.done() is False
        assert cold_task.done() is False

        active[1].release()
        probable = await probable_task
        active[2].release()
        cold = await cold_task

        continuation.release()
        probable.release()
        cold.release()

    asyncio.run(run())


def test_decode_capacity_one_still_follows_slot_weighted_drr():
    async def run():
        policy = AdmissionPolicy(
            continuation_weight=8,
            probable_cache_hit_weight=3,
            cold_weight=1,
            waiting_decode_waves=12,
        )
        controller = PDAdmissionController(lambda: 1, policy=policy)
        active = await controller.acquire(_request())
        acquired = asyncio.Queue()

        async def acquire(label, priority):
            lease = await controller.acquire(_request(priority))
            await acquired.put((label, lease))

        tasks = [
            asyncio.create_task(acquire(f"continuation-{index}", AdmissionPriority.CONTINUATION)) for index in range(8)
        ]
        tasks.extend(
            asyncio.create_task(acquire(f"probable-{index}", AdmissionPriority.PROBABLE_CACHE_HIT))
            for index in range(3)
        )
        tasks.append(asyncio.create_task(acquire("cold-0", AdmissionPriority.COLD)))
        await asyncio.sleep(0)

        active.release()
        expected = ["continuation"] * 8 + ["probable"] * 3 + ["cold"]
        for expected_prefix in expected:
            label, lease = await asyncio.wait_for(acquired.get(), timeout=1.0)
            assert label.startswith(expected_prefix)
            lease.release()

        await asyncio.gather(*tasks)
        assert controller.active_slots == 0

    asyncio.run(run())


def test_higher_priority_request_can_replace_a_queued_cold_request():
    async def run():
        controller = PDAdmissionController(lambda: 1)
        active = await controller.acquire(_request())
        cold_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.COLD)))
        await asyncio.sleep(0)
        continuation_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.CONTINUATION)))
        await asyncio.sleep(0)

        with pytest.raises(ServerBusyError, match="Superseded"):
            await cold_task

        active.release()
        continuation = await continuation_task
        continuation.release()

    asyncio.run(run())


def test_multi_choice_request_acquires_all_slots_atomically():
    async def run():
        controller = PDAdmissionController(lambda: 3)
        active = await controller.acquire(_request(decode_slots=2))
        multi_choice_task = asyncio.create_task(controller.acquire(_request(decode_slots=2)))
        await asyncio.sleep(0)

        assert controller.active_slots == 2
        assert multi_choice_task.done() is False

        active.release()
        multi_choice = await multi_choice_task
        assert controller.active_slots == 2
        multi_choice.release()

    asyncio.run(run())


def test_same_priority_small_request_backfills_a_blocked_gang_without_idle_slots():
    async def run():
        controller = PDAdmissionController(lambda: 3)
        active = [await controller.acquire(_request()) for _ in range(3)]
        multi_choice_task = asyncio.create_task(controller.acquire(_request(decode_slots=2)))
        later_task = asyncio.create_task(controller.acquire(_request()))
        await asyncio.sleep(0)

        active[0].release()
        later = await later_task
        assert multi_choice_task.done() is False
        assert controller.active_slots == 3
        assert all(deficit >= 0 for deficit in controller._deficits.values())

        active[1].release()
        assert multi_choice_task.done() is False

        later.release()
        multi_choice = await multi_choice_task
        assert controller.active_slots == 3
        multi_choice.release()
        active[2].release()

    asyncio.run(run())


def test_blocked_gang_keeps_backfill_open_for_a_later_small_request():
    async def run():
        controller = PDAdmissionController(
            lambda: 2,
            policy=AdmissionPolicy(waiting_decode_waves=2),
        )
        active = [await controller.acquire(_request()) for _ in range(2)]
        gang_task = asyncio.create_task(controller.acquire(_request(decode_slots=2)))
        await asyncio.sleep(0)

        active[0].release()
        assert gang_task.done() is False
        assert controller.active_slots == 1

        small_task = asyncio.create_task(controller.acquire(_request()))
        small = await small_task
        assert controller.active_slots == 2
        assert gang_task.done() is False

        small.release()
        assert gang_task.done() is False
        active[1].release()
        gang = await gang_task
        gang.release()

    asyncio.run(run())


def test_full_queue_of_session_blocked_waiters_cannot_leave_other_sessions_idle():
    async def run():
        controller = PDAdmissionController(lambda: 4)
        active_a = await controller.acquire(_request(AdmissionPriority.CONTINUATION, session_key="session-a"))
        queued_a_tasks = [
            asyncio.create_task(controller.acquire(_request(AdmissionPriority.CONTINUATION, session_key="session-a")))
            for _ in range(4)
        ]
        await asyncio.sleep(0)
        assert controller.queued_slots == 4
        assert controller.active_slots == 1

        session_b = await controller.acquire(_request(AdmissionPriority.CONTINUATION, session_key="session-b"))
        assert controller.active_slots == 2
        assert controller.queued_slots == 4

        session_b.release()
        active_a.release()
        for task in queued_a_tasks:
            lease = await task
            lease.release()
        assert controller.active_slots == 0

    asyncio.run(run())


def test_full_gang_queue_still_accepts_a_fitting_request_within_backfill_budget():
    async def run():
        controller = PDAdmissionController(lambda: 4)
        active = await controller.acquire(_request(decode_slots=3))
        gang_tasks = [asyncio.create_task(controller.acquire(_request(decode_slots=2))) for _ in range(2)]
        await asyncio.sleep(0)
        assert controller.queued_slots == 4
        assert controller.active_slots == 3

        small = await controller.acquire(_request())
        assert controller.active_slots == 4
        assert controller.queued_slots == 4
        assert controller._backfilled_slots == 1

        small.release()
        active.release()
        first_gang = await gang_tasks[0]
        second_gang = await gang_tasks[1]
        first_gang.release()
        second_gang.release()
        assert controller.active_slots == 0

    asyncio.run(run())


def test_gang_reservation_starts_after_one_backfill_wave_and_blocks_later_priority():
    async def run():
        controller = PDAdmissionController(
            lambda: 3,
            policy=AdmissionPolicy(waiting_decode_waves=5),
        )
        active = [await controller.acquire(_request()) for _ in range(3)]
        gang_task = asyncio.create_task(controller.acquire(_request(decode_slots=3)))
        backfill_tasks = [asyncio.create_task(controller.acquire(_request())) for _ in range(3)]
        await asyncio.sleep(0)

        backfills = []
        for active_lease, backfill_task in zip(active, backfill_tasks):
            active_lease.release()
            backfills.append(await backfill_task)
            assert gang_task.done() is False

        later_high_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.CONTINUATION)))
        await asyncio.sleep(0)

        backfills[0].release()
        await asyncio.sleep(0)
        assert later_high_task.done() is False
        assert gang_task.done() is False
        backfills[1].release()
        await asyncio.sleep(0)
        assert later_high_task.done() is False
        assert gang_task.done() is False

        backfills[2].release()
        gang = await gang_task
        assert later_high_task.done() is False

        gang.release()
        later_high = await later_high_task
        later_high.release()

    asyncio.run(run())


def test_same_session_is_fifo_while_other_sessions_can_make_progress():
    async def run():
        controller = PDAdmissionController(lambda: 2)
        first_a = await controller.acquire(_request(AdmissionPriority.CONTINUATION, session_key="a"))
        second_a_task = asyncio.create_task(
            controller.acquire(_request(AdmissionPriority.CONTINUATION, session_key="a"))
        )
        await asyncio.sleep(0)
        session_b = await controller.acquire(_request(AdmissionPriority.CONTINUATION, session_key="b"))

        assert second_a_task.done() is False
        session_b.release()
        assert second_a_task.done() is False

        first_a.release()
        second_a = await second_a_task
        second_a.release()

    asyncio.run(run())


def test_cancelled_waiter_is_removed_and_does_not_leak_capacity():
    async def run():
        controller = PDAdmissionController(lambda: 1)
        active = await controller.acquire(_request())
        waiting_task = asyncio.create_task(controller.acquire(_request()))
        await asyncio.sleep(0)

        waiting_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting_task

        assert controller.queued_request_count == 0
        assert controller.queued_slots == 0
        active.release()
        assert controller.active_slots == 0

    asyncio.run(run())


def test_cancelling_reserved_gang_removes_the_backfill_barrier():
    async def run():
        controller = PDAdmissionController(
            lambda: 2,
            policy=AdmissionPolicy(waiting_decode_waves=4),
        )
        active = [await controller.acquire(_request()) for _ in range(2)]
        gang_task = asyncio.create_task(controller.acquire(_request(decode_slots=2)))
        backfill_tasks = [asyncio.create_task(controller.acquire(_request())) for _ in range(2)]
        await asyncio.sleep(0)

        backfills = []
        for active_lease, backfill_task in zip(active, backfill_tasks):
            active_lease.release()
            backfills.append(await backfill_task)

        later_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.CONTINUATION)))
        await asyncio.sleep(0)
        assert later_task.done() is False

        gang_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await gang_task

        backfills[0].release()
        later = await later_task
        backfills[1].release()
        later.release()

    asyncio.run(run())


def test_wait_timeout_removes_request_from_queue():
    async def run():
        policy = AdmissionPolicy(
            continuation_max_wait_seconds=0.01,
            probable_cache_hit_max_wait_seconds=0.01,
            cold_max_wait_seconds=0.01,
        )
        controller = PDAdmissionController(lambda: 1, policy=policy)
        active = await controller.acquire(_request())

        with pytest.raises(ServerBusyError, match="wait timed out"):
            await controller.acquire(_request())

        assert controller.queued_request_count == 0
        active.release()

    asyncio.run(run())


def test_reserved_gang_timeout_removes_the_backfill_barrier():
    async def run():
        policy = AdmissionPolicy(
            continuation_max_wait_seconds=1.0,
            probable_cache_hit_max_wait_seconds=1.0,
            cold_max_wait_seconds=0.03,
            waiting_decode_waves=4,
        )
        controller = PDAdmissionController(lambda: 2, policy=policy)
        active = [await controller.acquire(_request()) for _ in range(2)]
        gang_task = asyncio.create_task(controller.acquire(_request(decode_slots=2)))
        await asyncio.sleep(0)

        active[0].release()
        first_backfill_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.CONTINUATION)))
        first_backfill = await first_backfill_task

        second_backfill_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.CONTINUATION)))
        await asyncio.sleep(0)
        active[1].release()
        second_backfill = await second_backfill_task

        later_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.CONTINUATION)))
        with pytest.raises(ServerBusyError, match="wait timed out"):
            await gang_task

        first_backfill.release()
        later = await later_task
        second_backfill.release()
        later.release()

    asyncio.run(run())


def test_capacity_changes_wake_waiters_without_overcommitting():
    async def run():
        capacity = [1]
        controller = PDAdmissionController(lambda: capacity[0])
        first = await controller.acquire(_request())
        second_task = asyncio.create_task(controller.acquire(_request()))
        await asyncio.sleep(0)

        capacity[0] = 2
        controller.on_capacity_change()
        second = await second_task
        assert controller.active_slots == 2

        capacity[0] = 1
        controller.on_capacity_change()
        third_task = asyncio.create_task(controller.acquire(_request()))
        await asyncio.sleep(0)
        assert third_task.done() is False

        first.release()
        await asyncio.sleep(0)
        assert third_task.done() is False
        second.release()
        third = await third_task
        third.release()

    asyncio.run(run())


def test_capacity_shrink_fails_an_oversized_blocked_gang_and_clears_state():
    async def run():
        capacity = [3]
        controller = PDAdmissionController(
            lambda: capacity[0],
            policy=AdmissionPolicy(waiting_decode_waves=2),
        )
        active = [await controller.acquire(_request()) for _ in range(3)]
        gang_task = asyncio.create_task(controller.acquire(_request(decode_slots=3)))
        await asyncio.sleep(0)

        active[0].release()
        assert gang_task.done() is False

        capacity[0] = 2
        controller.on_capacity_change()
        with pytest.raises(ServerBusyError, match="fell below queued request size"):
            await gang_task
        assert controller.queued_slots == 0

        small_task = asyncio.create_task(controller.acquire(_request()))
        await asyncio.sleep(0)
        assert small_task.done() is False
        active[1].release()
        small = await small_task
        active[2].release()
        small.release()

    asyncio.run(run())


def test_capacity_shrink_trims_queue_to_dynamic_slot_limit_by_priority_and_recency():
    async def run():
        capacity = [4]
        policy = AdmissionPolicy(waiting_decode_waves=2)
        controller = PDAdmissionController(lambda: capacity[0], policy=policy)
        active = await controller.acquire(_request(decode_slots=4))

        cold_tasks = [asyncio.create_task(controller.acquire(_request(AdmissionPriority.COLD))) for _ in range(3)]
        probable_tasks = [
            asyncio.create_task(controller.acquire(_request(AdmissionPriority.PROBABLE_CACHE_HIT))) for _ in range(2)
        ]
        continuation_tasks = [
            asyncio.create_task(controller.acquire(_request(AdmissionPriority.CONTINUATION))) for _ in range(3)
        ]
        await asyncio.sleep(0)
        assert controller.queued_slots == 8

        capacity[0] = 1
        controller.on_capacity_change()
        assert controller.queued_slots == capacity[0] * policy.waiting_decode_waves

        for task in cold_tasks + probable_tasks + [continuation_tasks[-1]]:
            with pytest.raises(ServerBusyError, match="queue capacity shrank"):
                await task
        assert continuation_tasks[0].done() is False
        assert continuation_tasks[1].done() is False

        active.release()
        first = await continuation_tasks[0]
        assert continuation_tasks[1].done() is False
        first.release()
        second = await continuation_tasks[1]
        second.release()

    asyncio.run(run())


def test_zero_capacity_fails_all_queued_requests():
    async def run():
        capacity = [2]
        controller = PDAdmissionController(
            lambda: capacity[0],
            policy=AdmissionPolicy(waiting_decode_waves=2),
        )
        active = [await controller.acquire(_request()) for _ in range(2)]
        tasks = [asyncio.create_task(controller.acquire(_request())) for _ in range(2)]
        await asyncio.sleep(0)

        capacity[0] = 0
        controller.on_capacity_change()
        for task in tasks:
            with pytest.raises(ServerBusyError, match="fell below queued request size"):
                await task
        assert controller.queued_slots == 0
        assert controller.queued_request_count == 0

        for lease in active:
            lease.release()

    asyncio.run(run())


def test_session_tracker_requires_a_recent_observed_success():
    now = [0.0]
    tracker = SessionTracker(ttl_seconds=10, max_sessions=2, clock=lambda: now[0])

    assert tracker.is_continuation("session-a") is False
    tracker.mark_success("session-a")
    assert tracker.is_continuation("session-a") is True

    now[0] = 11
    assert tracker.is_continuation("session-a") is False

    tracker.mark_success("session-a")
    tracker.mark_success("session-b")
    tracker.mark_success("session-c")
    assert tracker.is_continuation("session-a") is False
    assert tracker.is_continuation("session-b") is True
    assert tracker.is_continuation("session-c") is True


def test_successful_session_promotes_its_waiting_requests():
    async def run():
        controller = PDAdmissionController(
            lambda: 1,
            policy=AdmissionPolicy(waiting_decode_waves=2),
        )
        active = await controller.acquire(_request())
        same_session_task = asyncio.create_task(controller.acquire(_request(session_key="session-a")))
        other_cold_task = asyncio.create_task(controller.acquire(_request()))
        await asyncio.sleep(0)

        controller.promote_session("session-a")
        active.release()
        promoted = await same_session_task
        assert promoted.request.priority == AdmissionPriority.CONTINUATION
        assert other_cold_task.done() is False

        promoted.release()
        other = await other_cold_task
        other.release()

    asyncio.run(run())


def test_session_promotion_extends_the_wait_deadline():
    async def run():
        policy = AdmissionPolicy(
            continuation_max_wait_seconds=0.2,
            probable_cache_hit_max_wait_seconds=0.1,
            cold_max_wait_seconds=0.02,
        )
        controller = PDAdmissionController(lambda: 1, policy=policy)
        active = await controller.acquire(_request())
        waiting_task = asyncio.create_task(controller.acquire(_request(session_key="session-a")))
        await asyncio.sleep(0.005)

        controller.promote_session("session-a")
        await asyncio.sleep(0.025)
        assert waiting_task.done() is False

        active.release()
        promoted = await waiting_task
        assert promoted.request.priority == AdmissionPriority.CONTINUATION
        promoted.release()

    asyncio.run(run())


def test_requests_remain_fifo_within_the_same_priority_class():
    async def run():
        controller = PDAdmissionController(
            lambda: 1,
            policy=AdmissionPolicy(waiting_decode_waves=2),
        )
        active = await controller.acquire(_request())
        first_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.COLD, session_key="first")))
        second_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.COLD, session_key="second")))
        await asyncio.sleep(0)

        active.release()
        first = await first_task
        assert second_task.done() is False
        first.release()
        second = await second_task
        second.release()

    asyncio.run(run())
