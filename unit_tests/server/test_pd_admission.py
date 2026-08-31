import asyncio

import pytest

from lightllm.server.httpserver_for_pd_master.admission import (
    AdmissionPolicy,
    AdmissionPriority,
    AdmissionRequest,
    CacheCapacitySnapshot,
    PDAdmissionController,
    SessionTracker,
)
from lightllm.utils.error_utils import ServerBusyError


def _request(
    priority=AdmissionPriority.COLD,
    session_key=None,
    decode_slots=1,
    estimated_uncached_work=0,
):
    return AdmissionRequest(
        session_key=session_key,
        priority=priority,
        decode_slots=decode_slots,
        estimated_uncached_work=estimated_uncached_work,
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


def test_multi_choice_request_reserves_capacity_across_individual_releases():
    async def run():
        controller = PDAdmissionController(lambda: 3)
        active = [await controller.acquire(_request()) for _ in range(3)]
        multi_choice_task = asyncio.create_task(controller.acquire(_request(decode_slots=2)))
        later_task = asyncio.create_task(controller.acquire(_request()))
        await asyncio.sleep(0)

        active[0].release()
        await asyncio.sleep(0)
        assert multi_choice_task.done() is False
        assert later_task.done() is False

        active[1].release()
        multi_choice = await multi_choice_task
        assert later_task.done() is False

        active[2].release()
        later = await later_task
        multi_choice.release()
        later.release()

    asyncio.run(run())


def test_multi_choice_reservation_does_not_block_fitting_higher_priority_request():
    async def run():
        controller = PDAdmissionController(lambda: 3)
        active = [await controller.acquire(_request()) for _ in range(3)]

        # 先消费调度表中的 continuation 和 probable 配额，使下一次轮到 cold。
        first_continuation_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.CONTINUATION)))
        first_probable_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.PROBABLE_CACHE_HIT)))
        await asyncio.sleep(0)
        active[0].release()
        first_continuation = await first_continuation_task
        active[1].release()
        first_probable = await first_probable_task

        large_cold_task = asyncio.create_task(controller.acquire(_request(decode_slots=2)))
        later_continuation_task = asyncio.create_task(controller.acquire(_request(AdmissionPriority.CONTINUATION)))
        await asyncio.sleep(0)

        active[2].release()
        later_continuation = await later_continuation_task
        assert controller.active_slots == 3
        assert large_cold_task.done() is False

        first_continuation.release()
        assert large_cold_task.done() is False
        first_probable.release()
        large_cold = await large_cold_task

        later_continuation.release()
        large_cold.release()

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


def test_cold_capacity_tracks_live_cache_headroom_and_actual_miss_size():
    snapshot = [CacheCapacitySnapshot(total_tokens=200, capacity_tokens=1000)]
    controller = PDAdmissionController(
        lambda: 4,
        cache_capacity_provider=lambda: snapshot[0],
    )
    cold = _request(AdmissionPriority.COLD)

    controller.record_prefill_result(cold, prompt_tokens=100, cached_tokens=0)
    assert controller.cold_capacity == 4

    snapshot[0] = CacheCapacitySnapshot(total_tokens=800, capacity_tokens=1000)
    controller.on_capacity_change()
    assert controller.cold_capacity == 2

    snapshot[0] = CacheCapacitySnapshot(total_tokens=1000, capacity_tokens=1000)
    controller.on_capacity_change()
    assert controller.cold_capacity == 1


def test_cache_friendly_request_bypasses_cold_capacity():
    async def run():
        snapshot = CacheCapacitySnapshot(total_tokens=1000, capacity_tokens=1000)
        controller = PDAdmissionController(
            lambda: 3,
            cache_capacity_provider=lambda: snapshot,
        )
        cold_request = _request(AdmissionPriority.COLD)
        controller.record_prefill_result(cold_request, prompt_tokens=100, cached_tokens=0)

        first_cold = await controller.acquire(cold_request)
        second_cold_task = asyncio.create_task(controller.acquire(cold_request))
        await asyncio.sleep(0)
        assert second_cold_task.done() is False

        probable = await controller.acquire(_request(AdmissionPriority.PROBABLE_CACHE_HIT))
        assert controller.active_slots == 2
        probable.release()

        first_cold.release()
        second_cold = await second_cold_task
        second_cold.release()

    asyncio.run(run())


def test_probable_cache_hits_consume_cold_capacity_when_actual_hits_are_low():
    async def run():
        snapshot = CacheCapacitySnapshot(total_tokens=1000, capacity_tokens=1000)
        controller = PDAdmissionController(
            lambda: 3,
            cache_capacity_provider=lambda: snapshot,
        )
        probable_request = _request(AdmissionPriority.PROBABLE_CACHE_HIT)

        initially_trusted = await controller.acquire(probable_request)
        assert controller.active_cold_slots == 0
        controller.record_prefill_result(
            probable_request,
            prompt_tokens=100,
            cached_tokens=0,
        )
        assert controller.cold_capacity == 1

        first_gated = await controller.acquire(probable_request)
        assert controller.active_cold_slots == 1
        second_gated_task = asyncio.create_task(controller.acquire(probable_request))
        await asyncio.sleep(0)
        assert second_gated_task.done() is False

        # 可信度变化不能让已经取得的租约在释放时误扣冷槽位。
        initially_trusted.release()
        assert controller.active_cold_slots == 1
        assert second_gated_task.done() is False

        first_gated.release()
        second_gated = await second_gated_task
        second_gated.release()

    asyncio.run(run())


def test_smaller_cold_request_is_dispatched_first():
    async def run():
        controller = PDAdmissionController(lambda: 2)
        active = [await controller.acquire(_request()) for _ in range(2)]
        large_task = asyncio.create_task(controller.acquire(_request(estimated_uncached_work=100)))
        small_task = asyncio.create_task(controller.acquire(_request(estimated_uncached_work=10)))
        await asyncio.sleep(0)

        active[0].release()
        small = await small_task
        assert large_task.done() is False

        active[1].release()
        large = await large_task
        small.release()
        large.release()

    asyncio.run(run())
