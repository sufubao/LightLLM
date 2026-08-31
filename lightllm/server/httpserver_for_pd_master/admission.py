from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Callable, Deque, Dict, Optional

from lightllm.utils.error_utils import ServerBusyError


class AdmissionPriority(IntEnum):
    """请求的业务优先级；数值越大，获得服务的权重越高。"""

    COLD = 0
    PROBABLE_CACHE_HIT = 1
    CONTINUATION = 2


@dataclass(frozen=True, slots=True)
class AdmissionPolicy:
    """PD Master 内部准入策略。

    这些值表达产品层面的等待与公平策略，因此集中在一个对象中，不散落到
    调度控制流里。容量本身由已注册的 Decode 节点动态提供。
    """

    continuation_weight: int = 8
    probable_cache_hit_weight: int = 3
    cold_weight: int = 1
    continuation_max_wait_seconds: float = 30.0
    probable_cache_hit_max_wait_seconds: float = 15.0
    cold_max_wait_seconds: float = 5.0
    waiting_decode_waves: int = 1
    probable_cache_hit_threshold: float = 0.5
    active_session_ttl_seconds: float = 30 * 60.0
    max_tracked_sessions: int = 100_000

    def __post_init__(self) -> None:
        """校验准入策略中的权重、超时和容量参数。"""
        if (
            min(
                self.continuation_weight,
                self.probable_cache_hit_weight,
                self.cold_weight,
            )
            < 1
        ):
            raise ValueError("admission weights must be positive")
        if (
            min(
                self.continuation_max_wait_seconds,
                self.probable_cache_hit_max_wait_seconds,
                self.cold_max_wait_seconds,
            )
            <= 0
        ):
            raise ValueError("admission wait timeouts must be positive")
        if self.waiting_decode_waves < 1:
            raise ValueError("waiting_decode_waves must be positive")
        if not 0.0 <= self.probable_cache_hit_threshold <= 1.0:
            raise ValueError("probable_cache_hit_threshold must be between zero and one")
        if self.active_session_ttl_seconds <= 0:
            raise ValueError("active_session_ttl_seconds must be positive")
        if self.max_tracked_sessions < 1:
            raise ValueError("max_tracked_sessions must be positive")

    def weight(self, priority: AdmissionPriority) -> int:
        """返回指定优先级在轮转调度中的权重。"""
        if priority == AdmissionPriority.CONTINUATION:
            return self.continuation_weight
        if priority == AdmissionPriority.PROBABLE_CACHE_HIT:
            return self.probable_cache_hit_weight
        return self.cold_weight

    def max_wait_seconds(self, priority: AdmissionPriority) -> float:
        """返回指定优先级允许的最长排队时间。"""
        if priority == AdmissionPriority.CONTINUATION:
            return self.continuation_max_wait_seconds
        if priority == AdmissionPriority.PROBABLE_CACHE_HIT:
            return self.probable_cache_hit_max_wait_seconds
        return self.cold_max_wait_seconds


@dataclass(frozen=True, slots=True)
class AdmissionRequest:
    session_key: Optional[str]
    priority: AdmissionPriority
    decode_slots: int = 1

    def __post_init__(self) -> None:
        """校验请求需要原子获取的 Decode 槽位数。"""
        if self.decode_slots < 1:
            raise ValueError("decode_slots must be positive")


class SessionTracker:
    """只把服务端已经成功观察过的 Session 视为连续会话。"""

    def __init__(
        self,
        ttl_seconds: float,
        max_sessions: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """初始化带 TTL 和数量上限的 Session 记录器。"""
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._clock = clock
        self._last_success: OrderedDict[str, float] = OrderedDict()

    def is_continuation(self, session_key: Optional[str]) -> bool:
        """判断 Session 是否在有效期内成功返回过结果。"""
        if not session_key:
            return False
        now = self._clock()
        last_success = self._last_success.get(session_key)
        if last_success is None:
            return False
        if now - last_success > self.ttl_seconds:
            self._last_success.pop(session_key, None)
            return False
        self._last_success.move_to_end(session_key)
        return True

    def mark_success(self, session_key: Optional[str]) -> None:
        """记录 Session 最近一次成功返回结果的时间。"""
        if not session_key:
            return
        self._last_success[session_key] = self._clock()
        self._last_success.move_to_end(session_key)
        while len(self._last_success) > self.max_sessions:
            self._last_success.popitem(last=False)


@dataclass(slots=True)
class _WaitingRequest:
    request: AdmissionRequest
    enqueue_time: float
    deadline: float
    deadline_changed: asyncio.Event
    future: asyncio.Future


class AdmissionLease:
    """一次已经获得的 Decode 容量租约。"""

    def __init__(
        self,
        controller: "PDAdmissionController",
        request: AdmissionRequest,
        waited_seconds: float,
    ) -> None:
        """保存本次租约占用的 Decode 槽位。"""
        self._controller = controller
        self.request = request
        self.waited_seconds = waited_seconds
        self._released = False

    async def __aenter__(self) -> "AdmissionLease":
        """进入异步上下文并返回当前租约。"""
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        """退出异步上下文时自动释放租约。"""
        self.release()

    def release(self) -> None:
        """幂等释放本次占用的准入槽位。"""
        if self._released:
            return
        self._released = True
        self._controller._release(self)


class PDAdmissionController:
    """在请求派发到 P/D 节点之前提供有界、可取消的公平等待队列。"""

    _PRIORITY_ORDER = (
        AdmissionPriority.CONTINUATION,
        AdmissionPriority.PROBABLE_CACHE_HIT,
        AdmissionPriority.COLD,
    )

    def __init__(
        self,
        decode_capacity_provider: Callable[[], int],
        policy: Optional[AdmissionPolicy] = None,
        clock: Callable[[], float] = time.monotonic,
        state_change_callback: Optional[Callable[["PDAdmissionController"], None]] = None,
    ) -> None:
        """初始化容量提供器、优先级队列和调度状态。"""
        self.policy = policy or AdmissionPolicy()
        self._decode_capacity_provider = decode_capacity_provider
        self._clock = clock
        self._state_change_callback = state_change_callback
        self._active_slots = 0
        self._active_sessions = set()
        self._queues: Dict[AdmissionPriority, Deque[_WaitingRequest]] = {
            priority: deque() for priority in self._PRIORITY_ORDER
        }
        self._session_queues: Dict[str, Deque[_WaitingRequest]] = {}
        self._queued_slots = 0
        self._deficits: Dict[AdmissionPriority, int] = {priority: 0 for priority in self._PRIORITY_ORDER}
        self._priority_index = 0
        self._priority_visit_started = False
        self._blocked_waiter: Optional[_WaitingRequest] = None
        self._backfilled_slots = 0
        self._backfill_limit = 0
        self._reservation_active = False

    @property
    def active_slots(self) -> int:
        """返回当前已经发放的 Decode 槽位数。"""
        return self._active_slots

    @property
    def queued_slots(self) -> int:
        """返回等待队列中的 Decode 槽位总数。"""
        return self._queued_slots

    @property
    def queued_request_count(self) -> int:
        """返回三个优先级队列中的请求总数。"""
        return sum(len(queue) for queue in self._queues.values())

    def _capacity(self) -> int:
        """读取并规范化当前可用的 Decode 容量。"""
        return max(0, int(self._decode_capacity_provider()))

    async def acquire(self, request: AdmissionRequest) -> AdmissionLease:
        """立即发放租约或等待队列调度后再返回租约。"""
        capacity = self._capacity()
        if capacity <= 0 or request.decode_slots > capacity:
            raise ServerBusyError("PD decode capacity is unavailable")

        if self.queued_request_count == 0 and self._can_activate(request, capacity):
            lease = self._activate(request, waited_seconds=0.0)
            self._notify_state_change()
            return lease

        idle_fill_lease = self._try_activate_idle_fill(request, capacity)
        if idle_fill_lease is not None:
            self._notify_state_change()
            return idle_fill_lease

        loop = asyncio.get_running_loop()
        enqueue_time = self._clock()
        waiter = _WaitingRequest(
            request=request,
            enqueue_time=enqueue_time,
            deadline=enqueue_time + self.policy.max_wait_seconds(request.priority),
            deadline_changed=asyncio.Event(),
            future=loop.create_future(),
        )

        if not self._make_queue_room(waiter):
            raise ServerBusyError("PD master admission queue is full")

        self._enqueue(waiter)
        self._drain()

        try:
            return await self._wait_for_lease(waiter)
        except asyncio.TimeoutError as exc:
            lease = self._cancel_waiter_or_take_lease(waiter)
            if lease is not None:
                lease.release()
            raise ServerBusyError("PD master admission queue wait timed out") from exc
        except BaseException:
            lease = self._cancel_waiter_or_take_lease(waiter)
            if lease is not None:
                lease.release()
            raise

    def on_capacity_change(self) -> None:
        """Decode 容量变化后重置临时公平状态并重新驱动队列。"""
        self._clear_backfill_state()
        self._reset_deficits()
        self._drain()

    def promote_session(self, session_key: Optional[str]) -> None:
        """把同一 Session 尚未派发的请求提升为连续会话优先级。"""
        if not session_key:
            return
        session_queue = self._session_queues.get(session_key)
        if not session_queue:
            return

        if self._blocked_waiter in session_queue:
            self._clear_backfill_state(reset_deficits=True)
        for waiter in tuple(session_queue):
            old_priority = waiter.request.priority
            if old_priority == AdmissionPriority.CONTINUATION:
                continue
            self._queues[old_priority].remove(waiter)
            waiter.request = replace(waiter.request, priority=AdmissionPriority.CONTINUATION)
            waiter.deadline = max(
                waiter.deadline,
                waiter.enqueue_time + self.policy.continuation_max_wait_seconds,
            )
            waiter.deadline_changed.set()
            self._queues[AdmissionPriority.CONTINUATION].append(waiter)
            if not self._queues[old_priority]:
                self._deficits[old_priority] = 0
        self._drain()

    def _can_activate(self, request: AdmissionRequest, capacity: Optional[int] = None) -> bool:
        """检查 Decode 总容量和 Session 串行约束。"""
        if capacity is None:
            capacity = self._capacity()
        if self._active_slots + request.decode_slots > capacity:
            return False
        return request.session_key is None or request.session_key not in self._active_sessions

    def _try_activate_idle_fill(
        self,
        request: AdmissionRequest,
        capacity: int,
    ) -> Optional[AdmissionLease]:
        """在满队列拒绝前，用当前唯一可运行的新请求填充空槽。

        只覆盖两种不会越过可运行旧请求的场景：现有队列全部受
        Session 串行约束，或已保护 gang 仍在一波 bounded backfill 预算内。
        """
        if not self._can_activate(request, capacity):
            return None
        if request.session_key is not None and request.session_key in self._session_queues:
            return None

        blocked = self._blocked_waiter
        if blocked is None:
            has_grantable_waiter = any(
                not waiter.future.done() and self._session_is_grantable(waiter)
                for queue in self._queues.values()
                for waiter in queue
            )
            if has_grantable_waiter:
                return None
            return self._activate(request, waited_seconds=0.0)

        available_slots = capacity - self._active_slots
        if (
            self._reservation_active
            or blocked.future.done()
            or not self._session_is_grantable(blocked)
            or self._has_fitting_backfill(blocked, available_slots)
        ):
            return None

        remaining_backfill = self._backfill_limit - self._backfilled_slots
        if request.decode_slots > remaining_backfill:
            return None

        lease = self._activate(request, waited_seconds=0.0)
        self._backfilled_slots += request.decode_slots
        if self._backfilled_slots >= self._backfill_limit:
            self._reservation_active = True
        return lease

    def _activate(self, request: AdmissionRequest, waited_seconds: float) -> AdmissionLease:
        """占用所需槽位并创建对应的准入租约。"""
        self._active_slots += request.decode_slots
        if request.session_key is not None:
            self._active_sessions.add(request.session_key)
        return AdmissionLease(self, request, waited_seconds)

    def _release(self, lease: AdmissionLease) -> None:
        """归还租约槽位并继续调度等待请求。"""
        request = lease.request
        self._active_slots -= request.decode_slots
        if self._active_slots < 0:
            raise RuntimeError("PD admission active slot count became negative")
        if request.session_key is not None:
            self._active_sessions.discard(request.session_key)
        self._drain()

    def _waiting_capacity(self) -> int:
        """返回等待队列允许容纳的槽位总数。"""
        return self._capacity() * self.policy.waiting_decode_waves

    def _make_queue_room(self, incoming: _WaitingRequest) -> bool:
        """必要时淘汰低优先级请求，为新请求腾出队列空间。"""
        waiting_capacity = self._waiting_capacity()
        if incoming.request.decode_slots > waiting_capacity:
            return False

        required_slots = self._queued_slots + incoming.request.decode_slots - waiting_capacity
        if required_slots <= 0:
            return True

        victims = []
        released_slots = 0
        for priority in reversed(self._PRIORITY_ORDER):
            if priority >= incoming.request.priority:
                continue
            for waiter in reversed(self._queues[priority]):
                victims.append(waiter)
                released_slots += waiter.request.decode_slots
                if released_slots >= required_slots:
                    break
            if released_slots >= required_slots:
                break

        if released_slots < required_slots:
            return False

        for victim in victims:
            self._remove_waiter(victim)
            if not victim.future.done():
                victim.future.set_exception(ServerBusyError("Superseded by a higher-priority queued request"))
        return True

    def _enqueue(self, waiter: _WaitingRequest) -> None:
        """把等待项加入优先级队列和 Session 队列。"""
        self._queues[waiter.request.priority].append(waiter)
        self._queued_slots += waiter.request.decode_slots
        if waiter.request.session_key is not None:
            self._session_queues.setdefault(waiter.request.session_key, deque()).append(waiter)

    def _remove_waiter(self, waiter: _WaitingRequest) -> bool:
        """从所有索引中移除等待项并归还排队槽位。"""
        try:
            self._queues[waiter.request.priority].remove(waiter)
        except ValueError:
            return False

        self._queued_slots -= waiter.request.decode_slots
        session_key = waiter.request.session_key
        if session_key is not None:
            session_queue = self._session_queues[session_key]
            session_queue.remove(waiter)
            if not session_queue:
                self._session_queues.pop(session_key, None)
        if waiter is self._blocked_waiter:
            # 非正常移除（取消、超时、替换、缩容）放弃已经预扣的 gang
            # 服务机会；重置 DRR 状态比跨优先级退款更安全。
            self._clear_backfill_state(reset_deficits=True)
        if not self._queues[waiter.request.priority]:
            self._deficits[waiter.request.priority] = 0
        return True

    def _cancel_waiter_or_take_lease(self, waiter: _WaitingRequest) -> Optional[AdmissionLease]:
        """取消等待项，或取回已并发发放的租约用于释放。"""
        if self._remove_waiter(waiter):
            waiter.future.cancel()
            self._drain()
            return None
        if waiter.future.done() and not waiter.future.cancelled():
            try:
                result = waiter.future.result()
            except BaseException:
                return None
            if isinstance(result, AdmissionLease):
                return result
        return None

    async def _wait_for_lease(self, waiter: _WaitingRequest) -> AdmissionLease:
        """等待租约、优先级提升后的新截止时间或超时。"""
        while True:
            deadline_changed_task = asyncio.create_task(waiter.deadline_changed.wait())
            try:
                done, _ = await asyncio.wait(
                    (waiter.future, deadline_changed_task),
                    timeout=max(0.0, waiter.deadline - self._clock()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not deadline_changed_task.done():
                    deadline_changed_task.cancel()

            if waiter.future in done:
                return waiter.future.result()
            if deadline_changed_task in done:
                waiter.deadline_changed.clear()
                continue
            raise asyncio.TimeoutError

    def _session_is_grantable(self, waiter: _WaitingRequest) -> bool:
        """判断等待项是否满足同 Session 串行和 FIFO 约束。"""
        session_key = waiter.request.session_key
        if session_key is None:
            return True
        if session_key in self._active_sessions:
            return False
        return self._session_queues[session_key][0] is waiter

    def _first_grantable(
        self,
        priority: AdmissionPriority,
        excluded: Optional[_WaitingRequest] = None,
        available_slots: Optional[int] = None,
    ) -> Optional[_WaitingRequest]:
        """按类内 FIFO 返回第一个满足 Session 和可选槽位约束的等待项。"""
        for waiter in self._queues[priority]:
            if waiter is excluded:
                continue
            if not self._session_is_grantable(waiter):
                continue
            if available_slots is not None and waiter.request.decode_slots > available_slots:
                continue
            return waiter
        return None

    def _advance_priority(self) -> None:
        """结束当前 DRR 类访问并移到下一优先级。"""
        self._priority_index = (self._priority_index + 1) % len(self._PRIORITY_ORDER)
        self._priority_visit_started = False

    def _reset_deficits(self) -> None:
        """清空按 Decode 槽位计费的 DRR 临时信用。"""
        for priority in self._PRIORITY_ORDER:
            self._deficits[priority] = 0
        self._priority_index = 0
        self._priority_visit_started = False

    def _select_weighted(
        self,
        capacity: int,
        excluded: Optional[_WaitingRequest] = None,
        available_slots: Optional[int] = None,
    ) -> Optional[_WaitingRequest]:
        """用按槽位计费的 deficit round-robin 选择一个等待项。"""
        candidates = {
            priority: waiter
            for priority in self._PRIORITY_ORDER
            if (
                waiter := self._first_grantable(
                    priority,
                    excluded=excluded,
                    available_slots=available_slots,
                )
            )
            is not None
        }
        for priority in self._PRIORITY_ORDER:
            # 空类和暂时全部受 Session 串行约束的类都不能积攒无限信用。
            if self._first_grantable(priority) is None:
                self._deficits[priority] = 0
        if not candidates:
            return None

        only_priority = next(iter(candidates)) if len(candidates) == 1 else None
        while True:
            priority = self._PRIORITY_ORDER[self._priority_index]
            waiter = candidates.get(priority)
            if waiter is None:
                self._advance_priority()
                continue

            if not self._priority_visit_started:
                self._deficits[priority] += self.policy.weight(priority)
                self._priority_visit_started = True

            # 单一活跃类必须保持 work-conserving；直接补足若干轮 quantum，
            # 避免大 gang 仅因 DRR 信用暂时不足而留下 Decode 空槽。
            if only_priority == priority and self._deficits[priority] < waiter.request.decode_slots:
                quantum = self.policy.weight(priority)
                missing = waiter.request.decode_slots - self._deficits[priority]
                visits = (missing + quantum - 1) // quantum
                self._deficits[priority] += visits * quantum

            if waiter.request.decode_slots <= self._deficits[priority]:
                # 在选择点统一按 choice 槽位扣费。即使 gang 暂时因物理空槽
                # 不足进入 backfill，它的 DRR 服务机会也已经被完整计费。
                self._deficits[priority] -= waiter.request.decode_slots
                return waiter
            self._advance_priority()

    def _clear_backfill_state(self, reset_deficits: bool = False) -> None:
        """清除 gang backfill 或 reservation 的全部临时状态。"""
        self._blocked_waiter = None
        self._backfilled_slots = 0
        self._backfill_limit = 0
        self._reservation_active = False
        if reset_deficits:
            self._reset_deficits()

    def _start_backfill(self, waiter: _WaitingRequest, capacity: int) -> None:
        """为仅受当前可用槽位阻塞的 gang 启动一波有限 backfill。"""
        self._blocked_waiter = waiter
        self._backfilled_slots = 0
        self._backfill_limit = capacity
        self._reservation_active = False

    def _fail_oversized_waiters(self, capacity: int) -> None:
        """容量缩小时失败掉已经不可能原子获得所需槽位的等待项。"""
        for priority in self._PRIORITY_ORDER:
            for waiter in tuple(self._queues[priority]):
                if waiter.request.decode_slots <= capacity:
                    continue
                if self._remove_waiter(waiter) and not waiter.future.done():
                    waiter.future.set_exception(ServerBusyError("PD decode capacity fell below queued request size"))

    def _trim_queue_to_capacity(self, capacity: int) -> None:
        """容量缩小时按低优先级、同级最新顺序恢复等待队列上限。"""
        waiting_capacity = capacity * self.policy.waiting_decode_waves
        slots_to_remove = self._queued_slots - waiting_capacity
        if slots_to_remove <= 0:
            return

        victims = []
        removed_slots = 0
        for priority in reversed(self._PRIORITY_ORDER):
            for waiter in reversed(self._queues[priority]):
                victims.append(waiter)
                removed_slots += waiter.request.decode_slots
                if removed_slots >= slots_to_remove:
                    break
            if removed_slots >= slots_to_remove:
                break

        for victim in victims:
            if self._remove_waiter(victim) and not victim.future.done():
                victim.future.set_exception(ServerBusyError("PD master admission queue capacity shrank"))

    def _grant_waiter(self, waiter: _WaitingRequest) -> bool:
        """从队列移除已由 DRR 计费的等待项并原子发放租约。"""
        if waiter.future.done():
            self._remove_waiter(waiter)
            self._reset_deficits()
            return False

        priority = waiter.request.priority
        if waiter is self._blocked_waiter:
            # 正常兑现 reservation 时保留选择点已经完成的 DRR 扣费。
            self._clear_backfill_state()
        if not self._remove_waiter(waiter):
            self._reset_deficits()
            return False
        if not self._queues[priority]:
            self._deficits[priority] = 0

        lease = self._activate(
            waiter.request,
            waited_seconds=max(0.0, self._clock() - waiter.enqueue_time),
        )
        waiter.future.set_result(lease)
        return True

    def _has_fitting_backfill(self, blocked: _WaitingRequest, available_slots: int) -> bool:
        """判断是否有不含被保护 gang 的请求当前可以填充空槽。"""
        return any(
            self._first_grantable(
                priority,
                excluded=blocked,
                available_slots=available_slots,
            )
            is not None
            for priority in self._PRIORITY_ORDER
        )

    def _drain(self) -> None:
        """持续发放租约，并为受空槽碎片阻塞的 gang 提供有限 backfill。"""
        capacity = self._capacity()
        self._fail_oversized_waiters(capacity)
        self._trim_queue_to_capacity(capacity)

        while self._active_slots < capacity:
            available_slots = capacity - self._active_slots
            blocked = self._blocked_waiter
            if blocked is not None:
                if blocked.future.done():
                    self._remove_waiter(blocked)
                    continue
                if not self._session_is_grantable(blocked):
                    # Session 阻塞不是容量碎片，不能借此获得全局 reservation。
                    self._clear_backfill_state(reset_deficits=True)
                    continue
                if blocked.request.decode_slots <= available_slots:
                    self._grant_waiter(blocked)
                    continue
                if self._reservation_active:
                    break

                remaining_backfill = self._backfill_limit - self._backfilled_slots
                if remaining_backfill <= 0:
                    self._reservation_active = True
                    break
                backfill_slots = min(available_slots, remaining_backfill)
                waiter = self._select_weighted(
                    capacity,
                    excluded=blocked,
                    available_slots=backfill_slots,
                )
                if waiter is None:
                    # 没有可填当前空槽的请求时保留 backfill 机会；稍后到达的小请求
                    # 仍可使用本波预算。只有存在物理上可填、但会越过预算的请求时
                    # 才立即转入 reservation。
                    if self._has_fitting_backfill(blocked, available_slots):
                        self._reservation_active = True
                    break
                granted_slots = waiter.request.decode_slots
                if not self._grant_waiter(waiter):
                    continue
                self._backfilled_slots += granted_slots
                if self._backfilled_slots >= self._backfill_limit:
                    self._reservation_active = True
                continue

            waiter = self._select_weighted(capacity)
            if waiter is None:
                break
            if waiter.request.decode_slots > available_slots:
                # 此处 waiter 已满足 Session 约束且不超过总容量，唯一阻塞原因
                # 是当前空槽不足，因此可以安全启动 bounded backfill。
                self._start_backfill(waiter, capacity)
                continue
            self._grant_waiter(waiter)
        self._notify_state_change()

    def _notify_state_change(self) -> None:
        """通知外部记录最新的准入状态。"""
        if self._state_change_callback is not None:
            self._state_change_callback(self)
