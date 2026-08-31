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
        if priority == AdmissionPriority.CONTINUATION:
            return self.continuation_weight
        if priority == AdmissionPriority.PROBABLE_CACHE_HIT:
            return self.probable_cache_hit_weight
        return self.cold_weight

    def max_wait_seconds(self, priority: AdmissionPriority) -> float:
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
    estimated_uncached_work: int = 0

    def __post_init__(self) -> None:
        if self.decode_slots < 1:
            raise ValueError("decode_slots must be positive")
        if self.estimated_uncached_work < 0:
            raise ValueError("estimated_uncached_work must be non-negative")


@dataclass(frozen=True, slots=True)
class CacheCapacitySnapshot:
    """当前 PD Master 可使用的 Prefill Radix cache 份额。"""

    total_tokens: int
    capacity_tokens: int

    def __post_init__(self) -> None:
        if self.total_tokens < 0 or self.capacity_tokens < 0:
            raise ValueError("cache token counts must be non-negative")

    @property
    def free_tokens(self) -> int:
        return max(0, self.capacity_tokens - self.total_tokens)


class SessionTracker:
    """只把服务端已经成功观察过的 Session 视为连续会话。"""

    def __init__(
        self,
        ttl_seconds: float,
        max_sessions: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._clock = clock
        self._last_success: OrderedDict[str, float] = OrderedDict()

    def is_continuation(self, session_key: Optional[str]) -> bool:
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
        if not session_key:
            return
        self._last_success[session_key] = self._clock()
        self._last_success.move_to_end(session_key)
        while len(self._last_success) > self.max_sessions:
            self._last_success.popitem(last=False)


@dataclass(slots=True)
class _WaitingRequest:
    sequence_id: int
    request: AdmissionRequest
    enqueue_time: float
    future: asyncio.Future


class AdmissionLease:
    """一次已经获得的 Decode 容量租约。"""

    def __init__(
        self,
        controller: "PDAdmissionController",
        request: AdmissionRequest,
        waited_seconds: float,
    ) -> None:
        self._controller = controller
        self.request = request
        self.waited_seconds = waited_seconds
        self._released = False

    async def __aenter__(self) -> "AdmissionLease":
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()

    def release(self) -> None:
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
        cache_capacity_provider: Optional[Callable[[], Optional[CacheCapacitySnapshot]]] = None,
        policy: Optional[AdmissionPolicy] = None,
        clock: Callable[[], float] = time.monotonic,
        state_change_callback: Optional[Callable[["PDAdmissionController"], None]] = None,
    ) -> None:
        self.policy = policy or AdmissionPolicy()
        self._decode_capacity_provider = decode_capacity_provider
        self._cache_capacity_provider = cache_capacity_provider
        self._clock = clock
        self._state_change_callback = state_change_callback
        self._active_slots = 0
        self._active_cold_slots = 0
        self._average_cold_uncached_tokens: Optional[float] = None
        self._active_sessions = set()
        self._queues: Dict[AdmissionPriority, Deque[_WaitingRequest]] = {
            priority: deque() for priority in self._PRIORITY_ORDER
        }
        self._session_queues: Dict[str, Deque[_WaitingRequest]] = {}
        self._queued_slots = 0
        self._sequence_id = 0
        self._schedule = self._build_schedule()
        self._schedule_index = 0

    @property
    def active_slots(self) -> int:
        return self._active_slots

    @property
    def active_cold_slots(self) -> int:
        return self._active_cold_slots

    @property
    def cold_capacity(self) -> int:
        """返回在当前缓存余量下允许并发的冷请求槽位数。"""
        decode_capacity = self._capacity()
        if (
            decode_capacity <= 0
            or self._cache_capacity_provider is None
            or self._average_cold_uncached_tokens is None
            or self._average_cold_uncached_tokens <= 0
        ):
            return decode_capacity

        snapshot = self._cache_capacity_provider()
        if snapshot is None or snapshot.capacity_tokens <= 0:
            return decode_capacity

        # 剩余缓存能容纳几个“平均冷请求”，就开放几个冷槽位；至少保留一个
        # 探索槽位，使系统在缓存已满时仍能接纳新会话并持续获得反馈。
        requests_fitting_in_cache = int(snapshot.free_tokens / self._average_cold_uncached_tokens)
        return min(decode_capacity, max(1, requests_fitting_in_cache))

    @property
    def queued_slots(self) -> int:
        return self._queued_slots

    @property
    def queued_request_count(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    def _capacity(self) -> int:
        return max(0, int(self._decode_capacity_provider()))

    def _build_schedule(self) -> tuple[AdmissionPriority, ...]:
        remaining = {priority: self.policy.weight(priority) for priority in self._PRIORITY_ORDER}
        schedule = []
        while any(remaining.values()):
            for priority in self._PRIORITY_ORDER:
                if remaining[priority] > 0:
                    schedule.append(priority)
                    remaining[priority] -= 1
        return tuple(schedule)

    async def acquire(self, request: AdmissionRequest) -> AdmissionLease:
        capacity = self._capacity()
        if capacity <= 0 or request.decode_slots > capacity:
            raise ServerBusyError("PD decode capacity is unavailable")

        if self.queued_request_count == 0 and self._can_activate(request):
            lease = self._activate(request, waited_seconds=0.0)
            self._notify_state_change()
            return lease

        loop = asyncio.get_running_loop()
        waiter = _WaitingRequest(
            sequence_id=self._sequence_id,
            request=request,
            enqueue_time=self._clock(),
            future=loop.create_future(),
        )
        self._sequence_id += 1

        if not self._make_queue_room(waiter):
            raise ServerBusyError("PD master admission queue is full")

        self._enqueue(waiter)
        self._drain()

        try:
            return await asyncio.wait_for(
                asyncio.shield(waiter.future),
                timeout=self.policy.max_wait_seconds(request.priority),
            )
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
        self._drain()

    def record_prefill_result(
        self,
        request: AdmissionRequest,
        prompt_tokens: int,
        cached_tokens: int,
    ) -> None:
        """用冷请求的真实未命中量更新下一轮冷容量。"""
        if request.priority != AdmissionPriority.COLD:
            return

        prompt_tokens = max(0, int(prompt_tokens))
        cached_tokens = min(prompt_tokens, max(0, int(cached_tokens)))
        uncached_tokens = prompt_tokens - cached_tokens

        # 一个 Decode 波次作为自适应窗口：容量越大，单个样本对均值的影响越小。
        sample_window = max(1, self._capacity())
        alpha = 2.0 / (sample_window + 1.0)
        if self._average_cold_uncached_tokens is None:
            self._average_cold_uncached_tokens = float(uncached_tokens)
        else:
            self._average_cold_uncached_tokens += alpha * (uncached_tokens - self._average_cold_uncached_tokens)
        self._drain()

    def promote_session(self, session_key: Optional[str]) -> None:
        """把同一 Session 尚未派发的请求提升为连续会话优先级。"""
        if not session_key:
            return
        session_queue = self._session_queues.get(session_key)
        if not session_queue:
            return

        for waiter in tuple(session_queue):
            old_priority = waiter.request.priority
            if old_priority == AdmissionPriority.CONTINUATION:
                continue
            self._queues[old_priority].remove(waiter)
            waiter.request = replace(waiter.request, priority=AdmissionPriority.CONTINUATION)
            self._queues[AdmissionPriority.CONTINUATION].append(waiter)
        self._drain()

    def _has_cold_capacity(self, request: AdmissionRequest) -> bool:
        if request.priority != AdmissionPriority.COLD:
            return True
        return self._active_cold_slots + request.decode_slots <= self.cold_capacity

    def _can_activate(self, request: AdmissionRequest) -> bool:
        if self._active_slots + request.decode_slots > self._capacity():
            return False
        if not self._has_cold_capacity(request):
            return False
        return request.session_key is None or request.session_key not in self._active_sessions

    def _activate(self, request: AdmissionRequest, waited_seconds: float) -> AdmissionLease:
        self._active_slots += request.decode_slots
        if request.priority == AdmissionPriority.COLD:
            self._active_cold_slots += request.decode_slots
        if request.session_key is not None:
            self._active_sessions.add(request.session_key)
        return AdmissionLease(self, request, waited_seconds)

    def _release(self, lease: AdmissionLease) -> None:
        request = lease.request
        self._active_slots -= request.decode_slots
        if request.priority == AdmissionPriority.COLD:
            self._active_cold_slots -= request.decode_slots
        if self._active_slots < 0:
            raise RuntimeError("PD admission active slot count became negative")
        if self._active_cold_slots < 0:
            raise RuntimeError("PD admission active cold slot count became negative")
        if request.session_key is not None:
            self._active_sessions.discard(request.session_key)
        self._drain()

    def _waiting_capacity(self) -> int:
        return self._capacity() * self.policy.waiting_decode_waves

    def _make_queue_room(self, incoming: _WaitingRequest) -> bool:
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
        self._queues[waiter.request.priority].append(waiter)
        self._queued_slots += waiter.request.decode_slots
        if waiter.request.session_key is not None:
            self._session_queues.setdefault(waiter.request.session_key, deque()).append(waiter)

    def _remove_waiter(self, waiter: _WaitingRequest) -> bool:
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
        return True

    def _cancel_waiter_or_take_lease(self, waiter: _WaitingRequest) -> Optional[AdmissionLease]:
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

    def _first_grantable(self, priority: AdmissionPriority) -> Optional[_WaitingRequest]:
        candidates = []
        available_decode_slots = self._capacity() - self._active_slots
        for waiter in self._queues[priority]:
            session_key = waiter.request.session_key
            if session_key is not None and session_key in self._active_sessions:
                continue
            if session_key is not None and self._session_queues[session_key][0] is not waiter:
                continue
            if priority != AdmissionPriority.COLD:
                return waiter
            if waiter.request.decode_slots > self.cold_capacity:
                continue
            if waiter.request.decode_slots <= available_decode_slots and not self._has_cold_capacity(waiter.request):
                continue
            candidates.append(waiter)

        if not candidates:
            return None
        # 冷请求内部优先处理预计新增缓存最少的任务，在同等代价下保持 FIFO。
        return min(
            candidates,
            key=lambda waiter: (
                waiter.request.estimated_uncached_work,
                waiter.sequence_id,
            ),
        )

    def _select_next(self) -> Optional[_WaitingRequest]:
        schedule_size = len(self._schedule)
        for offset in range(schedule_size):
            index = (self._schedule_index + offset) % schedule_size
            waiter = self._first_grantable(self._schedule[index])
            if waiter is not None:
                self._schedule_index = (index + 1) % schedule_size
                return waiter
        return None

    def _drain(self) -> None:
        while self._active_slots < self._capacity():
            schedule_index = self._schedule_index
            waiter = self._select_next()
            if waiter is None:
                break
            if self._active_slots + waiter.request.decode_slots > self._capacity():
                # 为需要多个 choice slot 的老请求保留逐步释放出来的容量，避免永久饥饿。
                self._schedule_index = schedule_index
                break
            if not self._has_cold_capacity(waiter.request):
                self._schedule_index = schedule_index
                break
            if not self._remove_waiter(waiter):
                continue
            lease = self._activate(
                waiter.request,
                waited_seconds=max(0.0, self._clock() - waiter.enqueue_time),
            )
            if waiter.future.done():
                lease.release()
                continue
            waiter.future.set_result(lease)
        self._notify_state_change()

    def _notify_state_change(self) -> None:
        if self._state_change_callback is not None:
            self._state_change_callback(self)
