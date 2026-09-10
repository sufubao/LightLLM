"""Pinned checkpoint storage reserved before requests enter the inference loop."""

import bisect
import math
import queue
import threading
import weakref

import torch


class PinnedCheckpointArena:
    """Split one bounded allocation into independently owned tensor storages.

    A tensor view of the arena would make torch.save serialize the entire cache.
    frombuffer instead gives each window a storage of precisely its own size.
    The storage retains the memoryview; its finalizer returns the range only
    after the last tensor/view has released it. This object never owns allocated
    windows, entries, or the cache, so that finalizer cannot create an owner cycle.

    Callers must still fence asynchronous CUDA users before releasing storage:
    these window storages are not owned by PyTorch's pinned caching allocator.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self._buffer = memoryview(torch.empty(self.capacity, dtype=torch.uint8, pin_memory=True).numpy())
        self._free = [(0, self.capacity)] if self.capacity else []
        self._lock = threading.Lock()
        self._released = queue.SimpleQueue()

    def allocate(self, shape, dtype):
        shape = tuple(int(dim) for dim in shape)
        element_size = torch.empty((), dtype=dtype).element_size()
        size = math.prod(shape) * element_size
        if size == 0:
            return torch.empty(shape, dtype=dtype, device="cpu")
        if size < 0 or any(dim < 0 for dim in shape):
            raise ValueError("invalid checkpoint allocation shape")
        with self._lock:
            self._reclaim()
            best = None
            for index, (offset, available) in enumerate(self._free):
                start = (offset + element_size - 1) // element_size * element_size
                if start + size <= offset + available and (best is None or available < best[0]):
                    best = (available, index, offset, start)
            if best is None:
                return None
            available, index, offset, start = best
            remainder = []
            if start > offset:
                remainder.append((offset, start - offset))
            end = start + size
            if end < offset + available:
                remainder.append((end, offset + available - end))
            self._free[index : index + 1] = remainder
        try:
            window = self._buffer[start:end]
            # GC can run during free-list mutation. Its callback must neither
            # acquire our lock nor mutate the list being searched. SimpleQueue
            # supports reentrant put from finalizers; allocation drains it.
            weakref.finalize(window, self._released.put, (start, size))
        except BaseException:
            self._released.put((start, size))
            raise
        try:
            return torch.frombuffer(window, dtype=dtype).reshape(shape)
        finally:
            # An allocator error's traceback can outlive admission handling.
            # Successful storage owns the window; failed frames must not own it.
            window = None

    def _reclaim(self):
        while True:
            try:
                start, size = self._released.get_nowait()
            except queue.Empty:
                return
            index = bisect.bisect_left(self._free, (start,))
            if index and self._free[index - 1][0] + self._free[index - 1][1] == start:
                previous, previous_size = self._free.pop(index - 1)
                index -= 1
                start, size = previous, previous_size + size
            if index < len(self._free) and start + size == self._free[index][0]:
                _, next_size = self._free.pop(index)
                size += next_size
            self._free.insert(index, (start, size))
