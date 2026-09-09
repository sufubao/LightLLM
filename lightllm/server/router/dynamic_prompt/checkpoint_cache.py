"""Bounded, rank-local CPU storage for exact hybrid-model checkpoints.

The caller coordinates candidate selection and commit across its TP group.  This
module never performs distributed collectives.  In particular, a successful
``prepare`` is private until every participating rank can commit the checkpoint.

Transfers are synchronous on the producing CUDA stream in this first version.
Sources must already describe a frozen, committed model position; reading a live
request's state after another forward has started is not safe.  Returned leases
own the CPU source until onload/export finishes, including across ``clear``.
"""

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from numbers import Integral
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(eq=False)
class _Page:
    key: str
    tokens: Tuple[int, ...]
    origins: Tuple[int, ...]
    tensor: torch.Tensor
    references: int = 0

    @property
    def nbytes(self):
        return self.tensor.numel() * self.tensor.element_size()


@dataclass(eq=False)
class _Entry:
    serial: int
    epoch: int
    namespace: str
    length: int
    pages: Tuple[_Page, ...]
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    output_seed: Optional[torch.Tensor]
    leases: int = 0
    ready: bool = False
    retired: bool = False

    @property
    def state_bytes(self):
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.conv_state, self.ssm_state, self.output_seed)
            if tensor is not None
        )

    def tokens(self):
        return tuple(token for page in self.pages for token in page.tokens)

    def origins(self):
        return tuple(origin for page in self.pages for origin in page.origins)


@dataclass
class _Node:
    edge: Tuple[int, ...] = ()
    children: Dict[int, "_Node"] = field(default_factory=dict)
    entry: Optional[_Entry] = None
    parent: Optional["_Node"] = None


@dataclass
class PendingCheckpoint:
    """Private, fully copied data.  Only its originating cache may publish it."""

    cache: "CpuCheckpointCache"
    entry: _Entry
    consumed: bool = False

    @property
    def length(self):
        return self.entry.length


class CheckpointLease:
    """Read-only source ownership; close after all consumers have completed."""

    def __init__(self, cache, entry):
        self.cache = cache
        self.entry = entry
        self.closed = False

    @property
    def length(self):
        return self.entry.length

    @property
    def conv_state(self):
        self._check_open()
        return self.entry.conv_state

    @property
    def ssm_state(self):
        self._check_open()
        return self.entry.ssm_state

    @property
    def output_seed(self):
        self._check_open()
        return self.entry.output_seed

    @property
    def page_keys(self):
        self._check_open()
        return [page.key for page in self.entry.pages]

    @property
    def origins(self):
        self._check_open()
        return torch.tensor(self.entry.origins(), dtype=torch.int64, device="cpu")

    def _check_open(self):
        if self.closed:
            raise RuntimeError("checkpoint lease has been released")

    def close(self):
        self.cache.release(self)

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()


class CpuCheckpointCache:
    """Compressed token-prefix directory with shared, immutable CPU KV pages.

    ``max_bytes`` includes physical page capacity, independent state/seed tensors,
    unpublished preparations, and retired data still held by leases.  Metadata is
    bounded separately by ``max_entries``.  A tail page uses the same allocation
    size as a full page, but only its valid token range is copied or exported.
    """

    def __init__(
        self,
        max_bytes: int,
        max_entries: int,
        page_size: int = 8192,
        pin_memory: bool = True,
        draft_tail_dependency: bool = False,
    ):
        if max_bytes < 0 or max_entries < 0 or page_size <= 0:
            raise ValueError("invalid checkpoint cache capacity")
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self.page_size = int(page_size)
        self.pin_memory = pin_memory
        self.draft_tail_dependency = bool(draft_tail_dependency)
        self._lock = threading.RLock()
        self._epoch = 0
        self._serial = 0
        self._bytes = 0
        self._roots: Dict[str, _Node] = {}
        self._pages: Dict[str, _Page] = {}
        self._entries: Dict[int, _Entry] = {}
        self._nodes: Dict[int, _Node] = {}
        self._lru: OrderedDict[int, _Entry] = OrderedDict()

    @staticmethod
    def _tokens(tokens: Sequence[int]):
        if isinstance(tokens, torch.Tensor):
            if tokens.device.type != "cpu":
                raise ValueError("checkpoint token IDs must be on CPU")
            tokens = tokens.tolist()
        return tuple(int(token) for token in tokens)

    @staticmethod
    def _origins(origins, length):
        if origins is None:
            return (0,) * length
        if isinstance(origins, torch.Tensor):
            if origins.device.type != "cpu" or origins.dtype != torch.int64 or origins.ndim != 1:
                raise ValueError("checkpoint origins must be a CPU int64 vector")
            origins = origins.tolist()
        origins = tuple(origins)
        if len(origins) != length or any(
            isinstance(origin, bool) or not isinstance(origin, Integral) or not 0 <= origin < 2 ** 63
            for origin in origins
        ):
            raise ValueError("checkpoint origins must contain one nonnegative int64 per token")
        return tuple(int(origin) for origin in origins)

    def _page_specs(self, tokens, namespace, origins):
        return self._derive_page_specs(tokens, namespace, self.page_size, self.draft_tail_dependency, origins)

    @staticmethod
    def derive_page_keys(
        tokens,
        namespace="default",
        page_size=8192,
        draft_tail_dependency=False,
        origins=None,
    ):
        """Rekey a validated transport payload after an explicit layout conversion."""
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        tokens = CpuCheckpointCache._tokens(tokens)
        origins = CpuCheckpointCache._origins(origins, len(tokens))
        return [
            key
            for key, _, _ in CpuCheckpointCache._derive_page_specs(
                tokens, namespace, page_size, draft_tail_dependency, origins
            )
        ]

    @staticmethod
    def _derive_page_specs(tokens, namespace, page_size, draft_tail_dependency, origins):
        digest = hashlib.blake2b(digest_size=20, person=b"llm-checkpoint-2")
        encoded_namespace = namespace.encode("utf-8")
        digest.update(len(encoded_namespace).to_bytes(8, "little"))
        digest.update(encoded_namespace)
        specs = []
        for start in range(0, len(tokens), page_size):
            part = tokens[start : start + page_size]
            digest.update(np.asarray(part, dtype="<i8").tobytes())
            digest.update(np.asarray(origins[start : start + len(part)], dtype="<i8").tobytes())
            page_digest = digest.copy()
            if draft_tail_dependency:
                # Shifted draft KV at a page's last slot depends on the next
                # token.  A terminal page must never alias its later, interior
                # version even though their target token prefixes are identical.
                end = start + len(part)
                if end < len(tokens):
                    page_digest.update(b"\x01")
                    page_digest.update(np.asarray([tokens[end]], dtype="<i8").tobytes())
                else:
                    page_digest.update(b"\x00")
            specs.append((page_digest.hexdigest(), part, start))
        return specs

    @staticmethod
    def _common(left, right):
        for index, (a, b) in enumerate(zip(left, right)):
            if a != b:
                return index
        return min(len(left), len(right))

    def _candidates(self, tokens, namespace, max_length, require_output_seed):
        node = self._roots.get(namespace)
        offset = 0
        found = []
        while node is not None:
            entry = node.entry
            if entry is not None and entry.ready and not entry.retired:
                if not require_output_seed or entry.output_seed is not None:
                    found.append(entry)
            if offset >= min(len(tokens), max_length):
                break
            child = node.children.get(tokens[offset])
            if child is None or offset + len(child.edge) > max_length:
                break
            if tuple(tokens[offset : offset + len(child.edge)]) != child.edge:
                break
            offset += len(child.edge)
            node = child
        return found

    def candidate_lengths(self, tokens, max_length=None, require_output_seed=False, namespace="default"):
        """Return deepest-first candidates; callers intersect these across TP."""
        tokens = self._tokens(tokens)
        limit = len(tokens) if max_length is None else min(len(tokens), int(max_length))
        with self._lock:
            return [entry.length for entry in reversed(self._candidates(tokens, namespace, limit, require_output_seed))]

    def acquire(self, tokens, length: int, namespace="default"):
        """Acquire exactly ``length``; never silently select a shorter prefix."""
        tokens = self._tokens(tokens)
        with self._lock:
            candidates = self._candidates(tokens, namespace, min(length, len(tokens)), False)
            if not candidates or candidates[-1].length != length:
                return None
            entry = candidates[-1]
            entry.leases += 1
            self._lru.move_to_end(entry.serial)
            return CheckpointLease(self, entry)

    def match(self, tokens, max_length=None, require_output_seed=False, namespace="default"):
        tokens = self._tokens(tokens)
        with self._lock:
            lengths = self.candidate_lengths(tokens, max_length, require_output_seed, namespace)
            return self.acquire(tokens, lengths[0], namespace) if lengths else None

    def _insert_node(self, tokens, entry):
        node = self._roots.setdefault(entry.namespace, _Node())
        offset = 0
        while offset < len(tokens):
            child = node.children.get(tokens[offset])
            if child is None:
                child = _Node(edge=tokens[offset:], parent=node)
                node.children[child.edge[0]] = child
                node = child
                break
            common = self._common(tokens[offset:], child.edge)
            if common < len(child.edge):
                middle = _Node(edge=child.edge[:common], parent=node)
                node.children[middle.edge[0]] = middle
                child.edge = child.edge[common:]
                child.parent = middle
                middle.children[child.edge[0]] = child
                node = middle
            else:
                node = child
            offset += common
        old_entry = node.entry
        if old_entry is not None:
            self._retire(old_entry, prune=False)
        node.entry = entry
        self._nodes[entry.serial] = node

    def _remove_node(self, entry, prune=True):
        node = self._nodes.pop(entry.serial, None)
        if node is None:
            return
        node.entry = None
        while prune and node.parent is not None and node.entry is None:
            parent = node.parent
            if not node.children:
                del parent.children[node.edge[0]]
            elif len(node.children) == 1:
                child = next(iter(node.children.values()))
                child.edge = node.edge + child.edge
                child.parent = parent
                parent.children[node.edge[0]] = child
            else:
                break
            node = parent

    def _drop_entry(self, entry):
        self._entries.pop(entry.serial)
        self._bytes -= entry.state_bytes
        for page in entry.pages:
            self._unref_page(page)

    def _unref_page(self, page):
        page.references -= 1
        if page.references == 0:
            if self._pages.get(page.key) is page:
                del self._pages[page.key]
            self._bytes -= page.nbytes

    def _retire(self, entry, prune=True):
        entry.ready = False
        entry.retired = True
        self._lru.pop(entry.serial, None)
        self._remove_node(entry, prune)
        if entry.leases == 0:
            self._drop_entry(entry)

    def _make_room(self, needed_bytes):
        if needed_bytes > self.max_bytes or self.max_entries == 0:
            return False
        for entry in list(self._lru.values()):
            if self._bytes + needed_bytes <= self.max_bytes and len(self._entries) < self.max_entries:
                break
            if entry.leases == 0:
                self._retire(entry)
        return self._bytes + needed_bytes <= self.max_bytes and len(self._entries) < self.max_entries

    def _empty_cpu(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=self.pin_memory)

    @staticmethod
    def _finish_stream(tensors):
        devices = {tensor.device for tensor in tensors if tensor is not None and tensor.is_cuda}
        for device in devices:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            event.synchronize()

    def _prepare(
        self,
        tokens,
        origins,
        namespace,
        kv_shape,
        kv_dtype,
        conv_state,
        ssm_state,
        output_seed,
        copy_page,
    ):
        if not tokens:
            return None
        if conv_state is None or ssm_state is None:
            raise ValueError("both conv and SSM state are required")
        specs = self._page_specs(tokens, namespace, origins)
        # Hold matching sources while admission evicts other entries.  A COW
        # tail copies its existing CPU prefix; only the new suffix comes from GPU.
        candidates = self._candidates(tokens, namespace, len(tokens), False)
        base_entry = candidates[-1] if candidates else None
        base_pages = base_entry.pages if base_entry is not None else ()
        common_origins = self._common(origins, base_entry.origins()) if base_entry is not None else 0
        del candidates
        page_bytes = int(np.prod(kv_shape)) * torch.empty((), dtype=kv_dtype).element_size()
        state_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (conv_state, ssm_state, output_seed)
            if tensor is not None
        )
        existing = []
        for key, part, start in specs:
            page = self._pages.get(key)
            # Verify token content as well as its digest, and reject layout reuse.
            if page is not None and (
                page.tokens != part
                or page.origins != origins[start : start + len(part)]
                or tuple(page.tensor.shape) != tuple(kv_shape)
                or page.tensor.dtype != kv_dtype
            ):
                raise ValueError("checkpoint namespace reused with incompatible content or KV layout")
            existing.append(page)
        needed = state_bytes + sum(page is None for page in existing) * page_bytes
        # Exact shared pages can otherwise disappear when their owning checkpoint
        # is evicted by admission.  Temporary refs make the plan stable.
        for page in existing:
            if page is not None:
                page.references += 1
        for page in base_pages:
            page.references += 1
        created = []
        state_tensors = []
        try:
            if not self._make_room(needed):
                return None
            for index, ((key, part, start), page) in enumerate(zip(specs, existing)):
                if page is None:
                    tensor = self._empty_cpu(kv_shape, kv_dtype)
                    copied = 0
                    if index < len(base_pages):
                        old = base_pages[index]
                        copied = min(len(old.tokens), len(part), max(0, common_origins - start))
                        if old.tokens[:copied] != part[:copied]:
                            copied = 0
                        if self.draft_tail_dependency and copied == len(old.tokens):
                            # A previous checkpoint's terminal draft slot was
                            # not reusable.  Refresh it as it becomes interior.
                            copied = max(0, copied - 1)
                        if copied:
                            tensor[:, :copied].copy_(old.tensor[:, :copied])
                    copy_page(tensor, start, len(part), copied, key)
                    page = _Page(key, part, origins[start : start + len(part)], tensor)
                    created.append(page)
            for source in (conv_state, ssm_state, output_seed):
                destination = None
                if source is not None:
                    destination = self._empty_cpu(source.shape, source.dtype)
                    destination.copy_(source, non_blocking=source.is_cuda)
                state_tensors.append(destination)
            # The producer owns all sources through this fence.  No directory or
            # page becomes visible while a device copy is still outstanding.
            self._finish_stream((conv_state, ssm_state, output_seed))
            pages = iter(created)
            entry_pages = tuple(page if page is not None else next(pages) for page in existing)
            self._serial += 1
            entry = _Entry(self._serial, self._epoch, namespace, len(tokens), entry_pages, *state_tensors)
            for page in entry_pages:
                page.references += 1
            for page in created:
                self._pages[page.key] = page
                self._bytes += page.nbytes
            self._bytes += entry.state_bytes
            self._entries[entry.serial] = entry
            return PendingCheckpoint(self, entry)
        except BaseException:
            # A later state allocation may fail after an earlier asynchronous
            # D2H copy. Fence the producer before releasing temporary buffers
            # or returning its staging slot to the next capture.
            try:
                self._finish_stream((conv_state, ssm_state, output_seed))
            finally:
                created.clear()
                state_tensors.clear()
                tensor = destination = page = None
            raise
        finally:
            for page in existing:
                if page is not None:
                    self._unref_page(page)
            for page in base_pages:
                self._unref_page(page)

    def prepare(
        self,
        tokens,
        kv_buffer,
        mem_indexes,
        conv_state,
        ssm_state,
        output_seed=None,
        namespace="default",
        frozen_tail_kv=None,
        origins=None,
    ):
        """Copy a frozen position; call commit only after every TP rank succeeds.

        ``kv_buffer`` has shape [layers, token_slots, 2 * heads, head_dim].
        mem_indexes contains the complete logical prefix, in token order.
        origins identifies the actual forward that produced each token's KV;
        equal token IDs alone do not make numerically different histories safe
        to combine with the captured state. Production callers supply origins.
        """
        tokens = self._tokens(tokens)
        origins = self._origins(origins, len(tokens))
        if kv_buffer.ndim != 4 or mem_indexes.ndim != 1 or len(mem_indexes) != len(tokens):
            raise ValueError("invalid KV shape or checkpoint token indexes")
        if frozen_tail_kv is not None and tuple(frozen_tail_kv.shape) != (
            kv_buffer.shape[0],
            *kv_buffer.shape[2:],
        ):
            raise ValueError("frozen tail must contain one complete KV token")
        kv_shape = (kv_buffer.shape[0], self.page_size, *kv_buffer.shape[2:])

        def copy_page(destination, start, valid, copied, _key):
            try:
                if copied < valid:
                    indexes = mem_indexes[start + copied : start + valid].to(device=kv_buffer.device, dtype=torch.long)
                    source = kv_buffer.index_select(1, indexes)
                    if frozen_tail_kv is not None and start + valid == len(tokens):
                        source[:, -1].copy_(frozen_tail_kv)
                    destination[:, copied:valid].copy_(source, non_blocking=source.is_cuda)
            finally:
                # State can already be on CPU; in that case it cannot fence
                # this KV producer, including a partially completed copy.
                try:
                    self._finish_stream((kv_buffer,))
                finally:
                    source = indexes = None

        with self._lock:
            return self._prepare(
                tokens,
                origins,
                namespace,
                kv_shape,
                kv_buffer.dtype,
                conv_state,
                ssm_state,
                output_seed,
                copy_page,
            )

    def commit(self, pending):
        """Publish one prepared rank shard after coordinated TP admission."""
        with self._lock:
            if pending.cache is not self or pending.consumed:
                raise ValueError("invalid or already consumed checkpoint preparation")
            pending.consumed = True
            entry = pending.entry
            if entry.epoch != self._epoch:
                self._retire(entry)
                return False
            self._insert_node(entry.tokens(), entry)
            entry.ready = True
            self._lru[entry.serial] = entry
            return True

    def discard(self, pending):
        with self._lock:
            if pending.cache is not self or pending.consumed:
                raise ValueError("invalid or already consumed checkpoint preparation")
            pending.consumed = True
            self._retire(pending.entry)

    def release(self, lease):
        with self._lock:
            if lease.cache is not self:
                raise ValueError("lease belongs to another cache")
            if lease.closed:
                return
            lease.closed = True
            entry = lease.entry
            entry.leases -= 1
            if entry.retired and entry.leases == 0:
                self._drop_entry(entry)

    def load_kv(self, lease, kv_buffer, mem_indexes, start=0):
        """Load [start, lease.length) into the corresponding destination indexes.

        ``mem_indexes`` contains only that missing range, not the reused prefix.
        The lease must remain open through this method.  Destination state is
        restored separately by the model-specific committed-state adapter.
        """
        with self._lock:
            lease._check_open()
            if lease.cache is not self or not 0 <= start <= lease.length:
                raise ValueError("invalid checkpoint lease or load range")
            if len(mem_indexes) != lease.length - start:
                raise ValueError("destination indexes must cover precisely the missing prefix range")
            for index, page in enumerate(lease.entry.pages):
                page_start = index * self.page_size
                begin = max(start, page_start)
                end = page_start + len(page.tokens)
                if begin >= end:
                    continue
                source = page.tensor[:, begin - page_start : end - page_start].to(
                    device=kv_buffer.device, non_blocking=kv_buffer.is_cuda
                )
                indexes = mem_indexes[begin - start : end - start].to(device=kv_buffer.device, dtype=torch.long)
                kv_buffer.index_copy_(1, indexes, source)
            self._finish_stream((kv_buffer,))

    def export(self, lease, known_page_keys=()):
        """Return torch.save-compatible CPU payload; retain lease until serialized."""
        with self._lock:
            lease._check_open()
            if lease.cache is not self:
                raise ValueError("lease belongs to another cache")
            known = set(known_page_keys)
            entry = lease.entry
            return {
                "version": 1,
                "namespace": entry.namespace,
                "page_size": self.page_size,
                "draft_tail_dependency": self.draft_tail_dependency,
                "length": entry.length,
                "tokens": list(entry.tokens()),
                "origins": list(entry.origins()),
                "page_keys": lease.page_keys,
                # torch.save serializes an entire backing storage for views.
                # Own the valid tail bytes so neither page padding nor unrelated
                # old allocation contents enter the wire payload.
                "pages": {
                    page.key: (
                        page.tensor
                        if len(page.tokens) == self.page_size
                        else page.tensor[:, : len(page.tokens)].clone(memory_format=torch.contiguous_format)
                    )
                    for page in entry.pages
                    if page.key not in known
                },
                "conv_state": entry.conv_state,
                "ssm_state": entry.ssm_state,
                "output_seed": entry.output_seed,
            }

    def missing_page_keys(self, page_keys):
        with self._lock:
            return [key for key in page_keys if key not in self._pages]

    def prepare_import(self, payload):
        """Prepare a CPU payload, returning None if omitted base pages disappeared."""
        if payload["version"] != 1 or payload["page_size"] != self.page_size:
            raise ValueError("incompatible checkpoint format")
        if bool(payload.get("draft_tail_dependency", False)) != self.draft_tail_dependency:
            raise ValueError("incompatible checkpoint draft tail dependency")
        tokens = self._tokens(payload["tokens"])
        if "origins" not in payload:
            raise ValueError("checkpoint manifest lacks KV computation origins")
        origins = self._origins(payload["origins"], len(tokens))
        namespace = payload["namespace"]
        specs = self._page_specs(tokens, namespace, origins)
        if len(tokens) != payload["length"] or [key for key, _, _ in specs] != payload["page_keys"]:
            raise ValueError("checkpoint manifest does not match its token prefix and origins")
        provided = payload["pages"]
        with self._lock:
            if any(key not in provided and key not in self._pages for key, _, _ in specs):
                return None
            if not specs:
                return None
            first_key = specs[0][0]
            sample = provided[first_key] if first_key in provided else self._pages[first_key].tensor
            if sample.ndim != 4 or sample.device.type != "cpu":
                raise ValueError("checkpoint imports require CPU KV tensors")
            kv_shape = (sample.shape[0], self.page_size, *sample.shape[2:])
            for key, part, _ in specs:
                tensor = provided.get(key)
                if tensor is not None and (
                    tensor.device.type != "cpu"
                    or tuple(tensor.shape) != (sample.shape[0], len(part), *sample.shape[2:])
                    or tensor.dtype != sample.dtype
                ):
                    raise ValueError("invalid imported KV page layout")
            for name in ("conv_state", "ssm_state", "output_seed"):
                tensor = payload[name]
                if tensor is not None and tensor.device.type != "cpu":
                    raise ValueError("checkpoint imports require CPU state tensors")

            def copy_page(destination, _start, valid, copied, key):
                if copied < valid:
                    destination[:, copied:valid].copy_(provided[key][:, copied:valid])

            return self._prepare(
                tokens,
                origins,
                namespace,
                kv_shape,
                sample.dtype,
                payload["conv_state"],
                payload["ssm_state"],
                payload["output_seed"],
                copy_page,
            )

    def clear(self):
        """Invalidate lookup immediately; leased and pending buffers remain owned."""
        with self._lock:
            self._epoch += 1
            for entry in list(self._lru.values()):
                self._retire(entry)
            self._roots.clear()
            self._nodes.clear()
            self._pages.clear()

    def stats(self):
        with self._lock:
            return {
                "bytes": self._bytes,
                "max_bytes": self.max_bytes,
                "entries": len(self._entries),
                "ready_entries": len(self._lru),
                "pages": len(self._pages),
                "epoch": self._epoch,
            }
