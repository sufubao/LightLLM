"""Exact hybrid checkpoints at the inference/cache ownership boundary.

Capture runs on the producing stream before the next batch can overwrite a
request's recurrent state. Normal-mode publication uses a bounded worker;
the scheduler retains source requests until every TP rank finishes copying.
PD publication and CPU onload retain their explicit transfer fences.
"""

import hashlib
import queue
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.req_manager.linear_att import LinearStateSnapshot
from lightllm.server.router.dynamic_prompt.checkpoint_cache import CpuCheckpointCache
from lightllm.utils.dist_utils import create_new_group_for_current_dp
from lightllm.utils.log_utils import init_logger
from lightllm.utils.checkpoint_identity import get_checkpoint_identity
from lightllm.utils.envs_utils import get_unique_server_name

logger = init_logger(__name__)


@dataclass
class CaptureBatch:
    reqs: list
    output_lengths: list
    is_prefill: bool
    epoch: int
    cache_generation: int
    kv_origins: dict = field(default_factory=dict)
    staging: object = None
    output_seed: Optional[torch.Tensor] = None
    tail_kv: Optional[torch.Tensor] = None
    metadata: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class FrozenCheckpoint:
    req_id: int
    length: int
    output_len: int
    tokens: object
    origins: tuple
    mem_indexes: torch.Tensor
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    output_seed: Optional[torch.Tensor]
    tail_kv: torch.Tensor


class ExactPrefixCache:
    def __init__(self, backend):
        self.backend = backend
        self.args = backend.args
        self.req_manager = backend.model.req_manager
        self.mem_manager = backend.model.mem_manager
        self.cache = CpuCheckpointCache(
            max_bytes=self.args.exact_prefix_cache_mb * 1024 ** 2,
            max_entries=self.args.exact_prefix_cache_entries,
            page_size=self.args.exact_prefix_cache_page_size,
            draft_tail_dependency=bool(self.args.mtp_step),
            preallocate=True,
        )
        self.group = create_new_group_for_current_dp("gloo")
        self.world_size = dist.get_world_size(self.group)
        self._epoch = 0
        self._aux_epoch = 0
        self._producer = f"{get_unique_server_name()}:{backend.global_dp_rank}"
        self._cache_generation = 0
        self._staging = [
            self.req_manager.allocate_linear_state_staging(self.args.exact_prefix_cache_capture_slots)
            for _ in range(
                4 if (self.args.enable_decode_microbatch_overlap or self.args.enable_prefill_microbatch_overlap) else 2
            )
        ]
        self._busy = [False] * len(self._staging)
        self._warmup_kernels()
        self.transport = None
        self.target_fingerprint, self.draft_fingerprint, self.namespace = get_checkpoint_identity(self.args)
        if self.args.run_mode in ("prefill", "decode"):
            from lightllm.server.router.model_infer.mode_backend.pd.checkpoint_transport import PDCheckpointTransport

            self.transport = PDCheckpointTransport(
                backend,
                self.cache,
                namespace=self.namespace,
                target_namespace=f"{self.target_fingerprint}:target-only",
            )
        self.stats = dict(captured=0, skipped=0, hits=0, hit_tokens=0, head_only=0, loaded_tokens=0)
        self._async_publication = self.args.run_mode == "normal"
        self._pending_publications = {}
        self._publication_holds = Counter()
        self._completed_publications = {}
        self._publication_error = None
        if self._async_publication:
            # Worker collectives must never share ordering with scheduler
            # admission, restore, or the two CPU inference threads.
            self._publication_group = create_new_group_for_current_dp("gloo")
            self._publication_stream = torch.cuda.Stream(device=self.mem_manager.kv_buffer.device)
            self._publication_queue = queue.SimpleQueue()
            self._publication_done = queue.SimpleQueue()
            self._publication_ack = threading.Event()
            self._publication_thread = threading.Thread(
                target=self._publication_loop, name="exact-checkpoint-copy", daemon=True
            )
            self._publication_thread.start()
        logger.info(
            "exact prefix cache: CPU budget=%s MiB/rank, entries=%s, KV page=%s tokens, "
            "capture slots=%s x %s; recurrent states are independent of KV pages",
            self.args.exact_prefix_cache_mb,
            self.args.exact_prefix_cache_entries,
            self.args.exact_prefix_cache_page_size,
            self.args.exact_prefix_cache_capture_slots,
            len(self._staging),
        )

    def _warmup_kernels(self):
        """Compile bounded row-block variants before accepting requests."""
        max_rows = self.req_manager.max_request_num * (self.args.mtp_step + 1)
        row_capacity = 1 << (max_rows - 1).bit_length()
        device = self.mem_manager.kv_buffer.device
        zeros = torch.zeros(row_capacity, dtype=torch.int32, device=device)
        lengths = torch.ones_like(zeros)
        mask = torch.zeros(row_capacity, dtype=torch.bool, device=device)
        if self.args.mtp_step:
            from lightllm.common.basemodel.triton_kernel.mtp_utils import gen_b_req_mtp_start_loc

        count = 1
        while count <= row_capacity:
            # An empty mask only initializes staging metadata. It cannot read
            # or overwrite a live recurrent state or request's KV mapping.
            self.req_manager.freeze_linear_states(
                zeros[:count], zeros[:count], lengths[:count], mask[:count], self._staging[0]
            )
            if self.args.mtp_step:
                gen_b_req_mtp_start_loc(zeros[:count], num_reqs=count)
            count *= 2
        torch.cuda.current_stream(device).synchronize()

    def _all(self, value):
        if self.world_size == 1:
            return bool(value)
        flag = torch.tensor(int(bool(value)), dtype=torch.int32, device="cpu")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.group)
        return bool(flag.item())

    def _intersection(self, values):
        if self.world_size == 1:
            return sorted(values, reverse=True)
        lists = [None] * self.world_size
        dist.all_gather_object(lists, list(values), group=self.group)
        return sorted(set.intersection(*(set(v) for v in lists)), reverse=True)

    @staticmethod
    def _is_allocation_failure(error):
        if isinstance(error, (MemoryError, torch.OutOfMemoryError)):
            return True
        if not isinstance(error, RuntimeError):
            return False
        message = str(error).lower()
        allocator = any(
            name in message
            for name in ("defaultcpuallocator", "cachinghostallocator", "cudahostalloc", "pinned memory")
        )
        exhausted = any(
            text in message
            for text in ("can't allocate memory", "out of memory", "not enough memory", "allocation failed")
        )
        return allocator and exhausted

    def _spec_engine(self):
        engine = self.backend.spec_engine
        return getattr(engine, "common_engine", engine)

    def gpu_radix_key(self, origins, length):
        # CPU directory lookup has already checked the actual token prefix.
        # GPU sharing additionally requires the *same computation history*:
        # equal tokens computed by decode/prefill can have different KV bytes.
        if len(origins) < length or any(origin <= 0 for origin in origins[:length]):
            raise ValueError("GPU checkpoint references require complete KV provenance")
        return torch.tensor(origins[:length], dtype=torch.int64, device="cpu")

    def _origin(self, req, epoch, kind="forward"):
        value = f"{self._producer}:{kind}:{req.req_id}:{epoch}".encode()
        return (int.from_bytes(hashlib.sha256(value).digest()[:8], "little") & ((1 << 63) - 1)) or 1

    def _eligible(self, req):
        if req.sampling_param.disable_prompt_cache or req.infer_aborted:
            return False
        if req.sampling_param.shm_param.prompt_logprobs >= 0 or self.args.enable_return_routed_experts:
            # Endpoint hidden/state cannot reproduce per-token prompt outputs.
            return False
        # Image/audio embeddings and positional deltas require their own
        # execution identity; token IDs alone do not establish an equivalent run.
        if any(req.multimodal_params.get(key) for key in ("images", "audios", "videos")):
            return False
        if req.multimodal_params.get("mrope_position_delta", 0):
            return False
        engine = self._spec_engine()
        return not self.args.mtp_step or (engine is not None and engine.supports_exact_prefix_resume())

    def prepare_batch(self, model_input, run_reqs):
        """Bind immutable logical lengths before ModelInput moves to CUDA."""
        self._epoch += 1
        lengths = model_input.b_seq_len.tolist()
        ends = {}
        for req, length in zip(run_reqs, lengths):
            ends[req.req_idx] = (req, max(length, ends.get(req.req_idx, (None, 0))[1]))
        origins = {}
        for req_index, (req, length) in ends.items():
            prefix = req.exact_kv_origins
            if len(prefix) < req.cur_kv_len:
                raise ValueError("computed KV prefix is missing its provenance")
            prefix[req.cur_kv_len :] = [self._origin(req, self._epoch)] * (length - req.cur_kv_len)
            # Later forwards may only change the unaccepted/new suffix. Every
            # candidate that CPU post can publish lies within this batch's
            # accepted prefix, whose origins remain immutable. Keep its list
            # reference instead of copying a million-token prefix each decode.
            origins[req_index] = prefix
        return CaptureBatch(
            reqs=list(run_reqs),
            output_lengths=[length - req.shm_req.input_len + 1 for req, length in zip(run_reqs, lengths)],
            is_prefill=model_input.is_prefill,
            epoch=self._epoch,
            cache_generation=self._cache_generation,
            kv_origins=origins,
        )

    def capture(self, ticket, model_input, model_output, next_token_ids, accepted_index=None):
        """Freeze only selected rows, on the same stream as sample/verify."""
        if not ticket.reqs:
            return ticket
        slot = (ticket.epoch - 1) % len(self._staging)
        if self._busy[slot]:
            self.stats["skipped"] += 1
            return ticket
        candidate = [self._eligible(req) for req in ticket.reqs]
        if not ticket.is_prefill:
            # Known-length generation with EOS disabled needs no device capture
            # work between its prompt endpoint and its final output. Ineligible
            # requests likewise must not pay for empty staging/readback each step.
            candidate = [
                eligible
                and (
                    out == 1
                    or out == req.sampling_param.shm_param.max_new_tokens
                    or not req.sampling_param.shm_param.ignore_eos
                    or bool(req.stop_sequences)
                )
                for eligible, req, out in zip(candidate, ticket.reqs, ticket.output_lengths)
            ]
        if not any(candidate):
            return ticket
        from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager

        flags = candidate
        eos_allowed = []
        if not ticket.is_prefill:
            # A one-token remainder of chunked prefill is scheduled as decode.
            # It still produces the exact prompt endpoint and the first sample.
            flags = [
                eligible and (out == 1 or out == req.sampling_param.shm_param.max_new_tokens)
                for eligible, req, out in zip(candidate, ticket.reqs, ticket.output_lengths)
            ]
            eos_allowed = [
                eligible and not req.sampling_param.shm_param.ignore_eos
                for eligible, req in zip(candidate, ticket.reqs)
            ]
            if any(eos_allowed):
                flags += eos_allowed
        # Reuse pinned metadata for this staging slot. Its publication fence
        # completes the H2D read before the slot can be reused by either infer
        # thread. One asynchronous copy preserves the producing stream's order
        # without waiting for the preceding forward/propose work on the CPU.
        flags_gpu = g_pin_mem_manager.gen_from_list(
            key=f"exact_capture_flags_{slot}", data=flags, dtype=torch.bool
        ).cuda(non_blocking=True)
        mask = flags_gpu[: len(candidate)]
        if not ticket.is_prefill:
            token_ids = next_token_ids.reshape(-1)
            if any(eos_allowed):
                eos_mask = flags_gpu[len(candidate) :]
                for token_id in self.backend.eos_id:
                    mask |= (token_ids == token_id) & eos_mask
            # A last-token match is an inexpensive candidate hint. CPU post
            # checks the entire stop sequence and may discard a false positive.
            for row, (eligible, req) in enumerate(zip(candidate, ticket.reqs)):
                if eligible:
                    for sequence in req.stop_sequences:
                        if sequence:
                            mask[row] |= token_ids[row] == sequence[-1]
        if accepted_index is not None:
            mask &= accepted_index.to(dtype=torch.bool)
        staging = self._staging[slot]
        self._busy[slot] = True
        ticket.staging = staging
        self.req_manager.freeze_linear_states(
            model_input.b_req_idx,
            model_input.b_mtp_index,
            model_input.b_seq_len,
            mask,
            staging,
        )
        ticket.output_seed = model_output.output_seed
        # Speculative draft fill may rewrite its tail KV before CPU publication.
        # Freeze that packed token along with the recurrent state.
        source_indexes = self.req_manager.req_to_token_indexs[
            staging.req_indices.clamp_min(0).long(), (staging.exact_lengths - 1).clamp_min(0).long()
        ]
        source_indexes = torch.where(staging.req_indices >= 0, source_indexes, self.mem_manager.HOLD_TOKEN_MEMINDEX)
        ticket.tail_kv = self.mem_manager.kv_buffer.index_select(1, source_indexes.long())
        # The caller records its compute event after capture. Include metadata
        # readback in that event so CPU post only reads completed pinned data.
        ticket.metadata = g_pin_mem_manager.async_copy_from_gpu_tensor(
            key=f"exact_capture_metadata_{slot}",
            gpu_tensor=torch.stack((staging.req_indices, staging.exact_lengths, staging.source_rows)),
        )
        return ticket

    def finalize(self, ticket):
        if self._async_publication:
            self._enqueue_publication(ticket)
            return
        self._finalize_sync(ticket)

    def _enqueue_publication(self, ticket):
        if ticket is None:
            return
        # Capture admission may differ locally; queue epochs and resource
        # holds must nevertheless remain identical within each TP group.
        if not self._all(ticket.staging is not None):
            if ticket.staging is not None:
                self._busy[(ticket.epoch - 1) % len(self._staging)] = False
            return
        staging = ticket.staging
        metadata = ticket.metadata.tolist()
        descriptors = []
        for slot, (req_index, length, row) in enumerate(zip(*metadata)):
            descriptor = None
            if req_index >= 0:
                req = ticket.reqs[row]
                output_len = ticket.output_lengths[row]
                visible = getattr(req, "exact_visible_end", -1)
                if (
                    req.req_idx == req_index
                    and self._eligible(req)
                    and (
                        ticket.is_prefill or output_len == 1 or (req.finish_status.is_finished() and length <= visible)
                    )
                ):
                    # CPU post has resolved accepted tokens and stop sequences.
                    # Freeze all metadata here: a later batch can mutate req.
                    descriptor = FrozenCheckpoint(
                        req_id=req.req_id,
                        length=length,
                        output_len=output_len,
                        tokens=req.shm_req.shm_prompt_ids.arr[:length].copy(),
                        origins=tuple(ticket.kv_origins[req_index][:length]),
                        mem_indexes=self.req_manager.req_to_token_indexs[req_index, :length],
                        conv_state=staging.conv_state[slot],
                        ssm_state=staging.ssm_state[slot],
                        output_seed=None if ticket.output_seed is None else ticket.output_seed[row],
                        tail_kv=ticket.tail_kv[:, slot],
                    )
            descriptors.append(descriptor)
        # Hold every request in the ticket, including unselected slots. The
        # fixed ticket membership makes scheduler holds independent of local
        # eligibility and worker timing. Staging slots bound queue occupancy.
        self._pending_publications[ticket.epoch] = ticket
        self._publication_holds.update({req.req_id for req in ticket.reqs})
        self._publication_queue.put((ticket.epoch, ticket.cache_generation, descriptors))

    def _publication_status(self, value):
        if self.world_size == 1:
            return value
        status = torch.tensor(value, dtype=torch.int32, device="cpu")
        dist.all_reduce(status, op=dist.ReduceOp.MIN, group=self._publication_group)
        return int(status.item())

    def _publish_frozen(self, generation, descriptors):
        published = []
        skipped = 0
        # Fixed slot count ensures every TP rank executes the same collectives
        # even if a local abort or allocation failure rejects its descriptor.
        pending = None
        try:
            for descriptor in descriptors:
                if not self._publication_status(int(descriptor is not None and generation == self._cache_generation)):
                    continue
                pending = None
                error = None
                existing = None
                reusable = False
                try:
                    try:
                        existing = self.cache.acquire(descriptor.tokens, descriptor.length, namespace=self.namespace)
                        reusable = existing is not None and (
                            descriptor.output_seed is None or existing.output_seed is not None
                        )
                    except BaseException as exc:
                        if self._is_allocation_failure(exc):
                            logger.warning("checkpoint lookup skipped req=%s: %s", descriptor.req_id, exc)
                        else:
                            error = exc
                    status = self._publication_status(-1 if error is not None else int(reusable))
                    if status < 0:
                        raise RuntimeError("checkpoint publication failed on a TP rank") from error
                    if status == 1:
                        # Retain the existing KV/state/seed history as one unit.
                        # A lease keeps it alive across the TP presence check;
                        # clear() may still invalidate its directory generation.
                        if self._publication_status(int(generation == self._cache_generation)):
                            skipped += 1
                        continue
                finally:
                    if existing is not None:
                        existing.close()
                try:
                    pending = self.cache.prepare(
                        descriptor.tokens,
                        self.mem_manager.kv_buffer,
                        descriptor.mem_indexes,
                        descriptor.conv_state,
                        descriptor.ssm_state,
                        output_seed=descriptor.output_seed,
                        namespace=self.namespace,
                        frozen_tail_kv=descriptor.tail_kv,
                        origins=descriptor.origins,
                    )
                except BaseException as exc:
                    if self._is_allocation_failure(exc):
                        logger.warning("checkpoint allocation skipped req=%s: %s", descriptor.req_id, exc)
                    else:
                        error = exc
                status = self._publication_status(
                    -1 if error is not None else int(pending is not None and generation == self._cache_generation)
                )
                if status == 1:
                    # Only the scheduler may change the visible directory. A
                    # worker commit could let concurrent TP restores acquire
                    # different histories for the same tokens and length.
                    published.append((descriptor.req_id, descriptor.length, descriptor.output_len, pending))
                else:
                    if pending is not None:
                        self.cache.discard(pending)
                    if status < 0:
                        raise RuntimeError("checkpoint publication failed on a TP rank") from error
                    skipped += 1
        except BaseException:
            # A fatal rank error must not strand earlier prepared CPU pages.
            for _, _, _, prepared in published:
                if not prepared.consumed:
                    self.cache.discard(prepared)
            if pending is not None and not pending.consumed:
                self.cache.discard(pending)
            raise
        return published, skipped

    def _publication_loop(self):
        try:
            with torch.cuda.device(self.mem_manager.kv_buffer.device), torch.cuda.stream(self._publication_stream):
                while True:
                    epoch, generation, descriptors = self._publication_queue.get()
                    published, skipped = self._publish_frozen(generation, descriptors)
                    # prepare fences every GPU reader before reporting done.
                    # Do not retain the previous job while blocking on get().
                    descriptors = None
                    self._publication_done.put((epoch, published, skipped))
                    # Let the scheduler commit/discard before the next prepare
                    # can take the cache lock across another device transfer.
                    self._publication_ack.wait()
                    self._publication_ack.clear()
                    published = None
        except BaseException as error:
            self._publication_error = error
            logger.exception("exact checkpoint publication worker failed")

    def has_pending(self, req):
        return bool(self._publication_holds.get(req.req_id, 0))

    def poll_publications(self):
        """Retire globally completed jobs without waiting for unfinished copies."""
        if not self._pending_publications:
            return
        if not self._all(self._publication_error is None):
            raise RuntimeError("exact checkpoint publication worker failed") from self._publication_error
        while True:
            try:
                epoch, published, skipped = self._publication_done.get_nowait()
                self._completed_publications[epoch] = (published, skipped)
            except queue.Empty:
                break
        for epoch in reversed(self._intersection(self._completed_publications)):
            ticket = self._pending_publications.pop(epoch)
            published, skipped = self._completed_publications.pop(epoch)
            reqs = {req.req_id: req for req in ticket.reqs}
            current_generation = ticket.cache_generation == self._cache_generation
            for req_id, length, output_len, pending in published:
                if not current_generation:
                    self.cache.discard(pending)
                elif self.cache.commit(pending):
                    self.stats["captured"] += 1
                    reqs[req_id].exact_checkpoint_length = length
                    logger.debug("checkpoint published req=%s length=%s output_len=%s", req_id, length, output_len)
            for req_id in reqs:
                self._publication_holds[req_id] -= 1
                if not self._publication_holds[req_id]:
                    del self._publication_holds[req_id]
            if current_generation:
                self.stats["skipped"] += skipped
            self._busy[(epoch - 1) % len(self._staging)] = False
            self._publication_ack.set()

    def _finalize_sync(self, ticket):
        if ticket is None or ticket.staging is None:
            return
        staging = ticket.staging
        slot_id = (ticket.epoch - 1) % len(self._staging)
        try:
            if ticket.cache_generation != self._cache_generation:
                return
            # Called after the batch's normal post handler and compute event.
            metadata = ticket.metadata.tolist()
            for slot, (req_index, length, row) in enumerate(zip(*metadata)):
                if req_index < 0:
                    continue
                req = ticket.reqs[row]
                if req.req_idx != req_index or not self._eligible(req):
                    continue
                # A later stop decision cannot relabel this frozen state as an
                # earlier version. MTP rows after a stop are never admitted.
                output_len = ticket.output_lengths[row]
                visible_limit = getattr(req, "exact_visible_end", -1)
                is_prompt_end = output_len == 1
                if (
                    not ticket.is_prefill
                    and not is_prompt_end
                    and (not req.finish_status.is_finished() or length > visible_limit)
                ):
                    continue
                tokens = req.shm_req.shm_prompt_ids.arr[:length].copy()
                seed = None if ticket.output_seed is None else ticket.output_seed[row]
                pending = None
                try:
                    pending = self.cache.prepare(
                        tokens,
                        self.mem_manager.kv_buffer,
                        self.req_manager.req_to_token_indexs[req_index, :length],
                        staging.conv_state[slot],
                        staging.ssm_state[slot],
                        output_seed=seed,
                        namespace=self.namespace,
                        frozen_tail_kv=ticket.tail_kv[:, slot],
                        origins=ticket.kv_origins[req_index][:length],
                    )
                except (MemoryError, RuntimeError) as error:
                    if not self._is_allocation_failure(error):
                        raise
                    # A local allocation failure is an admission rejection on
                    # every TP rank, not an early exit before the collective.
                    logger.warning("checkpoint allocation skipped req=%s: %s", req.req_id, error)
                if self._all(pending is not None):
                    self.cache.commit(pending)
                    self.stats["captured"] += 1
                    req.exact_checkpoint_length = length
                    logger.debug("checkpoint published req=%s length=%s output_len=%s", req.req_id, length, output_len)
                    if self.transport is not None and self.backend.is_pd_decode_mode:
                        owner = bytes(req.sampling_param.shm_param.pd_checkpoint_owner_url).decode()
                        if owner:
                            self.transport.publish_checkpoint(
                                tokens,
                                self.namespace,
                                owner,
                                f"{req.req_id}:{ticket.epoch}:{length}",
                                owner_dp_index=getattr(req, "pd_checkpoint_owner_dp_index", 0),
                                owner_auth=bytes(req.sampling_param.shm_param.pd_checkpoint_owner_auth).decode(),
                            )
                else:
                    if pending is not None:
                        self.cache.discard(pending)
                    self.stats["skipped"] += 1
        finally:
            # prepare fences all readers of staging before it is reused.
            self._busy[slot_id] = False

    def restore(self, req):
        if req.cur_kv_len:
            return
        if not self._eligible(req):
            self.req_manager.init_linear_att_state(req)
            return
        tokens = req.get_input_token_ids().copy()
        full_hit = req.cur_output_len == 0 and self.backend.model.supports_exact_output_seed()
        lengths = self.cache.candidate_lengths(
            tokens,
            max_length=len(tokens) if full_hit else len(tokens) - 1,
            require_output_seed=bool(self.args.mtp_step),
            namespace=self.namespace,
        )
        if self.backend.is_pd_decode_mode:
            # Partial D KV and the state later produced by P may come from
            # different numerical histories. Until P/D negotiate a common
            # origin frontier, only D-local completion avoids mixing them.
            lengths = [length for length in lengths if length >= len(tokens) - 1]
        if full_hit and len(tokens) in lengths:
            lease = self.cache.acquire(tokens, len(tokens), namespace=self.namespace)
            # A background preparation may evict the candidate between lookup
            # and lease acquisition. Missing local candidates are intersected
            # away before any TP rank begins a restore.
            if lease is None or lease.output_seed is None:
                lengths.remove(len(tokens))
            if lease is not None:
                lease.close()
        lengths = self._intersection(lengths)
        if not lengths:
            self.req_manager.init_linear_att_state(req)
            return
        length = lengths[0]
        lease = self.cache.acquire(tokens, length, namespace=self.namespace)
        if not self._all(lease is not None):
            if lease is not None:
                lease.close()
            self.req_manager.init_linear_att_state(req)
            return
        try:
            origins = lease.origins.tolist()
            # A normal complete hit needs no draft repair before HEAD_ONLY.
            # Queue its KV, canonical state and seed on one stream, then fence
            # once while the CPU lease still protects every source window.
            join_restore = self.args.run_mode == "normal" and length == len(tokens)
            # The last packed KV slot must be private before the draft adapter
            # rebuilds its sampling-dependent tail.
            shared_limit = length - 1 if self.args.mtp_step else length
            node, gpu_length, values = (None, 0, None)
            if shared_limit:
                node, gpu_length, values = self.backend.radix_cache.match_prefix(
                    self.gpu_radix_key(origins, shared_limit),
                    update_refs=True,
                )
            needed = length - gpu_length
            available = self.req_manager.mem_manager.allocator.can_use_mem_size
            available += (
                self.backend.radix_cache.get_tree_total_tokens_num() - self.backend.radix_cache.get_refed_tokens_num()
            )
            if not self._all(needed <= available):
                if node is not None:
                    self.backend.radix_cache.dec_node_ref_counter(node)
                self.req_manager.init_linear_att_state(req)
                return
            self.backend.radix_cache.free_radix_cache_to_get_enough_token(needed)
            indexes = self.mem_manager.alloc(needed)
            if gpu_length:
                self.req_manager.req_to_token_indexs[req.req_idx, :gpu_length] = values.to(
                    device="cuda", dtype=torch.int32, non_blocking=join_restore
                )
            if needed:
                self.cache.load_kv(lease, self.mem_manager.kv_buffer, indexes, start=gpu_length, wait=not join_restore)
                self.req_manager.req_to_token_indexs[req.req_idx, gpu_length:length] = indexes.to(
                    device="cuda", non_blocking=join_restore
                )
            req.shared_kv_node = node
            event = torch.cuda.Event()
            event.record()
            snapshot = LinearStateSnapshot(lease.conv_state, lease.ssm_state, length, event)
            restored = self.req_manager.restore_linear_state(snapshot, req.req_idx)
            seed = lease.output_seed
            if join_restore:
                output_seed = seed.unsqueeze(0).to(device="cuda", non_blocking=True)
                torch.cuda.current_stream().synchronize()
            else:
                restored.synchronize()
            req.cur_kv_len = length
            req.exact_kv_origins = origins
            req.shm_req.shm_cur_kv_len = length
            req.shm_req.prompt_cache_len = length
            if length == len(tokens):
                req.exact_output_seed = output_seed if join_restore else seed.unsqueeze(0).to(device="cuda")
            elif self.args.mtp_step:
                self.resume_auxiliary(req, seed.unsqueeze(0).to(device="cuda"), int(tokens[length]))
            self.stats["hits"] += 1
            self.stats["hit_tokens"] += length
            self.stats["loaded_tokens"] += needed
            logger.info("exact checkpoint hit req=%s length=%s GPU=%s CPU=%s", req.req_id, length, gpu_length, needed)
        except BaseException:
            # A failed state/seed allocation can leave an earlier H2D copy in
            # flight. Arena windows are reclaimed when this lease is closed.
            torch.cuda.current_stream().synchronize()
            raise
        finally:
            lease.close()

    def resume_auxiliary(self, req, output_seed, next_token):
        if not self.args.mtp_step:
            return
        from lightllm.server.router.model_infer.infer_batch import g_infer_context
        from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager

        torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
        next_ids = g_pin_mem_manager.gen_from_list(key="exact_aux_next_ids", data=[next_token], dtype=torch.int64).cuda(
            non_blocking=True
        )
        self._resume_auxiliary_batch([req], output_seed, next_ids)
        torch.cuda.current_stream().synchronize()
        self._aux_epoch += 1
        req.exact_kv_origins[req.cur_kv_len - 1] = self._origin(req, self._aux_epoch, "draft-tail")

    def _resume_auxiliary_batch(self, reqs, output_seed, next_token_ids):
        """Enqueue private draft-tail repairs; the caller owns the completion fence."""
        if not self.args.mtp_step or not reqs:
            return []
        from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager

        lengths = [req.cur_kv_len for req in reqs]
        req_indexes = g_pin_mem_manager.gen_from_list(
            key="exact_aux_req_indexes", data=[req.req_idx for req in reqs], dtype=torch.int32
        ).cuda(non_blocking=True)
        seq_lengths = g_pin_mem_manager.gen_from_list(
            key="exact_aux_seq_lengths", data=lengths, dtype=torch.int32
        ).cuda(non_blocking=True)
        position_deltas = g_pin_mem_manager.gen_from_list(
            key="exact_aux_position_deltas",
            data=[req.multimodal_params.get("mrope_position_delta", 0) for req in reqs],
            dtype=torch.int32,
        ).cuda(non_blocking=True)
        # Each restored request already owns its last packed target/draft slot.
        # Gathering these slots keeps repairs independent for different lengths.
        mem_indexes = self.req_manager.req_to_token_indexs[req_indexes.long(), seq_lengths.long() - 1]
        batch_size = len(reqs)
        device = output_seed.device
        # FlashInfer filtered sampling may return int32 IDs. ModelInput and the
        # embedding path require int64; keep this conversion entirely on GPU.
        next_token_ids = next_token_ids.to(dtype=torch.int64)
        model_input = ModelInput(
            batch_size=batch_size,
            total_token_num=sum(lengths),
            max_q_seq_len=1,
            max_kv_seq_len=max(lengths),
            input_ids=next_token_ids,
            mem_indexes=mem_indexes,
            b_req_idx=req_indexes,
            b_seq_len=seq_lengths,
            b_mtp_index=torch.zeros(batch_size, dtype=torch.int32, device=device),
            b_position_delta=position_deltas,
            b_shared_seq_len=torch.zeros(batch_size, dtype=torch.int32, device=device),
            b_shared_radix_node_id=torch.full((batch_size,), -1, dtype=torch.int64, device=device),
            is_prefill=False,
            multimodal_params=[req.multimodal_params for req in reqs],
        )
        self._spec_engine().resume_auxiliary(model_input, output_seed, next_token_ids)

    def process_head_only(self, reqs):
        """Consume full-hit seeds on the scheduler stream before classification."""
        from lightllm.server.router.model_infer.infer_batch import InferReqUpdatePack
        from lightllm.server.router.model_infer.infer_batch import g_infer_context
        from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager

        head_reqs = []
        for req in reqs:
            seed = getattr(req, "exact_output_seed", None)
            if seed is None or req.cur_output_len or req.infer_aborted or req.finish_status.is_finished():
                continue
            if self.backend.is_pd_decode_mode and (
                req.pd_task_failed_num or req.pd_task_num != req.pd_task_success_num
            ):
                continue
            head_reqs.append(req)
        if not head_reqs:
            return

        torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
        seeds = torch.cat([req.exact_output_seed for req in head_reqs], dim=0)
        model_output = self.backend.model.forward_output_seed(seeds)
        req_indexes = g_pin_mem_manager.gen_from_list(
            key="exact_head_req_indexes", data=[req.req_idx for req in head_reqs], dtype=torch.int32
        ).cuda(non_blocking=True)
        mtp_indexes = torch.zeros(len(head_reqs), dtype=torch.int32, device=seeds.device)
        next_ids, ids_cpu, logprobs_cpu, ranks_cpu = self.backend._sample_and_scatter_token(
            logits=model_output.logits,
            b_req_idx=req_indexes,
            b_mtp_index=mtp_indexes,
            run_reqs=head_reqs,
            is_prefill=True,
            b_prefill_has_output_cpu=[True] * len(head_reqs),
            mask_func=self.backend.prefill_mask_func,
            pin_memory_namespace="exact_head_",
        )
        self._resume_auxiliary_batch(head_reqs, seeds, next_ids)
        # One fence covers sampled CPU outputs and every draft repair. It also
        # keeps all seeds and private slots alive before the next decode starts.
        torch.cuda.current_stream().synchronize()
        for row, req in enumerate(head_reqs):
            if self.args.mtp_step:
                # Target KV is unchanged; the repaired draft half has new provenance.
                self._aux_epoch += 1
                req.exact_kv_origins[req.cur_kv_len - 1] = self._origin(req, self._aux_epoch, "draft-tail")
            req.cur_output_len = 1
            req.exact_output_seed = None
            if self.args.mtp_step and self.args.run_mode == "normal" and not self.args.mtp_dynamic_verify:
                req.exact_mtp_needs_proposal = True
            self.backend._post_handle(
                run_reqs=[req],
                next_token_ids=ids_cpu[row : row + 1],
                next_token_logprobs=logprobs_cpu[row : row + 1],
                next_token_ranks=ranks_cpu[row : row + 1],
                run_reqs_update_packs=[InferReqUpdatePack(req, 1)],
                extra_post_req_handle_func=self.backend.extra_post_req_handle_func,
                pd_prefill_chunked_handle_func=self.backend.pd_prefill_chunked_handle_func,
            )
            self.stats["head_only"] += 1

    def clear(self):
        self._cache_generation += 1
        if self.transport is not None:
            self.transport.clear()
        self.cache.clear()

    def poll_imports(self):
        if self.transport is None:
            return
        imports = {item.import_id: item for item in self.transport.drain_imports()}
        for import_id in self._intersection(imports):
            item = imports[import_id]
            pending = None
            try:
                pending = self.cache.prepare_import(item.payload)
            except (ValueError, MemoryError, RuntimeError) as exc:
                logger.warning("checkpoint import rejected: %s", exc)
            success = self._all(pending is not None)
            if success:
                self.cache.commit(pending)
            elif pending is not None:
                self.cache.discard(pending)
            self.transport.finish_import(import_id, success)
