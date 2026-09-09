"""Exact hybrid checkpoints at the inference/cache ownership boundary.

Capture runs on the producing stream before the next batch can overwrite a
request's recurrent state. Publication runs in ordered CPU post processing.
The first implementation fences CPU page transfers; GPU hot hits keep their
radix references and only restore the independent recurrent state.
"""

import hashlib
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
        logger.info(
            "exact prefix cache: CPU budget=%s MiB/rank, entries=%s, KV page=%s tokens, "
            "capture slots=%s x %s; recurrent states are independent of KV pages",
            self.args.exact_prefix_cache_mb,
            self.args.exact_prefix_cache_entries,
            self.args.exact_prefix_cache_page_size,
            self.args.exact_prefix_cache_capture_slots,
            len(self._staging),
        )

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
        mask = torch.tensor(candidate, dtype=torch.bool, device="cuda")
        if not ticket.is_prefill:
            # A one-token remainder of chunked prefill is scheduled as decode.
            # It still produces the exact prompt endpoint and the first sample.
            prompt_end = torch.tensor([out == 1 for out in ticket.output_lengths], dtype=torch.bool, device="cuda")
            at_limit = torch.tensor(
                [
                    out == req.sampling_param.shm_param.max_new_tokens
                    for req, out in zip(ticket.reqs, ticket.output_lengths)
                ],
                dtype=torch.bool,
                device="cuda",
            )
            eos = torch.zeros_like(mask)
            token_ids = next_token_ids.reshape(-1)
            for token_id in self.backend.eos_id:
                eos |= token_ids == token_id
            eos_allowed = torch.tensor(
                [not req.sampling_param.shm_param.ignore_eos for req in ticket.reqs],
                dtype=torch.bool,
                device="cuda",
            )
            # A last-token match is an inexpensive candidate hint. CPU post
            # checks the entire stop sequence and may discard a false positive.
            stop = torch.zeros_like(mask)
            for row, req in enumerate(ticket.reqs):
                for sequence in req.stop_sequences:
                    if sequence:
                        stop[row] |= token_ids[row] == sequence[-1]
            mask &= prompt_end | at_limit | (eos & eos_allowed) | stop
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
        return ticket

    def finalize(self, ticket):
        if ticket is None or ticket.staging is None:
            return
        staging = ticket.staging
        slot_id = (ticket.epoch - 1) % len(self._staging)
        try:
            if ticket.cache_generation != self._cache_generation:
                return
            # Called after the batch's normal post handler and compute event.
            metadata = torch.stack((staging.req_indices, staging.exact_lengths, staging.source_rows)).cpu().tolist()
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
            if lease.output_seed is None:
                lengths.remove(len(tokens))
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
                    device="cuda", dtype=torch.int32
                )
            if needed:
                self.cache.load_kv(lease, self.mem_manager.kv_buffer, indexes, start=gpu_length)
                self.req_manager.req_to_token_indexs[req.req_idx, gpu_length:length] = indexes.to(device="cuda")
            req.shared_kv_node = node
            event = torch.cuda.Event()
            event.record()
            snapshot = LinearStateSnapshot(lease.conv_state, lease.ssm_state, length, event)
            self.req_manager.restore_linear_state(snapshot, req.req_idx).synchronize()
            req.cur_kv_len = length
            req.exact_kv_origins = origins
            req.shm_req.shm_cur_kv_len = length
            req.shm_req.prompt_cache_len = length
            seed = lease.output_seed
            if length == len(tokens):
                req.exact_output_seed = seed.unsqueeze(0).to(device="cuda")
            elif self.args.mtp_step:
                self.resume_auxiliary(req, seed.unsqueeze(0).to(device="cuda"), int(tokens[length]))
            self.stats["hits"] += 1
            self.stats["hit_tokens"] += length
            self.stats["loaded_tokens"] += needed
            logger.info("exact checkpoint hit req=%s length=%s GPU=%s CPU=%s", req.req_id, length, gpu_length, needed)
        finally:
            lease.close()

    def resume_auxiliary(self, req, output_seed, next_token):
        if not self.args.mtp_step:
            return
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
        length = req.cur_kv_len
        device = output_seed.device
        model_input = ModelInput(
            batch_size=1,
            total_token_num=length,
            max_q_seq_len=1,
            max_kv_seq_len=length,
            input_ids=torch.tensor([next_token], dtype=torch.int64, device=device),
            mem_indexes=self.req_manager.req_to_token_indexs[req.req_idx, length - 1 : length],
            b_req_idx=torch.tensor([req.req_idx], dtype=torch.int32, device=device),
            b_seq_len=torch.tensor([length], dtype=torch.int32, device=device),
            b_mtp_index=torch.zeros(1, dtype=torch.int32, device=device),
            b_position_delta=torch.tensor(
                [req.multimodal_params.get("mrope_position_delta", 0)], dtype=torch.int32, device=device
            ),
            b_shared_seq_len=torch.zeros(1, dtype=torch.int32, device=device),
            b_shared_radix_node_id=torch.full((1,), -1, dtype=torch.int64, device=device),
            is_prefill=False,
            multimodal_params=[req.multimodal_params],
        )
        self._spec_engine().resume_auxiliary(
            model_input, output_seed, torch.tensor([next_token], dtype=torch.int64, device=device)
        )
        torch.cuda.current_stream().synchronize()
        # The target tail is unchanged but its packed draft half was rebuilt
        # for this request's successor token. It now has private provenance.
        self._aux_epoch += 1
        req.exact_kv_origins[length - 1] = self._origin(req, self._aux_epoch, "draft-tail")

    def process_head_only(self, reqs):
        """Consume full-hit seeds on the scheduler stream before classification."""
        from lightllm.server.router.model_infer.infer_batch import InferReqUpdatePack
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        for req in reqs:
            seed = getattr(req, "exact_output_seed", None)
            if seed is None or req.cur_output_len or req.infer_aborted or req.finish_status.is_finished():
                continue
            if self.backend.is_pd_decode_mode and (
                req.pd_task_failed_num or req.pd_task_num != req.pd_task_success_num
            ):
                continue
            torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
            model_output = self.backend.model.forward_output_seed(seed)
            req_indexes = torch.tensor([req.req_idx], dtype=torch.int32, device="cuda")
            mtp_indexes = torch.zeros(1, dtype=torch.int32, device="cuda")
            next_ids, ids_cpu, logprobs_cpu, ranks_cpu = self.backend._sample_and_scatter_token(
                logits=model_output.logits,
                b_req_idx=req_indexes,
                b_mtp_index=mtp_indexes,
                run_reqs=[req],
                is_prefill=True,
                b_prefill_has_output_cpu=[True],
                mask_func=self.backend.prefill_mask_func,
                pin_memory_namespace="exact_head_",
            )
            torch.cuda.current_stream().synchronize()
            self.resume_auxiliary(req, seed, int(ids_cpu[0]))
            req.cur_output_len = 1
            req.exact_output_seed = None
            self.backend._post_handle(
                run_reqs=[req],
                next_token_ids=ids_cpu,
                next_token_logprobs=logprobs_cpu,
                next_token_ranks=ranks_cpu,
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
