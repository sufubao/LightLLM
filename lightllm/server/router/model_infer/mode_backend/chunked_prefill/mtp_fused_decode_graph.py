import bisect
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req
from lightllm.common.basemodel.triton_kernel.gather_token_id import gather_token
from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    linear_att_mtp_state_index_update,
    mtp_scatter_next_token_ids,
    mtp_verify,
)
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from .impl import ChunkedPrefillBackend

logger = init_logger(__name__)


@dataclass
class FusedStepOutput:
    next_token_ids: torch.Tensor
    next_token_logprobs: torch.Tensor
    mtp_accept_len: torch.Tensor
    accepted_index: torch.Tensor


@dataclass
class _GraphBundle:
    verify_graph: torch.cuda.CUDAGraph
    draft_graph: torch.cuda.CUDAGraph
    mtp_accept_len: torch.Tensor
    accepted_index: torch.Tensor
    main_hiddens: torch.Tensor


class MTPFusedDecodeGraph:
    """Capture one EAGLE MTP decode step as a verify graph and a draft graph."""

    def __init__(self, backend: "ChunkedPrefillBackend"):
        self.backend = backend
        self.model = backend.model
        self.draft_model = backend.draft_models[0]
        self.mtp_step = backend.mtp_step
        self.mtp_size = self.mtp_step + 1

        model_graph = self.model.graph
        self.graph_max_batch_size = model_graph.max_batch_size
        self.graph_max_len_in_batch = model_graph.graph_max_len_in_batch
        self.cuda_graph_batch_sizes = [
            batch_size for batch_size in model_graph.cuda_graph_batch_sizes if batch_size % self.mtp_size == 0
        ]
        self.mempool = model_graph.mempool
        self.torch_memory_saver = model_graph.torch_memory_saver
        self.vocab_size = self.model.vocab_size

        max_batch_size = self.graph_max_batch_size
        max_req_num = max_batch_size // self.mtp_size
        self.req_manager = self.model.req_manager
        self.sampling_manager = self.req_manager.req_sampling_params_manager

        self.b_req_idx = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
        self.b_seq_len = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
        self.b_mtp_index = torch.arange(self.mtp_size, dtype=torch.int32, device="cuda").repeat(max_req_num)
        self.mem_indexes = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
        self.chain_scratch = backend.mtp_chain_scratch
        self.input_ids = torch.zeros(max_batch_size, dtype=torch.int64, device="cuda")
        self.b_position_delta = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
        self._position_delta_rows = 0

        self.temperature = torch.ones(max_batch_size, dtype=torch.float32, device="cuda")
        self.top_k = torch.ones(max_batch_size, dtype=torch.int32, device="cuda")
        self.top_p = torch.ones(max_batch_size, dtype=torch.float32, device="cuda")
        self.philox_seed = torch.tensor([random.getrandbits(63)], dtype=torch.int64, device="cuda")
        self.philox_offset = torch.tensor([0], dtype=torch.int64, device="cuda")

        pin_int = dict(dtype=torch.int32, device="cpu", pin_memory=True)
        self.b_req_idx_pin = torch.zeros(max_batch_size, **pin_int)
        self.b_seq_len_pin = torch.zeros(max_batch_size, **pin_int)
        self.mem_indexes_pin = torch.zeros(max_batch_size, **pin_int)
        self.b_position_delta_pin = torch.zeros(max_batch_size, **pin_int)
        self.temperature_pin = torch.ones(max_batch_size, dtype=torch.float32, device="cpu", pin_memory=True)
        self.top_k_pin = torch.ones(max_batch_size, **pin_int)
        self.top_p_pin = torch.ones(max_batch_size, dtype=torch.float32, device="cpu", pin_memory=True)

        self.out_next_token_ids = torch.zeros(max_batch_size, dtype=torch.int64, device="cuda")
        self.out_next_token_logprobs = torch.zeros(max_batch_size, dtype=torch.float32, device="cuda")
        self.graphs = {}
        self._replay_batch_size = None

        self.hold_req_idx = self.req_manager.HOLD_REQUEST_ID
        self.hold_mem_index = self.model.mem_manager.HOLD_TOKEN_MEMINDEX

        from flashinfer.sampling import top_k_top_p_sampling_from_probs

        self._sample = top_k_top_p_sampling_from_probs

    def _find_graph_batch_size(self, batch_size: int) -> Optional[int]:
        index = bisect.bisect_left(self.cuda_graph_batch_sizes, batch_size)
        if index < len(self.cuda_graph_batch_sizes):
            return self.cuda_graph_batch_sizes[index]
        return None

    def can_run(self, decode_reqs: List[InferReq], max_kv_seq_len: int, batch_size: int) -> bool:
        if batch_size % self.mtp_size != 0 or self._find_graph_batch_size(batch_size) is None:
            return False
        if max_kv_seq_len + self.mtp_step > self.graph_max_len_in_batch:
            return False
        if self.backend.decode_mask_func is not None:
            return False

        for req in decode_reqs:
            if req.mtp_step != self.mtp_step:
                return False
            shm_param = req.sampling_param.shm_param
            if req.generator is not None or req.sampling_param.invalid_token_ids:
                return False
            if req.need_out_token_id_statistics:
                return False
            if shm_param.exponential_decay_length_penalty.to_tuple()[1] != 1.0:
                return False
            if shm_param.min_new_tokens > 1:
                output_len = req.get_cur_total_len() - req.shm_req.input_len
                if output_len < shm_param.min_new_tokens - 1:
                    return False
        return True

    @torch.no_grad()
    def warmup(self):
        logger.info("Begin capture mtp fused decode cudagraph.")
        for batch_size in self.cuda_graph_batch_sizes[::-1]:
            self._stage_warmup_inputs(batch_size)
            torch.cuda.synchronize()
            verify_ctx = self._run_verify_body(batch_size)
            self._run_draft_body(batch_size, verify_ctx)
            torch.cuda.synchronize()

            verify_graph = torch.cuda.CUDAGraph()
            with self.torch_memory_saver.cuda_graph(verify_graph, pool=self.mempool):
                verify_ctx = self._run_verify_body(batch_size)
            mtp_accept_len, accepted_index, main_hiddens = verify_ctx

            draft_graph = torch.cuda.CUDAGraph()
            with self.torch_memory_saver.cuda_graph(draft_graph, pool=self.mempool):
                self._run_draft_body(batch_size, verify_ctx)

            self.graphs[batch_size] = _GraphBundle(
                verify_graph=verify_graph,
                draft_graph=draft_graph,
                mtp_accept_len=mtp_accept_len,
                accepted_index=accepted_index,
                main_hiddens=main_hiddens,
            )
            verify_graph.replay()
            draft_graph.replay()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info(f"Capture mtp fused decode cudagraph success, sizes: {self.cuda_graph_batch_sizes}")

    def _stage_warmup_inputs(self, batch_size: int):
        self._reset_padding_linear_state()
        req_num = batch_size // self.mtp_size
        self.b_req_idx[:batch_size].fill_(self.hold_req_idx)
        seq_pattern = torch.arange(2, self.mtp_size + 2, dtype=torch.int32, device="cuda")
        self.b_seq_len[:batch_size].copy_(seq_pattern.repeat(req_num))
        self.mem_indexes[:batch_size].fill_(self.hold_mem_index)
        self.b_position_delta[:batch_size].zero_()
        self.temperature[:batch_size].fill_(1.0)
        self.top_k[:batch_size].fill_(1)
        self.top_p[:batch_size].fill_(1.0)

    def _reset_padding_linear_state(self):
        if self.backend.is_linear_att_mixed_model:
            self.req_manager.req_to_mtp_state_index[self.hold_req_idx].zero_()

    def _build_model_input(self, batch_size: int) -> ModelInput:
        return ModelInput(
            batch_size=batch_size,
            total_token_num=self.graph_max_len_in_batch * batch_size,
            max_q_seq_len=1,
            max_kv_seq_len=self.graph_max_len_in_batch,
            input_ids=self.input_ids[:batch_size],
            mem_indexes=self.mem_indexes[:batch_size],
            b_req_idx=self.b_req_idx[:batch_size],
            b_seq_len=self.b_seq_len[:batch_size],
            b_mtp_index=self.b_mtp_index[:batch_size],
            b_position_delta=self.b_position_delta[:batch_size],
            is_prefill=False,
            multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
        )

    def _forward_in_body(self, model, model_input: ModelInput):
        infer_state = model._create_inferstate(model_input)
        copy_kv_index_to_req(
            model.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_seq_len,
            infer_state.mem_index,
        )
        infer_state.init_some_extra_state(model)
        infer_state.init_att_state()
        return model._token_forward(infer_state)

    def _run_verify_body(self, batch_size: int):
        req_num = batch_size // self.mtp_size
        b_req_idx = self.b_req_idx[:batch_size]
        req_to_next_token_ids = self.sampling_manager.req_to_next_token_ids
        gather_token(
            req_to_next_token_ids,
            b_req_idx,
            self.b_mtp_index[:batch_size],
            out=self.input_ids[:batch_size],
        )

        model_output = self._forward_in_body(self.model, self._build_model_input(batch_size))
        logits = model_output.logits
        logits.div_(self.temperature[:batch_size].view(-1, 1))
        probs = torch.softmax(logits, dim=-1)
        sampled_ids = self._sample(
            probs,
            self.top_k[:batch_size],
            self.top_p[:batch_size],
            filter_apply_order="joint",
            deterministic=True,
            seed=self.philox_seed,
            offset=self.philox_offset,
            check_nan=False,
        )
        self.philox_offset += 4 * ((batch_size * self.vocab_size + 3) // 4)

        next_token_ids = self.out_next_token_ids[:batch_size]
        next_token_ids.copy_(sampled_ids)
        next_token_probs = torch.gather(probs, dim=1, index=next_token_ids.view(-1, 1))
        next_token_logprobs = self.out_next_token_logprobs[:batch_size]
        next_token_logprobs.copy_(torch.log(next_token_probs).view(-1))

        b_req_mtp_start_loc = torch.arange(req_num, dtype=torch.int32, device="cuda") * self.mtp_size
        mtp_accept_len, accepted_index = mtp_verify(
            req_to_next_token_ids=req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            new_next_token_ids=next_token_ids,
            b_req_idx=b_req_idx,
        )
        if self.backend.is_linear_att_mixed_model:
            linear_att_mtp_state_index_update(
                req_to_mtp_state_index=self.req_manager.req_to_mtp_state_index,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                b_req_idx=b_req_idx,
                b_mtp_index=self.b_mtp_index[:batch_size],
                accepted_index=accepted_index,
                max_mtp_step=self.mtp_size,
            )
        self.sampling_manager.update_reqs_out_token_counter_gpu(
            b_req_idx=b_req_idx,
            next_token_ids=next_token_ids,
            mask=accepted_index == 1,
        )
        return mtp_accept_len, accepted_index, model_output.mtp_main_output_hiddens

    def _run_draft_body(self, batch_size: int, verify_ctx):
        mtp_accept_len, _, draft_hiddens = verify_ctx
        req_num = batch_size // self.mtp_size
        b_req_idx = self.b_req_idx[:batch_size]
        b_seq_len = self.b_seq_len[:batch_size]
        draft_model_input = self._build_model_input(batch_size)
        draft_next_token_ids = self.out_next_token_ids[:batch_size]
        all_next_token_ids = [draft_next_token_ids]

        for step in range(self.mtp_step):
            draft_model_input.input_ids = draft_next_token_ids
            if step == 0:
                draft_model_input.mem_indexes = self.mem_indexes[:batch_size]
            else:
                draft_model_input.mem_indexes = self.chain_scratch[(step - 1) * batch_size : step * batch_size]
            draft_model_input.mtp_draft_input_hiddens = draft_hiddens
            draft_model_output = self._forward_in_body(self.draft_model, draft_model_input)
            draft_next_token_ids = torch.argmax(draft_model_output.logits, dim=-1)
            draft_hiddens = draft_model_output.mtp_main_output_hiddens
            all_next_token_ids.append(draft_next_token_ids)
            b_seq_len += 1

        if self.mtp_step > 1:
            b_seq_len -= self.mtp_step
            copy_kv_index_to_req(
                self.req_manager.req_to_token_indexs,
                b_req_idx,
                b_seq_len,
                self.mem_indexes[:batch_size],
            )
            b_seq_len += self.mtp_step

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)
        b_req_mtp_start_loc = torch.arange(req_num, dtype=torch.int32, device="cuda") * self.mtp_size
        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.sampling_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )

    def _flush_position_delta(self, has_delta: bool, batch_size: int):
        previous_rows = self._position_delta_rows
        if has_delta or previous_rows > 0:
            rows = max(batch_size, previous_rows)
            self.b_position_delta_pin[batch_size:rows].zero_()
            self.b_position_delta[:rows].copy_(self.b_position_delta_pin[:rows], non_blocking=True)
        self._position_delta_rows = batch_size if has_delta else 0

    def replay_verify(self, model_input: ModelInput, run_reqs: List[InferReq]) -> FusedStepOutput:
        real_batch_size = model_input.batch_size
        batch_size = self._find_graph_batch_size(real_batch_size)
        self._replay_batch_size = batch_size
        req_num = batch_size // self.mtp_size
        real_req_num = real_batch_size // self.mtp_size

        self.b_req_idx_pin[:real_batch_size].copy_(model_input.b_req_idx)
        self.b_seq_len_pin[:real_batch_size].copy_(model_input.b_seq_len)
        self.mem_indexes_pin[:real_batch_size].copy_(model_input.mem_indexes_cpu)
        self.b_position_delta_pin[:real_batch_size].copy_(model_input.b_position_delta)

        if batch_size != real_batch_size:
            self.b_req_idx_pin[real_batch_size:batch_size].fill_(self.hold_req_idx)
            padding_req_num = req_num - real_req_num
            padding_seq_len = torch.arange(2, self.mtp_size + 2, dtype=torch.int32).repeat(padding_req_num)
            self.b_seq_len_pin[real_batch_size:batch_size].copy_(padding_seq_len)
            self.mem_indexes_pin[real_batch_size:batch_size].fill_(self.hold_mem_index)
            self.b_position_delta_pin[real_batch_size:batch_size].zero_()

        for index, req in enumerate(run_reqs):
            shm_param = req.sampling_param.shm_param
            self.temperature_pin[index] = shm_param.temperature
            self.top_k_pin[index] = self.vocab_size if shm_param.top_k <= 0 else shm_param.top_k
            self.top_p_pin[index] = shm_param.top_p
        if batch_size != real_batch_size:
            self.temperature_pin[real_batch_size:batch_size].fill_(1.0)
            self.top_k_pin[real_batch_size:batch_size].fill_(1)
            self.top_p_pin[real_batch_size:batch_size].fill_(1.0)

        has_position_delta = bool(self.b_position_delta_pin[:real_batch_size].any())
        self.b_req_idx[:batch_size].copy_(self.b_req_idx_pin[:batch_size], non_blocking=True)
        self.b_seq_len[:batch_size].copy_(self.b_seq_len_pin[:batch_size], non_blocking=True)
        self.mem_indexes[:batch_size].copy_(self.mem_indexes_pin[:batch_size], non_blocking=True)
        self.temperature[:batch_size].copy_(self.temperature_pin[:batch_size], non_blocking=True)
        self.top_k[:batch_size].copy_(self.top_k_pin[:batch_size], non_blocking=True)
        self.top_p[:batch_size].copy_(self.top_p_pin[:batch_size], non_blocking=True)
        self._flush_position_delta(has_position_delta, batch_size)

        self._reset_padding_linear_state()
        bundle = self.graphs[batch_size]
        bundle.verify_graph.replay()
        return FusedStepOutput(
            next_token_ids=self.out_next_token_ids[:real_batch_size],
            next_token_logprobs=self.out_next_token_logprobs[:real_batch_size],
            mtp_accept_len=bundle.mtp_accept_len[:real_req_num],
            accepted_index=bundle.accepted_index[:real_batch_size],
        )

    def replay_draft(self):
        self.graphs[self._replay_batch_size].draft_graph.replay()
