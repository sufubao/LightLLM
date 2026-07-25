import os
import torch
import time
from typing import List, Optional, Callable, Dict, Any
from queue import Queue
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.common.basemodel.batch_objs import ModelOutput, ModelInput
from lightllm.common.basemodel.triton_kernel.gather_token_id import scatter_token
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req
from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    linear_att_mtp_state_index_update,
    mtp_scatter_next_token_ids,
)
from lightllm.common.mtp_scheduler import (
    MTPDecodePlanScheduler,
    load_mtp_decode_profile,
)
from lightllm.common.mtp_workspace import select_runtime_mtp_step
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from .control_state import ControlState

logger = init_logger(__name__)


def select_mtp_profile(decode_reqs: List[InferReq], runtime_mtp_step: int) -> str:
    if all(req.mtp_proposal_step >= runtime_mtp_step for req in decode_reqs):
        return "mtp"
    return "transition"


class ChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

        # 用于控制每一步是执行prefill 和 decode 还是跳过
        self.control_state_machine = ControlState()

        # 在 mtp 模式下切换绑定的prefill 和 decode 函数
        if get_env_start_args().mtp_mode:
            self.prefill = self.prefill_mtp
            self.decode = self.decode_mtp
            self.is_mtp_eagle = get_env_start_args().mtp_mode in ["eagle_with_att", "eagle_no_att"]
            self.num_mtp_models = 1 if self.is_mtp_eagle else get_env_start_args().mtp_step
            self._draft_decode_func = self._draft_decode_eagle if self.is_mtp_eagle else self._draft_decode_vanilla
        else:
            self.prefill = self.prefill_normal
            self.decode = self.decode_normal

        self.classed_req_strict_prefill = False
        self._last_mtp_profile = None
        self._mtp_profile_counts = {"dense": 0, "transition": 0, "mtp": 0}
        self._mtp_plan_scheduler = None
        self._selected_mtp_plan = None
        args = get_env_start_args()
        if getattr(args, "dynamic_mtp", False) and args.mtp_scheduler_profile is not None:
            self._mtp_plan_scheduler = MTPDecodePlanScheduler(
                workspace_rows=args.mtp_workspace_rows,
                max_mtp_step=args.max_mtp_step,
                profile=load_mtp_decode_profile(args.mtp_scheduler_profile),
            )
        return

    def _select_decode_candidates(self, decode_candidates: List[InferReq]) -> List[InferReq]:
        if self._mtp_plan_scheduler is None:
            self._selected_mtp_plan = None
            return decode_candidates
        selected, self._selected_mtp_plan = self._mtp_plan_scheduler.select(decode_candidates)
        return selected

    def _get_selected_runtime_mtp_step(self):
        if self._selected_mtp_plan is None:
            return None
        return self._selected_mtp_plan.mtp_step

    def _mark_mtp_plan_step(self):
        if self._mtp_plan_scheduler is not None:
            self._mtp_plan_scheduler.mark_mtp_step()
        return

    def _prepare_retained_mtp_workspace(self, model_input: ModelInput, decode_reqs: List[InferReq]):
        if getattr(self.model.req_manager, "memory_aware_mtp", False):
            model_input.use_contiguous_mtp_ssm_workspace = self.model.req_manager.can_use_contiguous_mtp_ssm_workspace(
                logical_batch_size=len(decode_reqs),
                runtime_mtp_step=model_input.runtime_mtp_step,
            )
            model_input.b_mtp_workspace_idx = self.model.req_manager.prepare_mtp_workspace(
                [req.req_idx for req in decode_reqs],
                runtime_mtp_step=model_input.runtime_mtp_step,
                use_contiguous_ssm_workspace=(model_input.use_contiguous_mtp_ssm_workspace),
            )
        return

    def infer_loop(self):
        torch.cuda.set_device(get_current_device_id())
        try:
            while True:
                event_pack = self.overlap_event_manager.get_overlap_event_pack()
                # 关闭overlap 模式
                if not self.support_overlap:
                    event_pack._close_overlap()

                event_pack.wait_to_forward()

                self._try_read_new_reqs()

                prefill_reqs, decode_reqs = self._get_classed_reqs(
                    no_decode=self.classed_req_no_decode,
                    strict_prefill=self.classed_req_strict_prefill,
                    recover_paused=self.control_state_machine.try_recover_paused_reqs(),
                )

                run_way = self.control_state_machine.select_run_way(prefill_reqs=prefill_reqs, decode_reqs=decode_reqs)

                if run_way.is_prefill():
                    # 进行一次流同步，保证 _try_read_new_reqs 中的一些算子操作，必然已经完成。
                    # 防止后续的推理流程读取到显存中可能存在错误的数据。
                    g_infer_context.get_overlap_stream().wait_stream(torch.cuda.current_stream())
                    self.prefill(
                        event_pack=event_pack,
                        prefill_reqs=prefill_reqs,
                    )
                    continue
                elif run_way.is_decode():
                    # 进行一次流同步，保证 _try_read_new_reqs 中的一些算子操作，必然已经完成。
                    # 防止后续的推理流程读取到显存中可能存在错误的数据。
                    g_infer_context.get_overlap_stream().wait_stream(torch.cuda.current_stream())
                    self.decode(
                        event_pack=event_pack,
                        decode_reqs=decode_reqs,
                    )
                    continue
                elif run_way.is_pass():
                    event_pack.notify_post_handle_and_wait_pre_post_handle()
                    event_pack.notify_forward_and_wait_post_handle()
                    event_pack.notify_pre_post_handle()
                    time.sleep(0.02)
                    continue

        except BaseException as e:
            self.logger.exception(str(e))
            raise e

    def prefill_normal(
        self,
        event_pack: OverlapEventPack,
        prefill_reqs: List[InferReq],
    ):
        # 第一阶段: 模型推理
        model_input, run_reqs = prepare_prefill_inputs(prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            _, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                logits=model_output.logits,
                b_req_idx=model_input.b_req_idx,
                b_mtp_index=model_input.b_mtp_index,
                run_reqs=run_reqs,
                is_prefill=True,
                b_prefill_has_output_cpu=model_input.b_prefill_has_output_cpu,
                mask_func=self.prefill_mask_func,
            )
            g_infer_context.copy_linear_att_state_to_cache_buffer(
                b_req_idx=model_input.b_req_idx,
                reqs=run_reqs,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
            pd_prefill_chunked_handle_func=self.pd_prefill_chunked_handle_func,
        )
        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def decode_normal(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
    ):
        model_input, run_reqs = prepare_decode_inputs(decode_reqs)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            _, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                logits=model_output.logits,
                b_req_idx=model_input.b_req_idx,
                b_mtp_index=model_input.b_mtp_index,
                run_reqs=run_reqs,
                is_prefill=False,
                mask_func=self.decode_mask_func,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def prefill_mtp(
        self,
        event_pack: OverlapEventPack,
        prefill_reqs: List[InferReq],
    ):
        model_input, run_reqs = prepare_prefill_inputs(prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            next_token_ids, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                logits=model_output.logits,
                b_req_idx=model_input.b_req_idx,
                b_mtp_index=model_input.b_mtp_index,
                run_reqs=run_reqs,
                is_prefill=True,
                b_prefill_has_output_cpu=model_input.b_prefill_has_output_cpu,
                mask_func=self.prefill_mask_func,
            )
            # mtp kv fill
            self._draft_prefill_forward(
                model_input=model_input, model_output=model_output, next_token_ids=next_token_ids
            )
            for req in run_reqs:
                req.mtp_proposal_step = self.mtp_step
            g_infer_context.copy_linear_att_state_to_cache_buffer(
                b_req_idx=model_input.b_req_idx,
                reqs=run_reqs,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()

        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
            pd_prefill_chunked_handle_func=self.pd_prefill_chunked_handle_func,
        )

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def _init_mtp_fused_graph(self):
        self.mtp_fused_graph = None
        self.mtp_fused_graphs = {}
        self._init_mtp_chain_scratch()
        if get_env_start_args().disable_cudagraph:
            return
        if not (self.is_mtp_eagle and self.num_mtp_models == 1):
            return
        if self.classed_req_no_decode or self.decode_mask_func is not None:
            return
        if self.enable_decode_microbatch_overlap or self.args.dp > 1:
            return
        from lightllm.common.basemodel.attention import FlashInferAttBackend, MlaFlashInferAttBackend

        flashinfer_backend_types = (FlashInferAttBackend, MlaFlashInferAttBackend)
        for model in (self.model, self.draft_models[0]):
            decode_backends = (model.decode_att_backend, getattr(model, "decode_att_backend1", None))
            if any(isinstance(att_backend, flashinfer_backend_types) for att_backend in decode_backends):
                logger.info("mtp fused decode graph disabled for FlashInfer attention backend")
                return
        from lightllm.utils.envs_utils import enable_diverse_mode_gqa_decode_fast_kernel

        if enable_diverse_mode_gqa_decode_fast_kernel():
            return
        if os.getenv("LIGHTLLM_DISABLE_MTP_FUSED_GRAPH", "0") == "1":
            logger.info("mtp fused decode graph disabled by env")
            return
        from .mtp_fused_decode_graph import MTPFusedDecodeGraph

        runtime_steps = range(1, self.mtp_step + 1) if getattr(self.args, "dynamic_mtp", False) else [self.mtp_step]
        for runtime_mtp_step in runtime_steps:
            graph = MTPFusedDecodeGraph(backend=self, runtime_mtp_step=runtime_mtp_step)
            graph.warmup()
            self.mtp_fused_graphs[runtime_mtp_step] = graph
        self.mtp_fused_graph = self.mtp_fused_graphs.get(self.mtp_step)
        return

    def _init_mtp_chain_scratch(self):
        self.mtp_chain_scratch = None
        if not (self.is_mtp_eagle and self.mtp_step > 1):
            return
        workspace_rows = getattr(
            get_env_start_args(),
            "mtp_workspace_rows",
            get_env_start_args().running_max_req_size,
        )
        scratch_num = max((workspace_rows // step) * (step + 1) * (step - 1) for step in range(1, self.mtp_step + 1))
        scratch_cpu = g_infer_context.req_manager.mem_manager.alloc(scratch_num)
        self.mtp_chain_scratch = scratch_cpu.cuda()
        logger.info(f"mtp chain scratch kv slots reserved: {scratch_num}")
        return

    def decode_mtp(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
    ):
        """
        MTP解码的通用流程，整合eagle和vanilla的共同逻辑
        """
        selected_mtp_step = self._get_selected_runtime_mtp_step()
        if selected_mtp_step is not None:
            runtime_mtp_step = selected_mtp_step
        elif getattr(get_env_start_args(), "dynamic_mtp", False):
            runtime_mtp_step = select_runtime_mtp_step(
                logical_batch_size=len(decode_reqs),
                workspace_rows=get_env_start_args().mtp_workspace_rows,
                max_mtp_step=get_env_start_args().max_mtp_step,
            )
        else:
            runtime_mtp_step = self.mtp_step
        if runtime_mtp_step == 0:
            self._mtp_profile_counts["dense"] += 1
            profile_key = ("dense", 0)
            if profile_key != self._last_mtp_profile:
                logger.info(
                    f"MTP decode profile=dense, runtime_mtp_step=0, "
                    f"logical_batch={len(decode_reqs)}, counts={self._mtp_profile_counts}"
                )
                self._last_mtp_profile = profile_key
            self._decode_transition_mtp_profile(
                event_pack=event_pack,
                decode_reqs=decode_reqs,
                runtime_mtp_step=0,
            )
            self._mark_mtp_plan_step()
            return
        profile = select_mtp_profile(
            decode_reqs=decode_reqs,
            runtime_mtp_step=runtime_mtp_step,
        )
        self._mtp_profile_counts[profile] += 1
        profile_key = (profile, runtime_mtp_step)
        if profile_key != self._last_mtp_profile:
            logger.info(
                f"MTP decode profile={profile}, runtime_mtp_step={runtime_mtp_step}, "
                f"logical_batch={len(decode_reqs)}, counts={self._mtp_profile_counts}"
            )
            self._last_mtp_profile = profile_key

        if profile == "transition":
            self._decode_transition_mtp_profile(
                event_pack=event_pack,
                decode_reqs=decode_reqs,
                runtime_mtp_step=runtime_mtp_step,
            )
            return

        model_input, run_reqs = prepare_decode_inputs(decode_reqs, runtime_mtp_step=runtime_mtp_step)

        mtp_fused_graph = self.mtp_fused_graphs.get(runtime_mtp_step)
        if mtp_fused_graph is not None and mtp_fused_graph.can_run(
            decode_reqs=decode_reqs,
            max_kv_seq_len=model_input.max_kv_seq_len,
            batch_size=model_input.batch_size,
        ):
            self._decode_mtp_fused(
                event_pack=event_pack,
                model_input=model_input,
                run_reqs=run_reqs,
                decode_reqs=decode_reqs,
                mtp_fused_graph=mtp_fused_graph,
            )
            return

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            self._prepare_retained_mtp_workspace(model_input=model_input, decode_reqs=decode_reqs)
            b_mtp_index_cpu = model_input.b_mtp_index
            model_output = self.model.forward(model_input)
            next_token_ids, next_token_logprobs = sample(model_output.logits, run_reqs, self.eos_id)
            # verify the next_token_ids
            b_req_mtp_start_loc = [index for index, mtp_index in enumerate(b_mtp_index_cpu) if mtp_index == 0]
            b_req_mtp_start_loc = g_pin_mem_manager.gen_from_list(
                key="b_req_mtp_start_loc",
                data=b_req_mtp_start_loc,
                dtype=torch.int32,
            ).cuda(non_blocking=True)

            mtp_accept_len, accepted_index = self._verify_mtp_v2(
                new_next_token_ids=next_token_ids,
                b_req_idx=model_input.b_req_idx,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
            )
            if self.is_linear_att_mixed_model:
                linear_att_mtp_state_index_update(
                    req_to_mtp_state_index=self.model.req_manager.req_to_mtp_state_index,
                    b_req_mtp_start_loc=b_req_mtp_start_loc,
                    b_req_idx=model_input.b_req_idx,
                    b_mtp_index=model_input.b_mtp_index,
                    accepted_index=accepted_index,
                    max_mtp_step=runtime_mtp_step + 1,
                )
            accepted_index_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="accepted_index",
                gpu_tensor=accepted_index,
            )
            mtp_accept_len_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="mtp_accept_len",
                gpu_tensor=mtp_accept_len,
            )
            verify_event = torch.cuda.Event()
            verify_event.record()

            next_token_ids_cpu, next_token_logprobs_cpu = self._async_copy_next_token_infos_to_pin_mem(
                next_token_ids, next_token_logprobs
            )

            # 调用具体的draft decode函数
            additional_mem_indexes_cpu = self._draft_decode_func(
                main_model_input=model_input,
                main_model_output=model_output,
                next_token_ids=next_token_ids,
                mtp_accept_len=mtp_accept_len,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                runtime_mtp_step=runtime_mtp_step,
            )

            g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
                b_req_idx=model_input.b_req_idx,
                next_token_ids=next_token_ids,
                mask=accepted_index == 1,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        verify_event.synchronize()
        self._mark_mtp_plan_step()
        verify_ok_reqs = [run_reqs[i] for i in range(len(run_reqs)) if accepted_index_cpu[i] == 1]
        update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()

        # 处理需要释放的内存索引
        need_free_mem_indexes = model_input.mem_indexes_cpu[accepted_index_cpu == 0]
        if additional_mem_indexes_cpu is not None:
            need_free_mem_indexes = torch.cat([need_free_mem_indexes, additional_mem_indexes_cpu], dim=0)

        self._update_mtp_accept_ratio(decode_reqs=decode_reqs, mtp_accept_len_cpu=mtp_accept_len_cpu)
        for req in decode_reqs:
            req.mtp_proposal_step = runtime_mtp_step
        select_mask = accepted_index_cpu.to(dtype=torch.bool)
        self._post_handle(
            run_reqs=verify_ok_reqs,
            next_token_ids=next_token_ids_cpu[select_mask],
            next_token_logprobs=next_token_logprobs_cpu[select_mask],
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        if len(need_free_mem_indexes) > 0:
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def _decode_transition_mtp_profile(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
        runtime_mtp_step: int,
    ):
        model_input, run_reqs = prepare_decode_inputs(decode_reqs, runtime_mtp_step=0)
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            if getattr(self.model.req_manager, "memory_aware_mtp", False):
                self.model.req_manager.materialize_mtp_state([req.req_idx for req in decode_reqs])
            model_output = self.model.forward(model_input)
            next_token_ids, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                logits=model_output.logits,
                b_req_idx=model_input.b_req_idx,
                b_mtp_index=model_input.b_mtp_index,
                run_reqs=run_reqs,
                is_prefill=False,
                mask_func=self.decode_mask_func,
            )

            req_num = len(decode_reqs)
            b_req_mtp_start_loc = torch.arange(req_num, dtype=torch.int32, device="cuda")
            mtp_accept_len = torch.ones(req_num, dtype=torch.int32, device="cuda")
            self._draft_decode_func(
                main_model_input=model_input,
                main_model_output=model_output,
                next_token_ids=next_token_ids,
                mtp_accept_len=mtp_accept_len,
                b_req_mtp_start_loc=b_req_mtp_start_loc,
                runtime_mtp_step=runtime_mtp_step,
            )

            sync_event = torch.cuda.Event()
            sync_event.record()

        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=False)

        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()
        for req in decode_reqs:
            req.mtp_proposal_step = runtime_mtp_step
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        event_pack.notify_pre_post_handle()
        return

    def _decode_mtp_fused(
        self,
        event_pack: OverlapEventPack,
        model_input: ModelInput,
        run_reqs: List[InferReq],
        decode_reqs: List[InferReq],
        mtp_fused_graph,
    ):
        """
        整个 mtp decode step (verify fwd + 采样 + verify + draft chain) 单 cuda graph 的执行路径。
        与 decode_mtp 的阶段/事件协议保持一致。
        """
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            self._prepare_retained_mtp_workspace(model_input=model_input, decode_reqs=decode_reqs)
            fused_out = mtp_fused_graph.replay_verify(model_input=model_input, run_reqs=run_reqs)
            accepted_index_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="accepted_index",
                gpu_tensor=fused_out.accepted_index,
            )
            mtp_accept_len_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="mtp_accept_len",
                gpu_tensor=fused_out.mtp_accept_len,
            )
            next_token_ids_cpu, next_token_logprobs_cpu = self._async_copy_next_token_infos_to_pin_mem(
                next_token_ids=fused_out.next_token_ids,
                next_token_logprobs=fused_out.next_token_logprobs,
            )
            verify_event = torch.cuda.Event()
            verify_event.record()

            mtp_fused_graph.replay_draft()
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        verify_event.synchronize()
        self._mark_mtp_plan_step()
        verify_ok_reqs = [run_reqs[i] for i in range(len(run_reqs)) if accepted_index_cpu[i] == 1]
        update_packs = self._pre_post_handle(verify_ok_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()

        need_free_mem_indexes = model_input.mem_indexes_cpu[accepted_index_cpu == 0]

        self._update_mtp_accept_ratio(decode_reqs=decode_reqs, mtp_accept_len_cpu=mtp_accept_len_cpu)
        for req in decode_reqs:
            req.mtp_proposal_step = model_input.runtime_mtp_step
        select_mask = accepted_index_cpu.to(dtype=torch.bool)
        self._post_handle(
            run_reqs=verify_ok_reqs,
            next_token_ids=next_token_ids_cpu[select_mask],
            next_token_logprobs=next_token_logprobs_cpu[select_mask],
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        if len(need_free_mem_indexes) > 0:
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def _draft_prefill_forward(self, model_input: ModelInput, model_output: ModelOutput, next_token_ids: torch.Tensor):
        # spec prefill: MTP, 这个地方只是为了填充draft model的 kv， 并不会使用生成的token_id。
        draft_model_input = model_input
        draft_model_output = model_output
        draft_next_token_ids_gpu = next_token_ids
        for draft_model_idx in range(self.num_mtp_models):
            draft_model_input = prepare_mtp_prefill_inputs(
                model_input=draft_model_input,
                b_next_token_ids=draft_next_token_ids_gpu,
                mtp_draft_input_hiddens=draft_model_output.mtp_main_output_hiddens,
            )
            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids_gpu = self._gen_argmax_token_ids(draft_model_output)
        return

    def _draft_decode_vanilla(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        runtime_mtp_step: int,
    ):
        # share some inference info with the main model
        draft_model_input = main_model_input
        draft_model_output = main_model_output
        draft_next_token_ids = next_token_ids
        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids)
        # process the draft model output
        for draft_model_idx in range(runtime_mtp_step):
            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.mtp_draft_input_hiddens = draft_model_output.mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids = self._gen_argmax_token_ids(draft_model_output)
            all_next_token_ids.append(draft_next_token_ids)

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]

        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=main_model_input.b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )
        return None

    def _draft_decode_eagle(
        self,
        main_model_input: ModelInput,
        main_model_output: ModelOutput,
        next_token_ids: torch.Tensor,
        mtp_accept_len: torch.Tensor,
        b_req_mtp_start_loc: torch.Tensor,
        runtime_mtp_step: int,
    ):
        batch_size = main_model_input.batch_size
        verify_mem_indexes = main_model_input.mem_indexes

        # share some inference info with the main model
        draft_model_input = main_model_input
        draft_model_output = main_model_output
        draft_next_token_ids = next_token_ids
        all_next_token_ids = []
        all_next_token_ids.append(next_token_ids)
        # process the draft model output
        for _step in range(runtime_mtp_step):
            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.mtp_draft_input_hiddens = draft_model_output.mtp_main_output_hiddens
            # spec decode: MTP
            draft_model_idx = _step % self.num_mtp_models
            draft_model_output: ModelOutput = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids = self._gen_argmax_token_ids(draft_model_output)
            draft_model_input.b_seq_len += 1
            draft_model_input.max_kv_seq_len += 1
            if _step + 1 < runtime_mtp_step:
                draft_model_input.mem_indexes = self.mtp_chain_scratch[_step * batch_size : (_step + 1) * batch_size]
            all_next_token_ids.append(draft_next_token_ids)

        draft_model_input.b_seq_len -= runtime_mtp_step
        if runtime_mtp_step > 1:
            copy_kv_index_to_req(
                self.model.req_manager.req_to_token_indexs,
                main_model_input.b_req_idx,
                draft_model_input.b_seq_len,
                verify_mem_indexes,
            )
        draft_model_input.mem_indexes = verify_mem_indexes

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]

        mtp_scatter_next_token_ids(
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=main_model_input.b_req_idx,
            mtp_accept_len=mtp_accept_len,
        )
        return None
