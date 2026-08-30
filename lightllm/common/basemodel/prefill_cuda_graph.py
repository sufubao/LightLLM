import torch
import triton
from typing import List, Tuple
from typing import Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor
from lightllm.distributed import dist_group_manager
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from .infer_struct import InferStateInfo
from .cuda_graph import CudaGraph

logger = init_logger(__name__)

PrefillGraphShape = Tuple[int, int, int]


class PrefillCudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self, decode_cuda_graph: CudaGraph, tp_world_size: int):
        self.graph = {}
        self.tp_world_size = tp_world_size
        if decode_cuda_graph is not None:
            self.mempool = decode_cuda_graph.mempool  # prefill 和 decode 共享一个 mempool
        else:
            self.mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None

        self.args = get_env_start_args()
        self.enable_prefill_microbatch_overlap = self.args.enable_prefill_microbatch_overlap
        self.max_handle_token_num = self.args.prefill_cudagraph_max_handle_token
        if self.args.batch_max_tokens is not None:
            self.max_handle_token_num = min(self.max_handle_token_num, self.args.batch_max_tokens)

        capture_shape_specs = self.args.prefill_cudagraph_capture_shapes
        if capture_shape_specs:
            graph_shapes = self.parse_capture_shapes(capture_shape_specs)
        else:
            graph_handle_token_nums = (
                list(range(4, 33, 4))
                + list(range(48, 257, 16))
                + list(range(288, 513, 32))
                + list(range(576, 1024 + 1, 64))
                + list(range(1280, 4096 + 1, 256))
                + list(range(4608, self.max_handle_token_num + 1, 512))
            )
            graph_handle_token_nums = [e for e in graph_handle_token_nums if e <= self.max_handle_token_num]
            graph_handle_token_nums.append(self.max_handle_token_num)
            if self.args.enable_tpsp_mix_mode:
                graph_handle_token_nums = [
                    triton.cdiv(e, self.tp_world_size) * self.tp_world_size for e in graph_handle_token_nums
                ]
            graph_shapes = [(e, 1, e) for e in sorted(set(graph_handle_token_nums))]

        oversized_shapes = [shape for shape in graph_shapes if shape[0] > self.max_handle_token_num]
        if oversized_shapes:
            raise ValueError(
                f"Prefill CUDA Graph shapes {oversized_shapes} exceed the effective max token count "
                f"{self.max_handle_token_num}"
            )
        self.graph_shapes = tuple(sorted(set(graph_shapes)))
        # Retained for the overlap warmup path, which currently supports single-request shapes only.
        self.graph_handle_token_nums = [shape[0] for shape in self.graph_shapes if shape[1] == 1]
        logger.info(f"prefill cuda graph exact capture shapes: {self.graph_shapes}")

    @staticmethod
    def parse_capture_shapes(shape_specs: List[str]) -> List[PrefillGraphShape]:
        graph_shapes = []
        for spec in shape_specs:
            try:
                total_token_num, batch_size, max_q_seq_len = (int(value) for value in spec.split(":"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid prefill CUDA Graph shape {spec!r}; expected total_tokens:batch_size:max_q_seq_len"
                ) from exc
            if batch_size <= 0 or max_q_seq_len <= 0:
                raise ValueError(f"Prefill CUDA Graph shape values must be positive, got {spec!r}")
            if not max_q_seq_len + batch_size - 1 <= total_token_num <= batch_size * max_q_seq_len:
                raise ValueError(
                    f"Prefill CUDA Graph shape {spec!r} cannot describe {batch_size} positive sequences "
                    f"whose exact maximum query length is {max_q_seq_len}"
                )
            graph_shapes.append((total_token_num, batch_size, max_q_seq_len))
        return graph_shapes

    @staticmethod
    def _shape_from_infer_state(infer_state: InferStateInfo) -> PrefillGraphShape:
        return (
            infer_state.input_ids.shape[0],
            infer_state.batch_size,
            infer_state.max_q_seq_len,
        )

    def can_run(self, handle_token_num: int, batch_size: int, max_q_seq_len: int):
        return (handle_token_num, batch_size, max_q_seq_len) in self.graph_shapes

    def need_capture(self, handle_token_num: int, batch_size: int, max_q_seq_len: int):
        return (handle_token_num, batch_size, max_q_seq_len) not in self.graph

    def _capture_prefill(
        self, prefill_func, input_tensors: List[torch.Tensor], infer_state: InferStateInfo
    ) -> List[torch.Tensor]:
        graph_shape = self._shape_from_infer_state(infer_state)
        infer_state.mem_pool = self.mempool
        infer_state.prefill_cuda_graph_create_graph_obj()
        infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
        graph_input_tensors: List[torch.Tensor] = [torch.empty_like(e) for e in input_tensors]
        graph_out_tensors: List[torch.Tensor] = prefill_func(graph_input_tensors, infer_state)
        graph_out_tensors = [e.contiguous() for e in graph_out_tensors]
        infer_state.prefill_cuda_graph_get_current_capture_graph().__exit__(None, None, None)

        graph_input_tensors = [tensor_to_no_ref_tensor(e) for e in graph_input_tensors]
        graph_out_tensors = [tensor_to_no_ref_tensor(e) for e in graph_out_tensors]

        graph_hidden_collector = infer_state.hidden_collector
        hidden_tensor_count, hidden_tensor_nbytes = graph_hidden_collector.release_graph_tensor_ownership()
        logger.info(
            f"Prefill CUDA Graph hidden collector 已完成 capture 状态托管："
            f"shape={graph_shape}, "
            f"collector={graph_hidden_collector.__class__.__name__}, "
            f"no_ref_tensor_count={hidden_tensor_count}, "
            f"no_ref_tensor_nbytes={hidden_tensor_nbytes}。"
            f"这些 tensor 的固定地址由 graph memory pool 管理，collector 仅保留无所有权引用供 replay 使用。"
        )
        self.graph[graph_shape] = (
            infer_state,
            graph_input_tensors,
            graph_out_tensors,
            graph_hidden_collector,
        )
        self.replay(input_tensors, infer_state)

        return graph_out_tensors

    def _capture_prefill_overlap(
        self,
        prefill_func,
        input_tensors: List[torch.Tensor],
        infer_state: InferStateInfo,
        input_tensors1: List[torch.Tensor],
        infer_state1: InferStateInfo,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # TODO
        raise NotImplementedError("not impl")

    def capture_prefill(
        self,
        prefill_func,
        input_tensors: List[torch.Tensor],
        infer_state: InferStateInfo,
        input_tensors1: Optional[List[torch.Tensor]] = None,
        infer_state1: Optional[InferStateInfo] = None,
    ):
        """
        Capture the cuda graph for the prefill stage.
        input_tensor1 and infer_state1 is used for the overlap.
        """
        if self.enable_prefill_microbatch_overlap:
            return self._capture_prefill_overlap(
                prefill_func=prefill_func,
                input_tensors=input_tensors,
                infer_state=infer_state,
                input_tensors1=input_tensors1,
                infer_state1=infer_state1,
            )
        else:
            assert input_tensors1 is None and infer_state1 is None
            return self._capture_prefill(
                prefill_func=prefill_func, input_tensors=input_tensors, infer_state=infer_state
            )

    def _replay(self, input_tensors: List[torch.Tensor], infer_state: InferStateInfo) -> List[torch.Tensor]:
        graph_shape = self._shape_from_infer_state(infer_state)
        graph_infer_state, graph_input_tensors, graph_output_tensors, graph_hidden_collector = self.graph[graph_shape]
        graph_infer_state: InferStateInfo = graph_infer_state
        for graph_in_tensor, in_tensor in zip(graph_input_tensors, input_tensors):
            graph_in_tensor.copy_(in_tensor)

        graph_infer_state.copy_for_prefill_cuda_graph(new_infer_state=infer_state)
        # 首次 capture 后 replay 时，infer_state 与 graph_infer_state 是同一对象，
        # 需要先创建运行时实例，避免 finish_output 清空 graph 中保存的 capture collector。
        if infer_state.hidden_collector is graph_hidden_collector:
            infer_state.hidden_collector = graph_hidden_collector.new_instance()
        infer_state.hidden_collector.restore_graph_state(graph_hidden_collector)
        graph_infer_state.prefill_replay(infer_state)

        return graph_output_tensors

    def _replay_overlap(
        self,
        input_tensors: List[torch.Tensor],
        infer_state: InferStateInfo,
        input_tensors1: List[torch.Tensor],
        infer_state1: InferStateInfo,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        raise NotImplementedError("not impl")

    def replay(self, input_tensors, infer_state, input_tensor1=None, infer_state1=None):
        if self.enable_prefill_microbatch_overlap:
            return self._replay_overlap(input_tensors, infer_state, input_tensor1, infer_state1)
        else:
            assert input_tensor1 is None and infer_state1 is None
            return self._replay(input_tensors, infer_state)

    @torch.no_grad()
    def warmup(self, model):
        logger.info("Begin capture prefill cudagraph, remove the --enable_prefill_cudagraph to disable it.")
        # for typing easy
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model

        # prefill cuda graph init
        for total_token_num, batch_size, max_q_seq_len in self.graph_shapes[::-1]:
            logger.info(
                f"Capture prefill cudagraph, shape: {(total_token_num, batch_size, max_q_seq_len)}"
            )
            if batch_size > model.req_manager.max_request_num:
                raise ValueError(
                    f"Prefill CUDA Graph batch size {batch_size} exceeds max request count "
                    f"{model.req_manager.max_request_num}"
                )
            input_ids = torch.ones(total_token_num, dtype=torch.int64, device="cuda")
            mem_indexes = model.mem_manager.alloc(len(input_ids)).cuda()
            b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
            seq_lens = [max_q_seq_len] + [1] * (batch_size - 1)
            remaining_tokens = total_token_num - sum(seq_lens)
            for index in range(1, batch_size):
                added_tokens = min(remaining_tokens, max_q_seq_len - 1)
                seq_lens[index] += added_tokens
                remaining_tokens -= added_tokens
            assert remaining_tokens == 0
            b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
            b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
            b_is_decode_req = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
            b_prefill_start_loc = b_seq_len.cumsum(dim=0, dtype=torch.int32) - b_seq_len

            model_input = ModelInput(
                batch_size=batch_size,
                total_token_num=total_token_num,
                max_q_seq_len=max_q_seq_len,
                max_kv_seq_len=max_q_seq_len,
                max_cache_len=0,
                input_ids=input_ids,
                mem_indexes=mem_indexes,
                b_req_idx=b_req_idx,
                b_mtp_index=b_mtp_index,
                b_seq_len=b_seq_len,
                b_is_decode_req=b_is_decode_req,
                b_ready_cache_len=b_ready_cache_len,
                b_prefill_start_loc=b_prefill_start_loc,
                is_prefill=True,
                b_prefill_has_output_cpu=[False] * batch_size,
                multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
                **model._gen_special_model_input(token_num=total_token_num),
            )
            model_output: ModelOutput = model.forward(model_input)
            del model_output
            del input_ids
            del mem_indexes
            del b_req_idx
            del b_seq_len

            model.mem_manager.free_all()
            model.req_manager.free_all()
            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.cuda.empty_cache()

        logger.info(
            f"Capture prefill cudagraph success for exact shapes: {self.graph_shapes}."
        )

    @torch.no_grad()
    def warmup_overlap(self, model):
        logger.info("Begin capture prefill overlap cudagraph, remove the --enable_prefill_cudagraph to disable it.")
        # for typing easy
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model

        for handle_token_num in self.graph_handle_token_nums[::-1]:
            prefill_batches = []
            for micro_batch_index in [0, 1]:
                # dummy prefill, capture the cudagraph
                total_token_num = handle_token_num
                input_ids = torch.tensor([1 for _ in range(total_token_num)], dtype=torch.int64, device="cuda")
                mem_indexes = model.mem_manager.alloc(len(input_ids)).cuda()
                b_req_idx = torch.tensor([model.req_manager.HOLD_REQUEST_ID], dtype=torch.int32, device="cuda")
                b_seq_len = torch.empty(1, dtype=torch.int32, device="cuda")
                b_seq_len.fill_(total_token_num)
                b_mtp_index = torch.zeros(1, dtype=torch.int32, device="cuda")
                b_is_decode_req = torch.zeros(1, dtype=torch.bool, device="cuda")
                b_ready_cache_len = torch.zeros(1, dtype=torch.int32, device="cuda")
                b_prefill_start_loc = torch.zeros(1, dtype=torch.int32, device="cuda")

                micro_batch = ModelInput(
                    batch_size=1,
                    total_token_num=total_token_num,
                    max_q_seq_len=total_token_num,
                    max_kv_seq_len=total_token_num,
                    max_cache_len=0,
                    input_ids=input_ids,
                    mem_indexes=mem_indexes,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    b_seq_len=b_seq_len,
                    b_is_decode_req=b_is_decode_req,
                    b_ready_cache_len=b_ready_cache_len,
                    b_prefill_start_loc=b_prefill_start_loc,
                    is_prefill=True,
                    b_prefill_has_output_cpu=[False],
                    multimodal_params=[{"images": [], "audios": []}],
                    **model._gen_special_model_input(token_num=total_token_num),
                )

                prefill_batches.append(micro_batch)
                del micro_batch

                for var_name, var_value in list(locals().items()):
                    if isinstance(var_value, torch.Tensor):
                        del locals()[var_name]
                torch.cuda.empty_cache()

            _, _ = model.microbatch_overlap_prefill(prefill_batches[0], prefill_batches[1])

            model.mem_manager.free_all()
            model.req_manager.free_all()

            del prefill_batches

            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.cuda.empty_cache()

        logger.info(
            f"Capture overlap cudagraph success, handle_token_num <={self.max_handle_token_num} "
            f" will infer with cudagraph."
        )
