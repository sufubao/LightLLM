from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.common.linear_att_cache_manager.layer_cache import LayerCache
from lightllm.common.linear_att_cache_manager.linear_att_buffer_manager import LinearAttCacheManager
from lightllm.utils.envs_utils import get_env_start_args

from .base import ReqManager


if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


@dataclass
class LinearStateSnapshot:
    """Canonical state for exactly ``exact_len`` processed input tokens.

    Both tensors have independent storage, or are protected by the owning
    capture ticket. A CPU snapshot retains its GPU source until the DMA event
    completes. Merely recording an event does not preserve source lifetime.
    """

    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    exact_len: int
    ready_event: torch.cuda.Event
    _sources: tuple = field(default_factory=tuple, repr=False)

    def wait(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        (stream or torch.cuda.current_stream()).wait_event(self.ready_event)

    def is_ready(self) -> bool:
        ready = self.ready_event.query()
        if ready:
            self._sources = ()
        return ready

    def to_cpu(self, stream: Optional[torch.cuda.Stream] = None) -> "LinearStateSnapshot":
        """Copy into pinned storage; callers must retain the result until ready."""
        if self.conv_state.device.type == "cpu":
            return self
        stream = stream or torch.cuda.current_stream(self.conv_state.device)
        conv = torch.empty(self.conv_state.shape, dtype=self.conv_state.dtype, device="cpu", pin_memory=True)
        ssm = torch.empty(self.ssm_state.shape, dtype=self.ssm_state.dtype, device="cpu", pin_memory=True)
        with torch.cuda.stream(stream):
            self.wait(stream)
            conv.copy_(self.conv_state, non_blocking=True)
            ssm.copy_(self.ssm_state, non_blocking=True)
            self.conv_state.record_stream(stream)
            self.ssm_state.record_stream(stream)
            event = torch.cuda.Event()
            event.record(stream)
        return LinearStateSnapshot(conv, ssm, self.exact_len, event, (self,))


@dataclass
class LinearStateCaptureStaging:
    """Bounded, reusable GPU storage. The caller owns its exclusive lease.

    Reuse is allowed only once every consumer/transfer of the previous capture
    has completed, not merely when ``ready_event`` is signaled. Slot metadata
    with req_indices == -1 denotes an unused slot and its state must not be read.
    """

    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    req_indices: torch.Tensor
    mtp_rows: torch.Tensor
    exact_lengths: torch.Tensor
    source_rows: torch.Tensor
    ready_event: Optional[torch.cuda.Event] = None

    def clone_snapshot(self, slot: int, exact_len: int) -> LinearStateSnapshot:
        """Detach a selected slot before reusing this staging allocation.

        The caller must have validated the selected slot's metadata and length.
        This method does not perform a hidden device-to-host metadata read.
        """
        assert 0 <= slot < self.conv_state.shape[0] and exact_len > 0
        assert self.ready_event is not None
        stream = torch.cuda.current_stream(self.conv_state.device)
        stream.wait_event(self.ready_event)
        conv = self.conv_state[slot].clone()
        ssm = self.ssm_state[slot].clone()
        event = torch.cuda.Event()
        event.record(stream)
        return LinearStateSnapshot(conv, ssm, exact_len, event)


class ReqManagerForMamba(ReqManager):
    def __init__(self, max_request_num, max_sequence_length, mem_manager, linear_config: LinearAttCacheConfig):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.mtp_step = get_env_start_args().mtp_step
        # 因为在mtp的推理中，需要标记每个请求对应的mtp index状态(conv state 和 ssm state)，在mtp对应序列中
        # 的真实位置，所以需要需要一个标记来记录，不然算子无法找到真实的处理起点。
        self.req_to_mtp_state_index = (
            torch.zeros((max_request_num + 1,), dtype=torch.int32, device="cuda") if self.mtp_step > 0 else None
        )
        # 突然想到， 在linear att 开启mtp的模式中，现在的prefill linear att 算子默认是从0的位置读取信息进行操作
        # 所以不能支持 prefill decode mixed 操作了，因为一个decode过的请求，重新用prefill 算子跑，会出现读错linear
        # 状态位置的问题。导致bug, 在这里加个断言，以后可以支持上 TODO
        if self.mtp_step > 0:
            assert get_env_start_args().enable_prefill_decode_mixed is False

        self.big_page_token_num = (
            get_env_start_args().linear_att_page_block_num * get_env_start_args().linear_att_hash_page_size
        )
        self.linear_config = linear_config

        self.req_to_conv_state = LayerCache(
            size=(max_request_num + 1),
            dtype=self.linear_config.conv_state_dtype,
            shape=self.linear_config.get_mtp_conv_state_shape(mtp_step=self.mtp_step),
            layer_num=self.linear_config.linear_layer_num,
            device="cuda",
        )
        self.req_to_ssm_state = LayerCache(
            size=(max_request_num + 1) * (self.mtp_step + 1),
            dtype=self.linear_config.ssm_state_dtype,
            shape=self.linear_config.get_ssm_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device="cuda",
        )
        return

    def init_linear_att_state(self, req: "InferReq"):
        conv_index = req.req_idx
        ssm_start = req.req_idx * (self.mtp_step + 1)
        self.req_to_conv_state.buffer[:, conv_index, ...].fill_(0)
        # #17: zero the FULL (mtp_step + 1)-row SSM block, not just canonical row +0, so a future
        # first-step verify reading offset>0 after fresh init never hits a never-written row (NaN).
        self.req_to_ssm_state.buffer[:, ssm_start : ssm_start + (self.mtp_step + 1), ...].fill_(0)
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return

    def allocate_linear_state_staging(self, capacity: int) -> LinearStateCaptureStaging:
        """Allocate once outside CUDA graph replay, subject to a caller budget."""
        if capacity <= 0:
            raise ValueError("linear state staging capacity must be positive")
        config = self.linear_config
        device = self.req_to_conv_state.buffer.device
        conv = torch.empty(
            (capacity, config.linear_layer_num, *config.get_conv_state_shape()),
            dtype=config.conv_state_dtype,
            device=device,
        )
        ssm = torch.empty(
            (capacity, config.linear_layer_num, *config.get_ssm_state_shape()),
            dtype=config.ssm_state_dtype,
            device=device,
        )
        metadata = [torch.empty(capacity, dtype=torch.int32, device=device) for _ in range(4)]
        return LinearStateCaptureStaging(conv, ssm, *metadata)

    def freeze_linear_states(
        self,
        req_indices: torch.Tensor,
        mtp_rows: torch.Tensor,
        exact_lengths: torch.Tensor,
        capture_mask: torch.Tensor,
        staging: LinearStateCaptureStaging,
    ) -> LinearStateCaptureStaging:
        """Freeze masked candidates on the producing stream before the next write.

        Candidates are retained in logical row order until staging is full;
        excess candidates are skipped. ``mtp_rows`` is request-local. In a
        prefill batch pass zeros; after verification select the exact accepted
        input row, accounting for stop truncation and the unprocessed sample.
        This method performs no host synchronization or per-request allocation.
        """
        from lightllm.common.basemodel.triton_kernel.linear_att.capture_state import freeze_linear_states

        freeze_linear_states(
            self.req_to_conv_state.buffer,
            self.req_to_ssm_state.buffer,
            req_indices,
            mtp_rows,
            exact_lengths,
            capture_mask,
            staging.req_indices,
            staging.mtp_rows,
            staging.exact_lengths,
            staging.source_rows,
            staging.conv_state,
            staging.ssm_state,
            self.mtp_step + 1,
        )
        staging.ready_event = torch.cuda.Event()
        staging.ready_event.record()
        return staging

    def freeze_linear_state(self, req_idx: int, exact_len: int, mtp_row: int = 0) -> LinearStateSnapshot:
        """Freeze one known boundary; use the bounded batch API on decode paths."""
        if not 0 <= req_idx < self.HOLD_REQUEST_ID:
            raise ValueError("cannot capture an invalid or padding request slot")
        if not 0 <= mtp_row <= self.mtp_step or exact_len <= 0:
            raise ValueError("invalid exact prefix length or request-local MTP row")
        conv_width = self.linear_config.get_conv_state_shape()[-1]
        conv = self.req_to_conv_state.buffer[:, req_idx, ..., mtp_row : mtp_row + conv_width].clone()
        ssm = self.req_to_ssm_state.buffer[:, req_idx * (self.mtp_step + 1) + mtp_row, ...].clone()
        event = torch.cuda.Event()
        event.record()
        return LinearStateSnapshot(conv, ssm, exact_len, event)

    def restore_linear_state(self, snapshot: LinearStateSnapshot, req_idx: int) -> torch.cuda.Event:
        """Restore a canonical snapshot without carrying speculative state rows."""
        if not 0 <= req_idx < self.HOLD_REQUEST_ID:
            raise ValueError("cannot restore an invalid or padding request slot")
        config = self.linear_config
        if snapshot.conv_state.shape != (config.linear_layer_num, *config.get_conv_state_shape()):
            raise ValueError("incompatible convolution state layout")
        if snapshot.ssm_state.shape != (config.linear_layer_num, *config.get_ssm_state_shape()):
            raise ValueError("incompatible SSM state layout")
        if snapshot.conv_state.dtype != config.conv_state_dtype or snapshot.ssm_state.dtype != config.ssm_state_dtype:
            raise ValueError("incompatible linear state dtype")
        snapshot.wait()
        conv_width = config.get_conv_state_shape()[-1]
        self.req_to_conv_state.buffer[:, req_idx, ...].zero_()
        self.req_to_conv_state.buffer[:, req_idx, ..., :conv_width].copy_(snapshot.conv_state, non_blocking=True)
        ssm_start = req_idx * (self.mtp_step + 1)
        self.req_to_ssm_state.buffer[:, ssm_start : ssm_start + self.mtp_step + 1, ...].zero_()
        self.req_to_ssm_state.buffer[:, ssm_start, ...].copy_(snapshot.ssm_state, non_blocking=True)
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req_idx] = 0
        event = torch.cuda.Event()
        event.record()
        # The caller retains snapshot until this H2D/read event has completed.
        return event

    def get_mamba_cache(self, layer_idx_in_all: int):
        assert (
            0 <= layer_idx_in_all < self.linear_config.all_layer_num
        ), f"invalid transformer layer index {layer_idx_in_all}"
        layer_idx_in_linear = layer_idx_in_all - (layer_idx_in_all // self.linear_config.full_attention_interval)
        conv_states = self.req_to_conv_state.buffer[layer_idx_in_linear]
        ssm_states = self.req_to_ssm_state.buffer[layer_idx_in_linear]
        return conv_states, ssm_states

    def copy_big_page_buffer_to_linear_att_state(self, big_page_buffer_idx: int, req: "InferReq"):
        big_page_buffers: LinearAttCacheManager = self.mem_manager.linear_att_big_page_buffers

        conv_state, ssm_state = big_page_buffers.get_state_cache(buffer_idx=big_page_buffer_idx)
        conv_dest = req.req_idx
        ssm_dest = req.req_idx * (self.mtp_step + 1)
        conv_cache_width = conv_state.shape[-1]
        self.req_to_conv_state.buffer[:, conv_dest, ..., :conv_cache_width] = conv_state
        self.req_to_ssm_state.buffer[:, ssm_dest, ...] = ssm_state
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return

    def copy_small_page_buffer_to_linear_att_state(
        self, req: "InferReq", linear_att_small_page_buffers: LinearAttCacheManager
    ):
        conv_state, ssm_state = linear_att_small_page_buffers.get_state_cache(
            buffer_idx=req.shared_kv_node.small_page_buffer_idx
        )
        conv_dest = req.req_idx
        ssm_dest = req.req_idx * (self.mtp_step + 1)
        conv_cache_width = conv_state.shape[-1]
        # TODO 下面这个从 cpu cache 拷贝数据的 gpu的操作，是否是阻塞的操作。
        # 同时，非连续对象的拷贝，可能存在效率问题。
        self.req_to_conv_state.buffer[:, conv_dest, ..., :conv_cache_width] = conv_state
        self.req_to_ssm_state.buffer[:, ssm_dest, ...] = ssm_state
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return
