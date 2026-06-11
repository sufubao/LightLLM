import gzip
import os
import shutil
import threading
import torch
from typing import Optional
from lightllm.server.core.objs.io_objs import StartProfileCmd, StopProfileCmd
from lightllm.server.core.objs.profile_status_board import (
    ProfileStatusBoard,
    STATE_ARMED,
    STATE_ERROR,
    STATE_FLUSHING,
    STATE_IDLE,
    STATE_RUNNING,
    ERROR_EXPORT_FAILED,
    ERROR_NONE,
    ERROR_START_FAILED,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_ACTIVITY_MAP = {
    "CPU": torch.profiler.ProfilerActivity.CPU,
    "GPU": torch.profiler.ProfilerActivity.CUDA,
}


def _default_profiler_factory(cmd: StartProfileCmd):
    activities = [_ACTIVITY_MAP[a] for a in cmd.activities if a in _ACTIVITY_MAP]
    if not activities:
        # torch 把空 activities 列表当成 "全部都采", 这里显式报错而不是静默放大采集范围。
        raise ValueError(f"no valid activities in {cmd.activities!r}, expected subset of {sorted(_ACTIVITY_MAP)}")
    return torch.profiler.profile(
        activities=activities,
        with_stack=cmd.with_stack,
        record_shapes=cmd.record_shapes,
    )


class WorkerProfilerManager:
    """
    每个推理 rank 进程一个实例, 状态机: IDLE -> ARMED -> RUNNING -> FLUSHING -> IDLE。
    一个 "step" 是调度器的一个 forward step, 不包含 infer_loop 的空转迭代;
    同一 step 内的 MTP draft forward 和 DP-overlap microbatch 都只算一个 step。
    on_cmd 和 on_step_boundary 都只会被持有 overlap event 令牌的 infer_loop 线程调用
    (令牌串行化了两个线程的 launch 区段), 锁只是防御性的。
    kineto (torch.profiler) 的 start/stop 必须发生在同一线程: 跨线程 stop 会直接抛
    RuntimeError 并泄漏 profiler callback。因此 start 时记录 owner 线程, 非 owner 线程
    命中停止条件时只标记 _pending_stop, 由 owner 线程在下一个 step/pass boundary 真正
    执行 stop —— 生产中 num_steps 的捕获窗口因此可能多出一个 forward (±1 step)。
    停止时先 torch.cuda.synchronize() 排空两个线程已发射的全部 GPU 工作, 再 stop/export,
    保证捕获窗口覆盖完整的 forward。stop/export 在 infer 线程内同步执行, FLUSHING 期间
    服务会停顿 (状态板上可观测), 这是接受的取舍。
    """

    def __init__(
        self,
        rank_in_node: int,
        dp_rank_in_node: int,
        node_world_size: int,
        profiler_factory=None,
        status_board: Optional[ProfileStatusBoard] = None,
    ):
        self.rank_in_node = rank_in_node
        self.dp_rank_in_node = dp_rank_in_node
        self.status_board = (
            status_board if status_board is not None else ProfileStatusBoard(num_worker_slots=node_world_size)
        )
        self._slot = rank_in_node
        self._profiler_factory = profiler_factory if profiler_factory is not None else _default_profiler_factory
        self._lock = threading.Lock()
        self._state = STATE_IDLE
        self._cmd: Optional[StartProfileCmd] = None
        self._profiler = None
        self._start_at_ct = 0
        self._target_ct: Optional[int] = None
        self._owner_thread_ident: Optional[int] = None
        self._pending_stop = False
        self.forward_ct = 0
        self.status_board.set_slot(
            self._slot, state=STATE_IDLE, profile_id=0, forward_ct=0, target_ct=0, error_code=ERROR_NONE
        )

    def on_cmd(self, cmd):
        with self._lock:
            if isinstance(cmd, StartProfileCmd):
                if self._state != STATE_IDLE:
                    logger.warning(f"ignore start_profile cmd, profiler busy in state {self._state}")
                    return
                self._cmd = cmd
                self._start_at_ct = max(cmd.start_step if cmd.start_step is not None else 0, self.forward_ct + 1)
                self._state = STATE_ARMED
                self.status_board.set_slot(
                    self._slot,
                    state=STATE_ARMED,
                    profile_id=cmd.profile_id,
                    forward_ct=self.forward_ct,
                    target_ct=0,
                    error_code=ERROR_NONE,
                )
            elif isinstance(cmd, StopProfileCmd):
                # profile_id=0 表示停止任意 capture; 非 0 时只停止匹配的 capture
                if cmd.profile_id and self._cmd is not None and cmd.profile_id != self._cmd.profile_id:
                    logger.warning(f"ignore stale stop_profile cmd for profile_id {cmd.profile_id}")
                    return
                if self._state == STATE_RUNNING:
                    self._try_stop_and_export()
                elif self._state == STATE_ARMED:
                    self._state = STATE_IDLE
                    self._cmd = None
                    self.status_board.set_slot(self._slot, state=STATE_IDLE, profile_id=0, target_ct=0)
        return

    def on_step_boundary(self):
        # 未开启 profiling 时的快路径, 只有一次整型比较的开销。
        if self._state == STATE_IDLE:
            self.forward_ct += 1
            return
        with self._lock:
            self.forward_ct += 1
            if self._state == STATE_RUNNING:
                if self._pending_stop or (self._target_ct is not None and self.forward_ct >= self._target_ct):
                    # 在本次 forward 发射之前停止; 若停止条件上次命中在非 owner 线程, 窗口可能多一个 forward。
                    self._try_stop_and_export()
                else:
                    self.status_board.set_slot(self._slot, forward_ct=self.forward_ct)
            elif self._state == STATE_ARMED and self.forward_ct >= self._start_at_ct:
                self._do_start()
        return

    def on_pass_boundary(self):
        """空转迭代中只用于补执行被推迟到 owner 线程的 stop, 不计步。"""
        if not self._pending_stop:
            return
        with self._lock:
            if self._state == STATE_RUNNING and self._pending_stop:
                self._try_stop_and_export()
        return

    def _try_stop_and_export(self):
        # kineto 的 start/stop 必须发生在同一线程 (跨线程 stop 实测直接抛错并泄漏 callback),
        # 非 owner 线程命中停止条件时只做标记, 由 owner 线程在下一个 boundary 真正执行 stop。
        # 因此生产中 num_steps 的捕获窗口可能多出一个 forward (±1)。
        if threading.get_ident() == self._owner_thread_ident:
            self._stop_and_export()
        else:
            if not self._pending_stop:
                logger.info("profiler stop deferred to owner thread")
            self._pending_stop = True
        return

    def _do_start(self):
        try:
            self._profiler = self._profiler_factory(self._cmd)
            self._profiler.start()
            self._owner_thread_ident = threading.get_ident()
            self._target_ct = self.forward_ct + self._cmd.num_steps if self._cmd.num_steps is not None else None
            self._state = STATE_RUNNING
            self.status_board.set_slot(
                self._slot, state=STATE_RUNNING, forward_ct=self.forward_ct, target_ct=self._target_ct or 0
            )
            logger.info(f"profiler started at forward_ct {self.forward_ct}, target_ct {self._target_ct}")
        except Exception as e:
            logger.exception(f"profiler start failed: {e}")
            self._profiler = None
            self._cmd = None
            self._state = STATE_IDLE
            self.status_board.set_slot(self._slot, state=STATE_ERROR, error_code=ERROR_START_FAILED)
        return

    def _stop_and_export(self):
        self.status_board.set_slot(self._slot, state=STATE_FLUSHING, forward_ct=self.forward_ct)
        cmd = self._cmd
        json_tmp_path = None
        gz_tmp_path = None
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._profiler.stop()
            os.makedirs(cmd.output_dir, exist_ok=True)
            trace_name = f"{cmd.profile_prefix}-{cmd.profile_id}-TP-{self.rank_in_node}-DP-{self.dp_rank_in_node}"
            final_path = os.path.join(cmd.output_dir, trace_name + ".trace.json.gz")
            json_tmp_path = final_path + ".json.tmp"
            gz_tmp_path = final_path + ".gz.tmp"
            self._profiler.export_chrome_trace(json_tmp_path)
            with open(json_tmp_path, "rb") as f_in, gzip.open(gz_tmp_path, "wb", compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(json_tmp_path)
            os.rename(gz_tmp_path, final_path)
            self.status_board.set_slot(self._slot, state=STATE_IDLE, error_code=ERROR_NONE, target_ct=0)
            logger.info(f"profiler trace exported to {final_path}")
        except Exception as e:
            logger.exception(f"profiler stop/export failed: {e}")
            for stale_path in (json_tmp_path, gz_tmp_path):
                if stale_path is not None:
                    try:
                        os.remove(stale_path)
                    except OSError:
                        pass
            self.status_board.set_slot(self._slot, state=STATE_ERROR, error_code=ERROR_EXPORT_FAILED)
        finally:
            self._profiler = None
            self._cmd = None
            self._target_ct = None
            self._pending_stop = False
            self._owner_thread_ident = None
            self._state = STATE_IDLE
        return
