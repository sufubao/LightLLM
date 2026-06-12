# Profile Server Endpoints (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add on-demand `POST /start_profile`, `POST /stop_profile`, `GET /profile_status` endpoints that capture per-rank `torch.profiler` chrome traces from the inference workers of a running LightLLM server.

**Architecture:** A `ProfileControlReq` rides the existing httpserver→router zmq PUSH socket; the router converts it to a `StartProfileCmd`/`StopProfileCmd` and broadcasts it to all worker ranks through the existing `ShmObjsIOBuffer` (same path as `AbortedReqCmd`). Each worker rank owns a `WorkerProfilerManager` state machine (IDLE→ARMED→RUNNING→FLUSHING→IDLE) whose transitions happen only at **forward-step boundaries** inside the infer loop. A per-rank shared-memory `ProfileStatusBoard` reports state back to the HTTP process (there is no reply channel). Spec: `docs/profile_server/design.md` (Part 3).

**Tech Stack:** Python, torch.profiler, zmq (existing sockets), multiprocessing shared memory (existing `create_or_link_shm` helper), FastAPI (existing app), pytest.

**Key codebase facts the implementer must know:**

- LightLLM is multi-process: httpserver, router, and N worker-rank processes are separate processes started with spawn. Config travels via `StartArgs` (`lightllm/server/core/objs/start_args_type.py`) + `get_env_start_args()`.
- httpserver→router messages: `HttpServerManager.send_to_router` (zmq PUSH, `httpserver/manager.py:50`) → router `zmq_recv_socket` (PULL), read in `_recv_new_reqs_and_schedule()` (`router/manager.py:490`), which today asserts `isinstance(recv_req, GroupReqIndexes)`.
- router→workers commands: the router writes a pickled list into one shared `ShmObjsIOBuffer` and calls `set_ready()` (see `_aborted_reqs()` at `router/manager.py:306`). Every rank's infer loop polls it via `_try_read_new_reqs()` and dispatches object types in `_read_reqs_buffer_and_init_reqs()` (`base_backend.py:419`). `AbortedReqCmd` / `StopStrMatchedReqCmd` (`core/objs/io_objs/group_req.py:31`) are the precedent.
- Each worker process runs **two** `infer_loop` threads (`base_backend.py:245-248`) for overlap mode, but an event "baton" (`event_pack.wait_to_forward()` at the top of `infer_loop`) serializes their CPU launch sections: all CUDA kernel launches of a step happen inside the stream block *before* `notify_forward_and_wait_post_handle()` passes the baton. Therefore a hook placed at the top of the forward branch runs while no other thread is launching kernels, and `torch.cuda.synchronize()` there drains all in-flight work. This is why the profiler state machine needs no extra cross-thread coordination beyond a defensive lock.
- A "profiling step" is an actual model forward batch — NOT a loop iteration (idle iterations run `run_way.is_pass()` and must not count).
- Lint: `black --line-length=120`, flake8 via `pre-commit run --files <files>`. Comments in Chinese or English are both fine (codebase mixes them).
- Unit tests live in `unit_tests/` mirroring the package tree; shm-based tests pass an explicit shm `name` to avoid needing `LIGHTLLM_UNIQUE_SERVICE_NAME_ID` env (see `unit_tests/server/core/objs/test_shm_array.py` for the style).
- `create_or_link_shm(name, expected_size)` (`lightllm/utils/shm_utils.py:9`) creates-or-links POSIX shm; new segments are zero-filled.

---

### Task 1: `--enable_profiling` launch flag

**Files:**
- Modify: `lightllm/server/api_cli.py` (after the `--disable_log_stats` argument, around line 272)
- Modify: `lightllm/server/core/objs/start_args_type.py` (near `disable_cudagraph`, line 124)

- [ ] **Step 1: Add the CLI argument**

In `lightllm/server/api_cli.py`, directly after the `--disable_log_stats` line (`parser.add_argument("--disable_log_stats", ...)`, line ~272), insert:

```python
    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        help="""enable the /start_profile /stop_profile /profile_status http endpoints for on-demand
        torch.profiler capture. trace output is restricted to subdirs of LIGHTLLM_TORCH_PROFILER_DIR
        (default /tmp/lightllm_profile).""",
    )
```

- [ ] **Step 2: Add the StartArgs field**

In `lightllm/server/core/objs/start_args_type.py`, directly after `disable_cudagraph: bool = field(default=False)` (line 124), insert:

```python
    enable_profiling: bool = field(default=False)
```

- [ ] **Step 3: Verify**

Run:
```bash
cd /mtc/sufubao/shared_home/sufubao/code/worktree-lightllm/profile_server
python -c "from lightllm.server.core.objs.start_args_type import StartArgs; assert StartArgs().enable_profiling is False; print('ok')"
python -m lightllm.server.api_server --help 2>&1 | grep -A1 enable_profiling
```
Expected: `ok`, and the help text shows `--enable_profiling`.

- [ ] **Step 4: Lint and commit**

```bash
pre-commit run --files lightllm/server/api_cli.py lightllm/server/core/objs/start_args_type.py
git add lightllm/server/api_cli.py lightllm/server/core/objs/start_args_type.py
git commit -m "feat(profiling): add --enable_profiling launch flag"
```

---

### Task 2: Profile command dataclasses

**Files:**
- Create: `lightllm/server/core/objs/io_objs/profile_cmd.py`
- Modify: `lightllm/server/core/objs/io_objs/__init__.py`
- Test: `unit_tests/server/core/objs/test_profile_cmd.py`

- [ ] **Step 1: Write the failing test**

Create `unit_tests/server/core/objs/test_profile_cmd.py`:

```python
import pickle
from lightllm.server.core.objs.io_objs import ProfileControlReq, StartProfileCmd, StopProfileCmd


def test_pickle_roundtrip():
    req = ProfileControlReq(action="start", profile_id=123, output_dir="/tmp/x", num_steps=8)
    restored = pickle.loads(pickle.dumps(req))
    assert restored == req

    cmd = StartProfileCmd(profile_id=123, output_dir="/tmp/x")
    assert pickle.loads(pickle.dumps(cmd)) == cmd

    stop = StopProfileCmd(profile_id=123)
    assert pickle.loads(pickle.dumps(stop)) == stop


def test_start_req_to_worker_cmd():
    req = ProfileControlReq(
        action="start",
        profile_id=42,
        output_dir="/tmp/traces",
        num_steps=10,
        start_step=5,
        activities=["CPU"],
        with_stack=False,
        record_shapes=True,
        profile_prefix="bench",
    )
    cmd = req.to_worker_cmd()
    assert isinstance(cmd, StartProfileCmd)
    assert cmd.profile_id == 42
    assert cmd.output_dir == "/tmp/traces"
    assert cmd.num_steps == 10
    assert cmd.start_step == 5
    assert cmd.activities == ["CPU"]
    assert cmd.with_stack is False
    assert cmd.record_shapes is True
    assert cmd.profile_prefix == "bench"


def test_stop_req_to_worker_cmd():
    req = ProfileControlReq(action="stop", profile_id=42)
    cmd = req.to_worker_cmd()
    assert isinstance(cmd, StopProfileCmd)
    assert cmd.profile_id == 42


def test_defaults():
    req = ProfileControlReq(action="start", profile_id=1)
    assert req.targets == ["worker"]
    assert req.activities == ["CPU", "GPU"]
    assert req.with_stack is True
    assert req.record_shapes is False
    assert req.num_steps is None
    assert req.start_step is None
    assert req.profile_prefix == "lightllm"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest unit_tests/server/core/objs/test_profile_cmd.py -v`
Expected: FAIL with `ImportError: cannot import name 'ProfileControlReq'`

- [ ] **Step 3: Implement the dataclasses**

Create `lightllm/server/core/objs/io_objs/profile_cmd.py`:

```python
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StartProfileCmd:
    profile_id: int
    output_dir: str
    num_steps: Optional[int] = None
    start_step: Optional[int] = None
    activities: List[str] = field(default_factory=lambda: ["CPU", "GPU"])
    with_stack: bool = True
    record_shapes: bool = False
    profile_prefix: str = "lightllm"


@dataclass
class StopProfileCmd:
    profile_id: int = 0


@dataclass
class ProfileControlReq:
    """httpserver -> router 的 profile 控制消息, router 转换为 worker cmd 后经 ShmObjsIOBuffer 广播。"""

    action: str  # "start" or "stop"
    profile_id: int
    targets: List[str] = field(default_factory=lambda: ["worker"])
    output_dir: str = ""
    num_steps: Optional[int] = None
    start_step: Optional[int] = None
    activities: List[str] = field(default_factory=lambda: ["CPU", "GPU"])
    with_stack: bool = True
    record_shapes: bool = False
    profile_prefix: str = "lightllm"

    def to_worker_cmd(self):
        if self.action == "start":
            return StartProfileCmd(
                profile_id=self.profile_id,
                output_dir=self.output_dir,
                num_steps=self.num_steps,
                start_step=self.start_step,
                activities=self.activities,
                with_stack=self.with_stack,
                record_shapes=self.record_shapes,
                profile_prefix=self.profile_prefix,
            )
        return StopProfileCmd(profile_id=self.profile_id)
```

Replace the contents of `lightllm/server/core/objs/io_objs/__init__.py` with:

```python
from .group_req import GroupReqIndexes, GroupReqObjs, AbortedReqCmd, StopStrMatchedReqCmd
from .profile_cmd import ProfileControlReq, StartProfileCmd, StopProfileCmd
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest unit_tests/server/core/objs/test_profile_cmd.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
pre-commit run --files lightllm/server/core/objs/io_objs/profile_cmd.py lightllm/server/core/objs/io_objs/__init__.py unit_tests/server/core/objs/test_profile_cmd.py
git add lightllm/server/core/objs/io_objs/profile_cmd.py lightllm/server/core/objs/io_objs/__init__.py unit_tests/server/core/objs/test_profile_cmd.py
git commit -m "feat(profiling): add profile control message dataclasses"
```

---

### Task 3: ProfileStatusBoard (per-rank shm status table)

**Files:**
- Create: `lightllm/server/core/objs/profile_status_board.py`
- Test: `unit_tests/server/core/objs/test_profile_status_board.py`

Why a table: a single status scalar cannot represent partial failure (rank 2 errored while ranks 0-1 record) and would be racily overwritten by multiple writer processes. One slot per worker rank + one router slot; each writer owns exactly one slot; the HTTP process only reads.

- [ ] **Step 1: Write the failing test**

Create `unit_tests/server/core/objs/test_profile_status_board.py`:

```python
from lightllm.server.core.objs.profile_status_board import (
    ProfileStatusBoard,
    STATE_IDLE,
    STATE_RUNNING,
    STATE_ERROR,
    ERROR_NONE,
    ERROR_START_FAILED,
)


def test_set_and_get_slot():
    board = ProfileStatusBoard(num_worker_slots=4, name="test_profile_status_board_a")
    board.set_slot(0, state=STATE_RUNNING, profile_id=99, forward_ct=7, target_ct=10, error_code=ERROR_NONE)
    slot = board.get_slot(0)
    assert slot == {"state": "running", "profile_id": 99, "forward_ct": 7, "target_ct": 10, "error_code": 0}


def test_partial_update_preserves_other_fields():
    board = ProfileStatusBoard(num_worker_slots=4, name="test_profile_status_board_b")
    board.set_slot(1, state=STATE_RUNNING, profile_id=5, forward_ct=1, target_ct=9, error_code=ERROR_NONE)
    board.set_slot(1, forward_ct=3)
    slot = board.get_slot(1)
    assert slot["forward_ct"] == 3
    assert slot["profile_id"] == 5
    assert slot["state"] == "running"


def test_router_slot_index():
    board = ProfileStatusBoard(num_worker_slots=4, name="test_profile_status_board_c")
    assert board.router_slot == 4
    board.set_slot(board.router_slot, state=STATE_ERROR, error_code=ERROR_START_FAILED)
    assert board.get_slot(board.router_slot)["state"] == "error"


def test_two_instances_share_memory():
    writer = ProfileStatusBoard(num_worker_slots=2, name="test_profile_status_board_d")
    reader = ProfileStatusBoard(num_worker_slots=2, name="test_profile_status_board_d")
    writer.set_slot(0, state=STATE_IDLE, profile_id=777)
    assert reader.get_slot(0)["profile_id"] == 777
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest unit_tests/server/core/objs/test_profile_status_board.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lightllm.server.core.objs.profile_status_board'`

- [ ] **Step 3: Implement the board**

Create `lightllm/server/core/objs/profile_status_board.py`:

```python
import numpy as np
from typing import Optional
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.shm_utils import create_or_link_shm

STATE_IDLE = 0
STATE_ARMED = 1
STATE_RUNNING = 2
STATE_FLUSHING = 3
STATE_ERROR = 4
STATE_NAMES = {0: "idle", 1: "armed", 2: "running", 3: "flushing", 4: "error"}

ERROR_NONE = 0
ERROR_START_FAILED = 1
ERROR_EXPORT_FAILED = 2
ERROR_CMD_DELIVERY_FAILED = 3

_FIELD_STATE = 0
_FIELD_PROFILE_ID = 1
_FIELD_FORWARD_CT = 2
_FIELD_TARGET_CT = 3
_FIELD_ERROR_CODE = 4
_NUM_FIELDS = 5


class ProfileStatusBoard:
    """
    profile 状态共享内存表: 每个 worker rank 一个 slot, 外加一个 router slot。
    每个 writer 进程只写自己的 slot, http 进程只读聚合, 所以无需加锁。
    """

    def __init__(self, num_worker_slots: int, name: Optional[str] = None):
        self.num_worker_slots = num_worker_slots
        self.num_slots = num_worker_slots + 1
        name = name if name is not None else f"{get_unique_server_name()}_profile_status_board"
        self.shm = create_or_link_shm(name, self.num_slots * _NUM_FIELDS * 8)
        self.arr = np.ndarray((self.num_slots, _NUM_FIELDS), dtype=np.int64, buffer=self.shm.buf)

    @property
    def router_slot(self) -> int:
        return self.num_worker_slots

    def set_slot(self, slot, state=None, profile_id=None, forward_ct=None, target_ct=None, error_code=None):
        for field_index, value in (
            (_FIELD_STATE, state),
            (_FIELD_PROFILE_ID, profile_id),
            (_FIELD_FORWARD_CT, forward_ct),
            (_FIELD_TARGET_CT, target_ct),
            (_FIELD_ERROR_CODE, error_code),
        ):
            if value is not None:
                self.arr[slot, field_index] = value
        return

    def get_slot(self, slot) -> dict:
        row = self.arr[slot]
        return {
            "state": STATE_NAMES.get(int(row[_FIELD_STATE]), "unknown"),
            "profile_id": int(row[_FIELD_PROFILE_ID]),
            "forward_ct": int(row[_FIELD_FORWARD_CT]),
            "target_ct": int(row[_FIELD_TARGET_CT]),
            "error_code": int(row[_FIELD_ERROR_CODE]),
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest unit_tests/server/core/objs/test_profile_status_board.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
pre-commit run --files lightllm/server/core/objs/profile_status_board.py unit_tests/server/core/objs/test_profile_status_board.py
git add lightllm/server/core/objs/profile_status_board.py unit_tests/server/core/objs/test_profile_status_board.py
git commit -m "feat(profiling): add per-rank shared-memory profile status board"
```

---

### Task 4: WorkerProfilerManager state machine

**Files:**
- Create: `lightllm/server/router/model_infer/mode_backend/profiler_manager.py`
- Test: `unit_tests/server/router/test_worker_profiler_manager.py`

Semantics (from the design doc):
- `on_cmd()` is called when the infer loop reads a `StartProfileCmd`/`StopProfileCmd` from the shm buffer — this only *arms* the state machine (or stops it).
- `on_step_boundary()` is called exactly once per real model forward, immediately **before** the forward is launched. It increments `forward_ct`, starts the profiler when armed-and-due, and stops/exports when `forward_ct` reaches the target — so a `num_steps=N` capture contains exactly N forwards.
- Both methods are only ever invoked by the baton-holding infer thread; the lock is defensive.
- Failures never leave the machine wedged: errors are written to the status board and the internal state returns to IDLE so a later attempt works.

- [ ] **Step 1: Write the failing test**

Create `unit_tests/server/router/test_worker_profiler_manager.py`:

```python
import gzip
import os
import pytest
from lightllm.server.core.objs.io_objs import StartProfileCmd, StopProfileCmd
from lightllm.server.core.objs.profile_status_board import (
    ProfileStatusBoard,
    ERROR_EXPORT_FAILED,
    ERROR_NONE,
    ERROR_START_FAILED,
)
from lightllm.server.router.model_infer.mode_backend.profiler_manager import WorkerProfilerManager


class FakeProfiler:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.exported_path = None

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def export_chrome_trace(self, path):
        self.exported_path = path
        with open(path, "w") as f:
            f.write("{}")


@pytest.fixture()
def board(request):
    return ProfileStatusBoard(num_worker_slots=2, name=f"test_wpm_board_{request.node.name}")


def make_manager(board, factory):
    return WorkerProfilerManager(
        rank_in_node=0, dp_rank_in_node=0, node_world_size=2, profiler_factory=factory, status_board=board
    )


def start_cmd(tmp_path, **kw):
    defaults = dict(profile_id=1, output_dir=str(tmp_path), activities=["CPU"], with_stack=False)
    defaults.update(kw)
    return StartProfileCmd(**defaults)


def test_auto_stop_after_num_steps(tmp_path, board):
    fake = FakeProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_cmd(start_cmd(tmp_path, num_steps=3))
    assert board.get_slot(0)["state"] == "armed"

    mgr.on_step_boundary()  # forward 1: starts capture
    assert fake.started and not fake.stopped
    assert board.get_slot(0)["state"] == "running"
    assert board.get_slot(0)["target_ct"] == 4

    mgr.on_step_boundary()  # forward 2
    mgr.on_step_boundary()  # forward 3
    assert not fake.stopped

    mgr.on_step_boundary()  # boundary of forward 4: stop BEFORE launching it
    assert fake.stopped
    trace_files = [f for f in os.listdir(tmp_path) if f.endswith(".trace.json.gz")]
    assert trace_files == ["lightllm-1-TP-0-DP-0.trace.json.gz"]
    with gzip.open(os.path.join(tmp_path, trace_files[0]), "rb") as f:
        assert f.read() == b"{}"
    assert board.get_slot(0)["state"] == "idle"
    assert board.get_slot(0)["error_code"] == ERROR_NONE


def test_start_step_delays_capture(tmp_path, board):
    fake = FakeProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_cmd(start_cmd(tmp_path, num_steps=2, start_step=3))
    mgr.on_step_boundary()  # forward 1
    mgr.on_step_boundary()  # forward 2
    assert not fake.started
    mgr.on_step_boundary()  # forward 3: starts here
    assert fake.started


def test_manual_stop(tmp_path, board):
    fake = FakeProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_cmd(start_cmd(tmp_path, num_steps=None))
    mgr.on_step_boundary()
    mgr.on_step_boundary()
    assert fake.started and not fake.stopped
    mgr.on_cmd(StopProfileCmd())
    assert fake.stopped
    assert board.get_slot(0)["state"] == "idle"


def test_stop_while_armed_cancels(tmp_path, board):
    fake = FakeProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_cmd(start_cmd(tmp_path, start_step=100))
    mgr.on_cmd(StopProfileCmd())
    mgr.on_step_boundary()
    assert not fake.started
    assert board.get_slot(0)["state"] == "idle"


def test_start_ignored_while_running(tmp_path, board):
    fake = FakeProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_cmd(start_cmd(tmp_path, profile_id=1))
    mgr.on_step_boundary()
    mgr.on_cmd(start_cmd(tmp_path, profile_id=2))
    assert board.get_slot(0)["profile_id"] == 1


def test_factory_failure_reports_error_and_recovers(tmp_path, board):
    def bad_factory(cmd):
        raise RuntimeError("boom")

    mgr = make_manager(board, bad_factory)
    mgr.on_cmd(start_cmd(tmp_path))
    mgr.on_step_boundary()
    assert board.get_slot(0)["state"] == "error"
    assert board.get_slot(0)["error_code"] == ERROR_START_FAILED

    # 故障后可恢复: 换一个正常 factory 的同一实例语义无法注入, 这里验证状态机回到可再次 arm 的状态
    mgr.on_cmd(start_cmd(tmp_path, profile_id=9))
    assert board.get_slot(0)["state"] == "armed"


def test_export_failure_reports_error(tmp_path, board):
    class BadExportProfiler(FakeProfiler):
        def export_chrome_trace(self, path):
            raise RuntimeError("export boom")

    fake = BadExportProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_cmd(start_cmd(tmp_path, num_steps=1))
    mgr.on_step_boundary()  # start
    mgr.on_step_boundary()  # stop -> export fails
    assert board.get_slot(0)["state"] == "error"
    assert board.get_slot(0)["error_code"] == ERROR_EXPORT_FAILED


def test_idle_fast_path_counts_forwards(tmp_path, board):
    fake = FakeProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_step_boundary()
    mgr.on_step_boundary()
    assert mgr.forward_ct == 2
    assert not fake.started
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest unit_tests/server/router/test_worker_profiler_manager.py -v`
Expected: FAIL with `ModuleNotFoundError` for `profiler_manager`

- [ ] **Step 3: Implement the manager**

Create `lightllm/server/router/model_infer/mode_backend/profiler_manager.py`:

```python
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
    return torch.profiler.profile(
        activities=activities,
        with_stack=cmd.with_stack,
        record_shapes=cmd.record_shapes,
    )


class WorkerProfilerManager:
    """
    每个推理 rank 进程一个实例, 状态机: IDLE -> ARMED -> RUNNING -> FLUSHING -> IDLE。
    一个 "step" 是一次真实的模型 forward batch, 不包含 infer_loop 的空转迭代。
    on_cmd 和 on_step_boundary 都只会被持有 overlap event 令牌的 infer_loop 线程调用
    (令牌串行化了两个线程的 launch 区段), 锁只是防御性的。
    停止时先 torch.cuda.synchronize() 排空两个线程已发射的全部 GPU 工作, 再 stop/export,
    保证捕获窗口正好覆盖 num_steps 个完整 forward。
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
                if self._state == STATE_RUNNING:
                    self._stop_and_export()
                elif self._state == STATE_ARMED:
                    self._state = STATE_IDLE
                    self._cmd = None
                    self.status_board.set_slot(self._slot, state=STATE_IDLE)
        return

    def on_step_boundary(self):
        # 未开启 profiling 时的快路径, 只有一次整型比较的开销。
        if self._state == STATE_IDLE:
            self.forward_ct += 1
            return
        with self._lock:
            self.forward_ct += 1
            if self._state == STATE_RUNNING:
                if self._target_ct is not None and self.forward_ct >= self._target_ct:
                    # 在本次 forward 发射之前停止, 捕获窗口正好是 num_steps 个 forward。
                    self._stop_and_export()
                else:
                    self.status_board.set_slot(self._slot, forward_ct=self.forward_ct)
            elif self._state == STATE_ARMED and self.forward_ct >= self._start_at_ct:
                self._do_start()
        return

    def _do_start(self):
        try:
            self._profiler = self._profiler_factory(self._cmd)
            self._profiler.start()
            self._target_ct = self.forward_ct + self._cmd.num_steps if self._cmd.num_steps is not None else None
            self._state = STATE_RUNNING
            self.status_board.set_slot(
                self._slot, state=STATE_RUNNING, forward_ct=self.forward_ct, target_ct=self._target_ct or 0
            )
            logger.info(f"profiler started at forward_ct {self.forward_ct}, target_ct {self._target_ct}")
        except BaseException as e:
            logger.exception(f"profiler start failed: {e}")
            self._profiler = None
            self._cmd = None
            self._state = STATE_IDLE
            self.status_board.set_slot(self._slot, state=STATE_ERROR, error_code=ERROR_START_FAILED)
        return

    def _stop_and_export(self):
        self.status_board.set_slot(self._slot, state=STATE_FLUSHING, forward_ct=self.forward_ct)
        cmd = self._cmd
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._profiler.stop()
            os.makedirs(cmd.output_dir, exist_ok=True)
            trace_name = f"{cmd.profile_prefix}-{cmd.profile_id}-TP-{self.rank_in_node}-DP-{self.dp_rank_in_node}"
            tmp_path = os.path.join(cmd.output_dir, trace_name + ".trace.json")
            self._profiler.export_chrome_trace(tmp_path)
            final_path = tmp_path + ".gz"
            with open(tmp_path, "rb") as f_in, gzip.open(final_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(tmp_path)
            self.status_board.set_slot(self._slot, state=STATE_IDLE, error_code=ERROR_NONE)
            logger.info(f"profiler trace exported to {final_path}")
        except BaseException as e:
            logger.exception(f"profiler stop/export failed: {e}")
            self.status_board.set_slot(self._slot, state=STATE_ERROR, error_code=ERROR_EXPORT_FAILED)
        finally:
            self._profiler = None
            self._cmd = None
            self._target_ct = None
            self._state = STATE_IDLE
        return
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest unit_tests/server/router/test_worker_profiler_manager.py -v`
Expected: 8 passed

- [ ] **Step 5: Lint and commit**

```bash
pre-commit run --files lightllm/server/router/model_infer/mode_backend/profiler_manager.py unit_tests/server/router/test_worker_profiler_manager.py
git add lightllm/server/router/model_infer/mode_backend/profiler_manager.py unit_tests/server/router/test_worker_profiler_manager.py
git commit -m "feat(profiling): add worker profiler manager state machine"
```

---

### Task 5: Wire the manager into the worker backends

**Files:**
- Modify: `lightllm/server/router/model_infer/mode_backend/base_backend.py` (imports line 24; init_model after line 234; `_read_reqs_buffer_and_init_reqs` line 419)
- Modify: `lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py` (`infer_loop`, lines 73-90)
- Modify: `lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py` (`infer_loop`, run_way branches around line 116)

- [ ] **Step 1: base_backend — imports**

In `base_backend.py`, change line 24 from:

```python
from lightllm.server.core.objs.io_objs import AbortedReqCmd, StopStrMatchedReqCmd
```

to:

```python
from lightllm.server.core.objs.io_objs import AbortedReqCmd, StopStrMatchedReqCmd, StartProfileCmd, StopProfileCmd
```

and after line 51 (`from .multi_level_kv_cache import MultiLevelKvCacheModule`) add:

```python
from .profiler_manager import WorkerProfilerManager
```

- [ ] **Step 2: base_backend — instantiate the manager**

In `init_model`, directly after `self.shm_pd_trans_io_buffer = ShmObjsIOBuffer(tail_str="pd")` (line ~234), insert:

```python
        # profile 状态机, 由 /start_profile http 接口经 router 广播的 cmd 驱动。
        self.profiler_manager = WorkerProfilerManager(
            rank_in_node=self.rank_in_node,
            dp_rank_in_node=self.dp_rank_in_node,
            node_world_size=self.node_world_size,
        )
```

(`init_rank_infos()` ran earlier in `init_model`, so the rank attributes exist.)

- [ ] **Step 3: base_backend — dispatch the cmds**

In `_read_reqs_buffer_and_init_reqs` (line ~419), the dispatch currently reads:

```python
                elif isinstance(obj, (AbortedReqCmd, StopStrMatchedReqCmd)):
                    if obj.req_id in g_infer_context.requests_mapping:
                        req: InferReq = g_infer_context.requests_mapping[obj.req_id]
                        req.infer_aborted = True
                else:
                    assert False, f"error type {type(obj)}"
```

Add a branch so it becomes:

```python
                elif isinstance(obj, (AbortedReqCmd, StopStrMatchedReqCmd)):
                    if obj.req_id in g_infer_context.requests_mapping:
                        req: InferReq = g_infer_context.requests_mapping[obj.req_id]
                        req.infer_aborted = True
                elif isinstance(obj, (StartProfileCmd, StopProfileCmd)):
                    self.profiler_manager.on_cmd(obj)
                else:
                    assert False, f"error type {type(obj)}"
```

- [ ] **Step 4: chunked_prefill infer_loop — step-boundary hooks**

In `chunked_prefill/impl.py` `infer_loop`, add `self.profiler_manager.on_step_boundary()` as the first statement of the `is_prefill()` and `is_decode()` branches (NOT the `is_pass()` branch — idle iterations are not steps):

```python
                if run_way.is_prefill():
                    self.profiler_manager.on_step_boundary()
                    # 进行一次流同步，保证 _try_read_new_reqs 中的一些算子操作，必然已经完成。
                    # 防止后续的推理流程读取到显存中可能存在错误的数据。
                    g_infer_context.get_overlap_stream().wait_stream(torch.cuda.current_stream())
                    self.prefill(
                        event_pack=event_pack,
                        prefill_reqs=prefill_reqs,
                    )
                    continue
                elif run_way.is_decode():
                    self.profiler_manager.on_step_boundary()
                    # 进行一次流同步，保证 _try_read_new_reqs 中的一些算子操作，必然已经完成。
                    # 防止后续的推理流程读取到显存中可能存在错误的数据。
                    g_infer_context.get_overlap_stream().wait_stream(torch.cuda.current_stream())
                    self.decode(
                        event_pack=event_pack,
                        decode_reqs=decode_reqs,
                    )
                    continue
```

- [ ] **Step 5: dp_backend infer_loop — same hooks**

In `dp_backend/impl.py` `infer_loop` (run_way branching starts ~line 116), insert `self.profiler_manager.on_step_boundary()` as the first statement inside the `run_way.is_prefill()` branch and the `run_way.is_decode()` branch, exactly as in Step 4. (In DP overlap mode one loop iteration may launch two microbatches; that iteration counts as ONE step — acceptable and documented in the design doc.)

- [ ] **Step 6: Verify imports and existing tests**

Run:
```bash
python -c "from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import ChunkedPrefillBackend; print('ok')"
python -c "from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend; print('ok')" 2>/dev/null || python -c "import lightllm.server.router.model_infer.mode_backend.dp_backend.impl as m; print('ok')"
pytest unit_tests/server/router/test_worker_profiler_manager.py unit_tests/server/core/objs/test_profile_cmd.py -q
```
Expected: both `ok`, tests pass.

- [ ] **Step 7: Lint and commit**

```bash
pre-commit run --files lightllm/server/router/model_infer/mode_backend/base_backend.py lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py lightllm/server/router/model_infer/mode_backend/dp_backend/impl.py
git add -u
git commit -m "feat(profiling): wire profiler manager into worker infer loops"
```

---

### Task 6: Router — dispatch ProfileControlReq and broadcast worker cmds

**Files:**
- Modify: `lightllm/server/router/manager.py` (imports lines 19-23; `__init__` after line 97; `_recv_new_reqs_and_schedule` line 490; new method after `_stop_str_matched_reqs` line 320)

- [ ] **Step 1: Extend imports**

In `router/manager.py`, change the io_objs import block (lines 19-23) to:

```python
from lightllm.server.core.objs.io_objs import (
    GroupReqIndexes,
    AbortedReqCmd,
    StopStrMatchedReqCmd,
    ProfileControlReq,
)
```

and after the `ShmObjsIOBuffer` import (line 27) add:

```python
from lightllm.server.core.objs.profile_status_board import (
    ProfileStatusBoard,
    STATE_ERROR,
    STATE_IDLE,
    ERROR_CMD_DELIVERY_FAILED,
    ERROR_NONE,
)
```

- [ ] **Step 2: Create the status board in `__init__`**

After `self.shm_reqs_io_buffer = ShmObjsIOBuffer()` (line 97) add:

```python
        self.profile_status_board = ProfileStatusBoard(num_worker_slots=self.node_world_size)
```

- [ ] **Step 3: Dispatch in the recv loop**

In `_recv_new_reqs_and_schedule` (line ~496), replace:

```python
            for _ in range(self.recv_max_count):
                recv_req: GroupReqIndexes = self.zmq_recv_socket.recv_pyobj(zmq.NOBLOCK)
                if isinstance(recv_req, GroupReqIndexes):
                    self._add_req(recv_req)
                else:
                    assert False, f"Error Req Inf {recv_req}"
```

with:

```python
            for _ in range(self.recv_max_count):
                recv_req: GroupReqIndexes = self.zmq_recv_socket.recv_pyobj(zmq.NOBLOCK)
                if isinstance(recv_req, GroupReqIndexes):
                    self._add_req(recv_req)
                elif isinstance(recv_req, ProfileControlReq):
                    await self._handle_profile_control_req(recv_req)
                else:
                    assert False, f"Error Req Inf {recv_req}"
```

- [ ] **Step 4: Add the handler method**

After `_stop_str_matched_reqs` (line ~320) add:

```python
    async def _handle_profile_control_req(self, profile_req: ProfileControlReq):
        if "worker" not in profile_req.targets:
            return
        # 有界等待: profile cmd 不能让 router 主循环无限自旋 (buffer 可能被卡住的 rank 占住)。
        for _ in range(5000):
            if self.shm_reqs_io_buffer.is_empty():
                break
            await asyncio.sleep(0.001)
        else:
            logger.error(
                f"profile cmd '{profile_req.action}' (id={profile_req.profile_id}) dropped: "
                f"shm_reqs_io_buffer busy for 5s"
            )
            self.profile_status_board.set_slot(
                self.profile_status_board.router_slot,
                state=STATE_ERROR,
                profile_id=profile_req.profile_id,
                error_code=ERROR_CMD_DELIVERY_FAILED,
            )
            return
        self.shm_reqs_io_buffer.write_obj([profile_req.to_worker_cmd()])
        self.shm_reqs_io_buffer.set_ready()
        self.profile_status_board.set_slot(
            self.profile_status_board.router_slot,
            state=STATE_IDLE,
            profile_id=profile_req.profile_id,
            error_code=ERROR_NONE,
        )
        return
```

- [ ] **Step 5: Verify import and tests**

Run:
```bash
python -c "import lightllm.server.router.manager; print('ok')"
pytest unit_tests/server/core/objs/test_profile_status_board.py -q
```
Expected: `ok`, tests pass.

- [ ] **Step 6: Lint and commit**

```bash
pre-commit run --files lightllm/server/router/manager.py
git add lightllm/server/router/manager.py
git commit -m "feat(profiling): route profile control reqs through router to workers"
```

---

### Task 7: HTTP endpoints + HttpServerManager plumbing

**Files:**
- Modify: `lightllm/server/httpserver/manager.py` (imports line 28; `__init__` after line 99; new method near `transfer_to_next_module` line 626)
- Modify: `lightllm/server/api_http.py` (imports; new endpoints after `/token_load`, line 229)

- [ ] **Step 1: HttpServerManager — imports and status board**

In `httpserver/manager.py`, change line 28 from:

```python
from lightllm.server.core.objs.io_objs import GroupReqObjs
```

to:

```python
from lightllm.server.core.objs.io_objs import GroupReqObjs, ProfileControlReq
```

and after it add:

```python
from lightllm.server.core.objs.profile_status_board import ProfileStatusBoard
```

In `__init__`, after `self.shm_req_manager = ShmReqManager()` (line 99), add:

```python
        if args.enable_profiling:
            self.profile_status_board = ProfileStatusBoard(num_worker_slots=args.tp // args.nnodes)
        else:
            self.profile_status_board = None
```

- [ ] **Step 2: HttpServerManager — send method**

Before `async def transfer_to_next_module(` (line ~626) add:

```python
    async def send_profile_control(self, profile_req: ProfileControlReq):
        self.send_to_router.send_pyobj(profile_req, protocol=pickle.HIGHEST_PROTOCOL)
        return
```

(`pickle` is already imported at the top of the file.)

- [ ] **Step 3: api_http — imports and constants**

In `api_http.py`, add to the existing import section:

```python
from lightllm.server.core.objs.io_objs import ProfileControlReq
```

(`os`, `time`, `HTTPStatus`, `JSONResponse`, `Request`, `Response` are already imported.) After the import block, near other module-level constants, add:

```python
LIGHTLLM_PROFILE_DIR_ROOT = os.getenv("LIGHTLLM_TORCH_PROFILER_DIR", "/tmp/lightllm_profile")
_PROFILE_ALLOWED_ACTIVITIES = {"CPU", "GPU"}
```

- [ ] **Step 4: api_http — endpoints**

Insert after the `/token_load` endpoint (line ~229), before `/generate`:

```python
def _check_profiling_enabled():
    if not g_objs.args.enable_profiling:
        return create_error_response(
            HTTPStatus.NOT_IMPLEMENTED, "profiling is not enabled, launch the server with --enable_profiling"
        )
    return None


@app.post("/start_profile", summary="Arm an on-demand torch.profiler capture on all worker ranks")
async def start_profile(request: Request) -> Response:
    error = _check_profiling_enabled()
    if error is not None:
        return error
    try:
        body = await request.json()
    except Exception:
        body = {}

    targets = body.get("targets", ["worker"])
    if any(target != "worker" for target in targets):
        return create_error_response(HTTPStatus.NOT_IMPLEMENTED, "only the 'worker' target is supported")

    activities = body.get("activities", ["CPU", "GPU"])
    if not set(activities).issubset(_PROFILE_ALLOWED_ACTIVITIES):
        return create_error_response(
            HTTPStatus.BAD_REQUEST, f"activities must be a subset of {sorted(_PROFILE_ALLOWED_ACTIVITIES)}"
        )

    root_dir = os.path.realpath(LIGHTLLM_PROFILE_DIR_ROOT)
    output_dir = os.path.realpath(body.get("output_dir") or root_dir)
    if output_dir != root_dir and not output_dir.startswith(root_dir + os.sep):
        return create_error_response(HTTPStatus.BAD_REQUEST, f"output_dir must be under {root_dir}")

    profile_id = int(time.time() * 1000)
    profile_req = ProfileControlReq(
        action="start",
        profile_id=profile_id,
        targets=targets,
        output_dir=output_dir,
        num_steps=body.get("num_steps"),
        start_step=body.get("start_step"),
        activities=activities,
        with_stack=bool(body.get("with_stack", True)),
        record_shapes=bool(body.get("record_shapes", False)),
        profile_prefix=str(body.get("profile_prefix", "lightllm")),
    )
    await g_objs.httpserver_manager.send_profile_control(profile_req)
    # 202: 仅代表命令已入队, worker 实际状态请轮询 /profile_status。
    return JSONResponse(
        {"status": "accepted", "profile_id": profile_id, "output_dir": output_dir}, status_code=202
    )


@app.post("/stop_profile", summary="Stop a running capture and flush traces")
async def stop_profile(request: Request) -> Response:
    error = _check_profiling_enabled()
    if error is not None:
        return error
    profile_req = ProfileControlReq(action="stop", profile_id=0)
    await g_objs.httpserver_manager.send_profile_control(profile_req)
    return JSONResponse({"status": "accepted"}, status_code=202)


@app.get("/profile_status", summary="Per-rank profiler state")
async def profile_status(request: Request) -> Response:
    error = _check_profiling_enabled()
    if error is not None:
        return error
    board = g_objs.httpserver_manager.profile_status_board
    return JSONResponse(
        {
            "workers": [board.get_slot(slot) for slot in range(board.num_worker_slots)],
            "router": board.get_slot(board.router_slot),
        },
        status_code=200,
    )
```

- [ ] **Step 5: Verify imports**

Run:
```bash
python -c "import lightllm.server.api_http; print('ok')"
python -c "import lightllm.server.httpserver.manager; print('ok')"
```
Expected: `ok` twice.

- [ ] **Step 6: Lint and commit**

```bash
pre-commit run --files lightllm/server/api_http.py lightllm/server/httpserver/manager.py
git add lightllm/server/api_http.py lightllm/server/httpserver/manager.py
git commit -m "feat(profiling): add /start_profile /stop_profile /profile_status endpoints"
```

---

### Task 8: End-to-end verification + usage doc

**Files:**
- Create: `docs/profile_server/usage.md`
- No code changes; this validates the whole chain against a real server.

- [ ] **Step 1: Launch a real server with profiling enabled**

Use the `running-lightllm-with-docker` project skill to launch a server (any small model, e.g. Qwen3-8B, TP=1 is enough), adding `--enable_profiling` to the launch args and exporting `LIGHTLLM_TORCH_PROFILER_DIR=/tmp/lightllm_profile` into the server environment. Wait for readiness per the skill.

- [ ] **Step 2: Negative check — gate**

Against a server launched WITHOUT the flag (or before adding it):
```bash
curl -s -X POST http://127.0.0.1:8000/start_profile | head -c 200
```
Expected: error JSON with code 501 mentioning `--enable_profiling`.

- [ ] **Step 3: Arm a capture, drive traffic, poll status**

```bash
curl -s -X POST http://127.0.0.1:8000/start_profile -H "Content-Type: application/json" \
  -d '{"num_steps": 8, "activities": ["CPU", "GPU"], "with_stack": false}'
# expected: {"status": "accepted", "profile_id": <id>, "output_dir": "/tmp/lightllm_profile"}

curl -s http://127.0.0.1:8000/profile_status
# expected: workers state "armed" (no traffic yet)

curl -s -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
  -d '{"inputs": "What is AI?", "parameters": {"max_new_tokens": 32}}' > /dev/null

curl -s http://127.0.0.1:8000/profile_status
# expected: workers cycle armed -> running (forward_ct/target_ct advancing) -> flushing -> idle
```

- [ ] **Step 4: Validate the trace files**

```bash
ls /tmp/lightllm_profile/
# expected: lightllm-<profile_id>-TP-0-DP-0.trace.json.gz (one per rank)
gunzip -t /tmp/lightllm_profile/lightllm-*-TP-0-DP-0.trace.json.gz && echo "gzip ok"
python -c "import gzip, json, glob; p=glob.glob('/tmp/lightllm_profile/*.trace.json.gz')[0]; json.load(gzip.open(p)); print('valid chrome trace json')"
```
Expected: `gzip ok` and `valid chrome trace json`. Optionally load the file at https://ui.perfetto.dev to eyeball it.

- [ ] **Step 5: Manual stop path**

```bash
curl -s -X POST http://127.0.0.1:8000/start_profile -d '{}'          # no num_steps
curl -s -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
  -d '{"inputs": "Hello", "parameters": {"max_new_tokens": 16}}' > /dev/null
curl -s -X POST http://127.0.0.1:8000/stop_profile
sleep 2 && curl -s http://127.0.0.1:8000/profile_status               # workers back to "idle"
ls /tmp/lightllm_profile/                                             # second trace present
```

- [ ] **Step 6: Bad-input checks**

```bash
curl -s -X POST http://127.0.0.1:8000/start_profile -d '{"output_dir": "/etc"}' | head -c 200
# expected: 400, "output_dir must be under /tmp/lightllm_profile"
curl -s -X POST http://127.0.0.1:8000/start_profile -d '{"activities": ["MEM"]}' | head -c 200
# expected: 400, activities subset error
curl -s -X POST http://127.0.0.1:8000/start_profile -d '{"targets": ["router"]}' | head -c 200
# expected: 501, only 'worker' target supported
```

- [ ] **Step 7: Write the usage doc**

Create `docs/profile_server/usage.md`:

```markdown
# On-demand profiling (`/start_profile`)

Launch the server with `--enable_profiling`. Traces are written under
`LIGHTLLM_TORCH_PROFILER_DIR` (default `/tmp/lightllm_profile`), one gzipped
chrome trace per worker rank: `{prefix}-{profile_id}-TP-{t}-DP-{d}.trace.json.gz`.
View them at https://ui.perfetto.dev.

## Capture a fixed window (recommended)

    curl -X POST http://HOST:PORT/start_profile -H "Content-Type: application/json" \
      -d '{"num_steps": 16, "activities": ["CPU", "GPU"], "with_stack": true}'

`num_steps` counts real model forward batches (idle scheduler iterations do not
count). The capture starts at the next forward after the command reaches the
workers and auto-stops after exactly `num_steps` forwards. `start_step` arms the
capture to begin at a future absolute forward count (skip warmup).

## Manual stop

Omit `num_steps`, then `POST /stop_profile`. Flushing happens in the background.

## Status

`GET /profile_status` returns per-rank `{state, profile_id, forward_ct,
target_ct, error_code}`. States: idle / armed / running / flushing / error.
Error codes: 1 = profiler start failed, 2 = trace export failed,
3 = router could not deliver the command (shm buffer busy).
A 202 from start/stop only means the command was queued — poll this endpoint.

## Caveats

- Decode runs inside CUDA graphs: kernels appear in traces but lose Python-op
  correlation. For kernel→source attribution relaunch with `--disable_cudagraph`.
- `with_stack: true` makes traces large; keep `num_steps` ≤ ~20 under load.
- Multi-node: each node's HTTP port profiles that node's ranks only.
- Covered backends: chunked_prefill (default) and dp_backend infer loops.
```

- [ ] **Step 8: Commit and final check**

```bash
pre-commit run --all-files
pytest unit_tests/server/core/objs/test_profile_cmd.py unit_tests/server/core/objs/test_profile_status_board.py unit_tests/server/router/test_worker_profiler_manager.py -q
git add docs/profile_server/usage.md
git commit -m "docs(profiling): add usage guide for on-demand profiling endpoints"
```

---

## Out of scope (later phases, per design doc)

NVTX step-phase ranges, `CUDA_PROFILER`/nsys capture window, `MEM` snapshots, router/httpserver CPU profiling (`targets` other than `worker`), `profile_by_stage`, trace merging, detokenization target, benchmark-client `--profile` flag, PD-mode backends (`continues_batch/pd_mode`, `pd_nixl`) and other backends that override `infer_loop`.
