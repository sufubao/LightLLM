import gzip
import os
import threading
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
    board = ProfileStatusBoard(num_worker_slots=2, name=f"test_wpm_board_{request.node.name}")
    board.arr[:] = 0  # 清掉可能残留的上次运行数据
    yield board
    board.shm.close()
    try:
        board.shm.unlink()
    except FileNotFoundError:
        pass


def make_manager(board, factory):
    return WorkerProfilerManager(
        rank_in_node=0, dp_rank_in_node=0, node_world_size=2, profiler_factory=factory, status_board=board
    )


def start_cmd(tmp_path, **kw):
    defaults = dict(profile_id=1, output_dir=str(tmp_path), activities=["CPU"], with_stack=False)
    defaults.update(kw)
    return StartProfileCmd(**defaults)


def run_in_thread(fn):
    t = threading.Thread(target=fn)
    t.start()
    t.join()


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

    # 故障后状态机必须可恢复: 还能再次 arm
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


def test_stop_while_idle_is_noop(tmp_path, board):
    fake = FakeProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_cmd(StopProfileCmd())
    assert not fake.stopped
    assert board.get_slot(0)["state"] == "idle"


def test_export_failure_leaves_no_partial_files(tmp_path, board):
    class BadExportProfiler(FakeProfiler):
        def export_chrome_trace(self, path):
            with open(path, "w") as f:
                f.write("partial")
            raise RuntimeError("export boom")

    fake = BadExportProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_cmd(start_cmd(tmp_path, num_steps=1))
    mgr.on_step_boundary()
    mgr.on_step_boundary()
    assert board.get_slot(0)["error_code"] == ERROR_EXPORT_FAILED
    assert os.listdir(tmp_path) == []


def test_cross_thread_stop_deferred_to_owner(tmp_path, board):
    fake = FakeProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_cmd(start_cmd(tmp_path, num_steps=None))
    mgr.on_step_boundary()  # owner = 当前线程
    run_in_thread(lambda: mgr.on_cmd(StopProfileCmd()))
    assert not fake.stopped  # 非 owner 线程不能 stop, 只做标记
    assert board.get_slot(0)["state"] == "running"
    mgr.on_step_boundary()  # owner 线程下一个 boundary 真正 stop
    assert fake.stopped
    assert board.get_slot(0)["state"] == "idle"


def test_pass_boundary_flushes_pending_stop(tmp_path, board):
    fake = FakeProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_cmd(start_cmd(tmp_path, num_steps=None))
    mgr.on_step_boundary()
    run_in_thread(lambda: mgr.on_cmd(StopProfileCmd()))
    assert not fake.stopped
    mgr.on_pass_boundary()  # owner 线程空转时补执行 stop
    assert fake.stopped


def test_target_reached_on_non_owner_thread_defers(tmp_path, board):
    fake = FakeProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_cmd(start_cmd(tmp_path, num_steps=2))
    mgr.on_step_boundary()  # forward 1, owner = 当前线程
    mgr.on_step_boundary()  # forward 2
    run_in_thread(mgr.on_step_boundary)  # forward 3 的 boundary 在另一线程命中 target
    assert not fake.stopped
    mgr.on_step_boundary()  # owner 线程 boundary: 真正 stop (窗口多了 1 个 forward)
    assert fake.stopped


def test_idle_fast_path_counts_forwards(tmp_path, board):
    fake = FakeProfiler()
    mgr = make_manager(board, lambda cmd: fake)
    mgr.on_step_boundary()
    mgr.on_step_boundary()
    assert mgr.forward_ct == 2
    assert not fake.started
