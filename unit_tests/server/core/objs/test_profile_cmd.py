import pickle
import pytest
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


def test_unknown_action_raises():
    req = ProfileControlReq(action="bogus", profile_id=1)
    with pytest.raises(ValueError):
        req.to_worker_cmd()
