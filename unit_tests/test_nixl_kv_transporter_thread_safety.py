import threading
import time

import pytest

from lightllm.server.pd_io_struct import PDChunckedTransTask, PDAgentMetadata
from lightllm.server.router.model_infer.mode_backend.pd import nixl_kv_transporter


class _FakeBuffer:
    shape = (1, 1, 1, 1, 1)

    @staticmethod
    def element_size():
        return 1


class _FakeConfig:
    def __init__(self, sync_mode=None):
        self.sync_mode = sync_mode
        self.capture_telemetry = False


class _FakeSyncMode:
    NIXL_THREAD_SYNC_STRICT = object()


class _FakeNixlAgent:
    instances = []

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.add_calls = []
        self.remove_calls = []
        self.released_dlists = []
        self.released_xfers = []
        self.send_calls = []
        self.fail_next_send = False
        self._native_call_lock = threading.Lock()
        self._native_calls = 0
        self.max_native_calls = 0
        self.__class__.instances.append(self)

    def _enter_native_call(self):
        with self._native_call_lock:
            self._native_calls += 1
            self.max_native_calls = max(self.max_native_calls, self._native_calls)

    def _leave_native_call(self):
        with self._native_call_lock:
            self._native_calls -= 1

    def register_memory(self, _buffer):
        return [(1000, 1, 0, "")]

    def get_xfer_descs(self, pages_data, _mem_type):
        return pages_data

    def prep_xfer_dlist(self, agent_name, descs, _mem_type):
        return (agent_name, tuple(descs))

    def get_agent_metadata(self):
        return b"local-metadata"

    def get_serialized_descs(self, _reg_desc):
        return b"local-page-desc"

    def get_new_notifs(self):
        return {}

    def add_remote_agent(self, metadata):
        self._enter_native_call()
        try:
            time.sleep(0.01)
            self.add_calls.append(metadata)
            return metadata.decode()
        finally:
            self._leave_native_call()

    def deserialize_descs(self, _page_reg_desc):
        return [(2000, 1, 0, "")]

    def remove_remote_agent(self, peer_name):
        self._enter_native_call()
        try:
            self.remove_calls.append(peer_name)
        finally:
            self._leave_native_call()

    def release_dlist_handle(self, handle):
        self.released_dlists.append(handle)

    def send_notif(self, remote_agent_name, notif_msg):
        self._enter_native_call()
        try:
            self.send_calls.append((remote_agent_name, notif_msg))
            if self.fail_next_send:
                self.fail_next_send = False
                raise RuntimeError("remote disconnected")
        finally:
            self._leave_native_call()

    def make_prepped_xfer(self, *_args):
        return object()

    def transfer(self, _handle):
        return "PROC"

    def check_xfer_state(self, _handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        self.released_xfers.append(handle)

    def deregister_memory(self, _reg_desc):
        return None


@pytest.fixture
def transporter(monkeypatch):
    _FakeNixlAgent.instances.clear()
    monkeypatch.setattr(nixl_kv_transporter, "NixlWrapper", _FakeNixlAgent)
    monkeypatch.setattr(nixl_kv_transporter, "nixl_agent_config", _FakeConfig, raising=False)
    monkeypatch.setattr(nixl_kv_transporter, "nixl_thread_sync_t", _FakeSyncMode, raising=False)
    return nixl_kv_transporter.NixlKVTransporter(node_id=1, tp_idx=0, kv_move_buffer=_FakeBuffer())


def _remote_agent(name="peer"):
    return PDAgentMetadata(
        agent_name=name,
        agent_metadata=name.encode(),
        num_pages=1,
        page_reg_desc=b"remote-page-desc",
    )


def _task(remote_agent):
    return PDChunckedTransTask(
        request_id=1,
        start_kv_index=0,
        end_kv_index=1,
        time_out_secs=60,
        pd_master_node_id=1,
        prefill_dp_index=0,
        decode_dp_index=0,
        src_device_id=0,
        dst_device_id=0,
        mem_indexes=[0],
        prefill_agent_name=remote_agent.agent_name,
        prefill_agent_metadata=remote_agent.agent_metadata,
        prefill_num_pages=remote_agent.num_pages,
        prefill_page_reg_desc=remote_agent.page_reg_desc,
        decode_agent_name="decode",
        decode_agent_metadata=b"decode",
        decode_num_pages=1,
        decode_page_reg_desc=b"decode-page-desc",
        first_gen_token_id=None,
        first_gen_token_logprob=None,
        src_page_index=0,
        dst_page_index=0,
    )


def _run_concurrently(callables):
    barrier = threading.Barrier(len(callables) + 1)
    errors = []

    def run(callable_):
        barrier.wait()
        try:
            callable_()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(callable_,)) for callable_ in callables]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    return errors


def test_uses_strict_nixl_sync_mode(transporter):
    assert transporter.nixl_agent.config.sync_mode is _FakeSyncMode.NIXL_THREAD_SYNC_STRICT


def test_concurrent_peer_admission_calls_native_add_once(transporter):
    remote_agent = _remote_agent()

    errors = _run_concurrently(
        [
            lambda: transporter.connect_add_remote_agent(remote_agent),
            lambda: transporter.connect_add_remote_agent(remote_agent),
        ]
    )

    assert errors == []
    assert transporter.nixl_agent.add_calls == [b"peer"]
    assert transporter.nixl_agent.max_native_calls == 1


def test_failed_send_is_removed_before_one_serialized_reconnect(transporter):
    remote_agent = _remote_agent()
    task = _task(remote_agent)
    transporter.connect_add_remote_agent(remote_agent)
    transporter.nixl_agent.fail_next_send = True

    errors = _run_concurrently(
        [
            lambda: transporter.send_write_ready_task_to_prefill_node(task),
            lambda: transporter.send_write_ready_task_to_prefill_node(task),
        ]
    )

    assert len(errors) == 1
    assert str(errors[0]) == "remote disconnected"
    assert transporter.nixl_agent.add_calls == [b"peer", b"peer"]
    assert transporter.nixl_agent.remove_calls == ["peer"]
    assert transporter._peer_generations == {"peer": 2}
    assert transporter.nixl_agent.max_native_calls == 1


def test_active_transfer_defers_removal_and_stale_generation_cannot_remove_reconnect(transporter):
    remote_agent = _remote_agent("decode")
    task = _task(_remote_agent())
    handle = transporter.write_blocks_paged(task)
    first_generation = transporter._peer_generations["decode"]

    with transporter._nixl_lock:
        transporter._mark_remote_agent_broken_locked("decode", first_generation)

    assert "decode" in transporter.remote_agents
    assert "decode" in transporter._broken_remote_agents
    with pytest.raises(RuntimeError, match="active transfers drain"):
        transporter.connect_add_remote_agent(remote_agent)

    transporter.release_xfer_handle(handle)
    transporter.connect_add_remote_agent(remote_agent)
    second_generation = transporter._peer_generations["decode"]
    assert second_generation > first_generation

    with transporter._nixl_lock:
        transporter._mark_remote_agent_broken_locked("decode", first_generation)

    assert transporter._peer_generations == {"decode": second_generation}
    assert "decode" in transporter.remote_agents
