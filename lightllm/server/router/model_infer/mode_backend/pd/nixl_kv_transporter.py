import pickle
import copy
import os
import threading
import time
from typing import Dict, Optional, Tuple
from torch import Tensor
from lightllm.server.pd_io_struct import PDChunckedTransTask, PDAgentMetadata
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

try:
    from nixl._api import nixl_agent as NixlWrapper
    from nixl._api import nixlBind
    from nixl._api import nixl_agent_config
    from nixl._api import nixl_thread_sync_t

    logger.info("Nixl is available")
except ImportError:
    logger.warning("nixl is not installed, which is required for pd disagreggation!!!")
    NixlWrapper = None


class NixlKVTransporter:
    def __init__(self, node_id: int, tp_idx: int, kv_move_buffer: Tensor):
        self.node_id = node_id
        self.tp_idx = tp_idx
        # A transporter is shared by the notification, transfer, status, and
        # failure-handling threads. NIXL's internal synchronization protects
        # individual calls; this lock also makes LightLLM's compound peer
        # lifecycle operations atomic.
        self._nixl_lock = threading.RLock()
        self.capture_telemetry = os.getenv("LIGHTLLM_NIXL_CAPTURE_TELEMETRY", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        conf = nixl_agent_config(sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT)
        if self.capture_telemetry:
            conf.capture_telemetry = True
            logger.info("NIXL telemetry enabled")
        self.nixl_agent = NixlWrapper(self.agent_name, conf)
        self._register_kv_move_buffer(kv_move_buffer=kv_move_buffer)
        self.remote_agents: Dict[str, PDAgentMetadata] = {}
        self._peer_generations: Dict[str, int] = {}
        self._next_peer_generation = 0
        self._broken_remote_agents: set[str] = set()
        self._active_xfers: Dict[int, Tuple[object, str, int]] = {}
        return

    @property
    def agent_name(self) -> str:
        return f"{self.node_id}_{self.tp_idx}"

    @property
    def agent_metadata(self):
        with self._nixl_lock:
            return self.nixl_agent.get_agent_metadata()

    @property
    def local_page_mem_desc(self):
        with self._nixl_lock:
            return self.nixl_agent.get_serialized_descs(self.page_reg_desc)

    def get_new_notifs(self) -> Dict[str, list[bytes]]:
        with self._nixl_lock:
            return self.nixl_agent.get_new_notifs()

    def _register_kv_move_buffer(self, kv_move_buffer: Tensor):
        self.num_pages, self.page_size, self.num_layers, self.kv_head_num, self.head_dims = kv_move_buffer.shape
        self.dtype_byte_size = kv_move_buffer.element_size()
        self.page_len = self.page_size * self.num_layers * self.kv_head_num * self.head_dims * self.dtype_byte_size
        self.page_reg_desc = self.nixl_agent.register_memory(kv_move_buffer)
        self.page_local_xfer_handles = self._create_paged_xfer_handles(self.page_reg_desc, self.num_pages)

    def _create_paged_xfer_handles(self, reg_desc: "nixlBind.nixlRegDList", page_num: int, agent_name: str = ""):
        base_addr, _, device_id, _ = reg_desc[0]
        pages_data = []
        for page_id in range(page_num):
            pages_data.append((base_addr + page_id * self.page_len, self.page_len, device_id))
        descs = self.nixl_agent.get_xfer_descs(pages_data, "VRAM")
        return self.nixl_agent.prep_xfer_dlist(agent_name, descs, "VRAM")

    def connect_add_remote_agent(self, remote_agent: PDAgentMetadata):
        with self._nixl_lock:
            self._ensure_remote_agent_locked(remote_agent)
        return

    @staticmethod
    def _same_remote_agent(left: PDAgentMetadata, right: PDAgentMetadata) -> bool:
        return (
            left.agent_name == right.agent_name
            and left.agent_metadata == right.agent_metadata
            and left.num_pages == right.num_pages
            and left.page_reg_desc == right.page_reg_desc
        )

    def _ensure_remote_agent_locked(self, remote_agent: PDAgentMetadata) -> Tuple[PDAgentMetadata, int]:
        peer_name = remote_agent.agent_name
        current_agent = self.remote_agents.get(peer_name)
        current_generation = self._peer_generations.get(peer_name)

        if current_agent is not None and not self._same_remote_agent(current_agent, remote_agent):
            self._mark_remote_agent_broken_locked(peer_name, current_generation)
            current_agent = self.remote_agents.get(peer_name)

        if peer_name in self._broken_remote_agents:
            if current_agent is not None:
                self._remove_remote_agent_locked(peer_name, expected_generation=current_generation)
            else:
                try:
                    self.nixl_agent.remove_remote_agent(peer_name)
                    self._broken_remote_agents.discard(peer_name)
                except BaseException as e:
                    raise RuntimeError(f"NIXL remote agent {peer_name} is broken and could not be removed") from e
            current_agent = self.remote_agents.get(peer_name)

        if current_agent is None:
            self._connect_add_remote_agent_locked(remote_agent)
            current_agent = self.remote_agents[peer_name]
            current_generation = self._peer_generations[peer_name]

        if peer_name in self._broken_remote_agents:
            raise RuntimeError(f"NIXL remote agent {peer_name} is unavailable while active transfers drain")

        return current_agent, current_generation

    def _connect_add_remote_agent_locked(self, remote_agent: PDAgentMetadata):
        if remote_agent.agent_name in self.remote_agents:
            return

        start_time = time.time()
        peer_name = self.nixl_agent.add_remote_agent(remote_agent.agent_metadata)
        if isinstance(peer_name, bytes):
            peer_name = peer_name.decode()

        assert (
            peer_name == remote_agent.agent_name
        ), f"Peer name {peer_name} does not match remote name {remote_agent.agent_name}"

        self._next_peer_generation += 1
        generation = self._next_peer_generation
        self.remote_agents[peer_name] = remote_agent
        self._peer_generations[peer_name] = generation
        try:
            page_mem_desc = self.nixl_agent.deserialize_descs(remote_agent.page_reg_desc)
            remote_agent.page_xfer_handles = self._create_paged_xfer_handles(
                page_mem_desc, remote_agent.num_pages, agent_name=peer_name
            )
        except BaseException:
            self._broken_remote_agents.add(peer_name)
            self._remove_remote_agent_locked(peer_name, expected_generation=generation)
            raise

        logger.info(
            f"Added remote agent {peer_name} generation {generation} "
            f"with mem desc {page_mem_desc} cost time: {time.time() - start_time} s"
        )
        self._broken_remote_agents.discard(peer_name)
        return

    def remove_remote_agent(self, peer_name: str):
        with self._nixl_lock:
            generation = self._peer_generations.get(peer_name)
            if generation is None:
                logger.warning(f"try to remove remote agent, but peer name {peer_name} agent did not exist")
                return
            self._mark_remote_agent_broken_locked(peer_name, generation)
        return

    def _has_active_xfers_locked(self, peer_name: str, generation: int) -> bool:
        return any(
            active_peer_name == peer_name and active_generation == generation
            for _, active_peer_name, active_generation in self._active_xfers.values()
        )

    def _mark_remote_agent_broken_locked(self, peer_name: str, generation: Optional[int]):
        if generation is None or self._peer_generations.get(peer_name) != generation:
            return
        self._broken_remote_agents.add(peer_name)
        self._remove_remote_agent_locked(peer_name, expected_generation=generation)

    def _remove_remote_agent_locked(self, peer_name: str, expected_generation: Optional[int] = None) -> bool:
        generation = self._peer_generations.get(peer_name)
        if generation is None:
            return False
        if expected_generation is not None and generation != expected_generation:
            return False
        if self._has_active_xfers_locked(peer_name, generation):
            self._broken_remote_agents.add(peer_name)
            logger.warning(
                f"defer removing remote agent {peer_name} generation {generation} until active transfers drain"
            )
            return False

        remote_agent = self.remote_agents[peer_name]
        try:
            self.nixl_agent.remove_remote_agent(remote_agent.agent_name)
        except BaseException as e:
            self._broken_remote_agents.add(peer_name)
            logger.error(f"remove remote agent {peer_name} generation {generation} failed")
            logger.exception(str(e))
            return False

        self.remote_agents.pop(peer_name, None)
        self._peer_generations.pop(peer_name, None)
        self._broken_remote_agents.discard(peer_name)
        if remote_agent.page_xfer_handles is not None:
            try:
                self.nixl_agent.release_dlist_handle(remote_agent.page_xfer_handles)
            except BaseException as e:
                logger.error(f"release remote agent {peer_name} descriptor handle failed")
                logger.exception(str(e))
            finally:
                remote_agent.page_xfer_handles = None
        return True

    def _send_notif_locked(self, peer_name: str, generation: int, notif_msg: bytes):
        try:
            self.nixl_agent.send_notif(remote_agent_name=peer_name, notif_msg=notif_msg)
        except BaseException:
            self._mark_remote_agent_broken_locked(peer_name, generation)
            raise

    def send_write_done_task_to_decode_node(self, trans_task: PDChunckedTransTask):
        decode_agent_name = trans_task.decode_agent_name
        with self._nixl_lock:
            _, generation = self._ensure_remote_agent_locked(trans_task.create_decode_agent_obj())
            new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
            new_trans_task.write_stage = "done"
            new_trans_task.mem_indexes = None
            new_trans_task.xfer_handle = None
            new_trans_task.decode_agent_metadata = None
            new_trans_task.decode_page_reg_desc = None
            new_trans_task.prefill_agent_name = self.agent_name
            new_trans_task.prefill_agent_metadata = self.agent_metadata
            new_trans_task.prefill_num_pages = self.num_pages
            new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
            self._send_notif_locked(decode_agent_name, generation, pickle.dumps(new_trans_task))
        return

    def send_write_request_task_to_decode_node(self, trans_task: PDChunckedTransTask):
        decode_agent_name = trans_task.decode_agent_name
        with self._nixl_lock:
            _, generation = self._ensure_remote_agent_locked(trans_task.create_decode_agent_obj())
            new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
            new_trans_task.write_stage = "request"
            new_trans_task.mem_indexes = None
            new_trans_task.xfer_handle = None
            new_trans_task.prefill_agent_name = self.agent_name
            new_trans_task.prefill_agent_metadata = self.agent_metadata
            new_trans_task.prefill_num_pages = self.num_pages
            new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
            self._send_notif_locked(decode_agent_name, generation, pickle.dumps(new_trans_task))
        return

    def send_write_ready_task_to_prefill_node(self, trans_task: PDChunckedTransTask):
        prefill_agent_name = trans_task.prefill_agent_name
        with self._nixl_lock:
            _, generation = self._ensure_remote_agent_locked(trans_task.create_prefill_agent_obj())
            new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
            new_trans_task.write_stage = "ready"
            new_trans_task.mem_indexes = None
            new_trans_task.xfer_handle = None
            new_trans_task.decode_agent_name = self.agent_name
            new_trans_task.decode_agent_metadata = self.agent_metadata
            new_trans_task.decode_num_pages = self.num_pages
            new_trans_task.decode_page_reg_desc = self.local_page_mem_desc
            self._send_notif_locked(prefill_agent_name, generation, pickle.dumps(new_trans_task))
        return

    def send_error_info_to_prefill_node(self, trans_task: PDChunckedTransTask):
        # decode node 主动发送错误信息给 prefill node, 但是只有到达一定阶段的任务才有对端的信息
        # 才能发送
        if trans_task.prefill_agent_name is None:
            return

        try:
            prefill_agent_name = trans_task.prefill_agent_name
            with self._nixl_lock:
                _, generation = self._ensure_remote_agent_locked(trans_task.create_prefill_agent_obj())
                assert trans_task.error_info is not None
                new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
                new_trans_task.write_stage = "error"
                new_trans_task.mem_indexes = None
                new_trans_task.xfer_handle = None
                new_trans_task.decode_agent_name = self.agent_name
                new_trans_task.decode_agent_metadata = self.agent_metadata
                new_trans_task.decode_num_pages = self.num_pages
                new_trans_task.decode_page_reg_desc = self.local_page_mem_desc
                self._send_notif_locked(prefill_agent_name, generation, pickle.dumps(new_trans_task))
        except BaseException as e:
            logger.error(f"send error info to prefill node failed: {trans_task.to_str()}")
            logger.exception(str(e))
        return

    def send_error_info_to_decode_node(self, trans_task: PDChunckedTransTask):
        try:
            decode_agent_name = trans_task.decode_agent_name
            with self._nixl_lock:
                _, generation = self._ensure_remote_agent_locked(trans_task.create_decode_agent_obj())
                assert trans_task.error_info is not None
                new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
                new_trans_task.write_stage = "error"
                new_trans_task.mem_indexes = None
                new_trans_task.xfer_handle = None
                new_trans_task.prefill_agent_name = self.agent_name
                new_trans_task.prefill_agent_metadata = self.agent_metadata
                new_trans_task.prefill_num_pages = self.num_pages
                new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc
                self._send_notif_locked(decode_agent_name, generation, pickle.dumps(new_trans_task))
        except BaseException as e:
            logger.error(f"send error info to decode node failed: {trans_task.to_str()}")
            logger.exception(str(e))
        return

    def write_blocks_paged(
        self,
        trans_task: PDChunckedTransTask,
    ) -> int:
        """
        prefill node call this function to write kv blocks into decode node pages
        """
        decode_agent_name = trans_task.decode_agent_name
        with self._nixl_lock:
            remote_agent, generation = self._ensure_remote_agent_locked(trans_task.create_decode_agent_obj())
            assert trans_task.src_page_index is not None and trans_task.dst_page_index is not None
            handle = None
            try:
                handle = self.nixl_agent.make_prepped_xfer(
                    "WRITE",
                    self.page_local_xfer_handles,
                    [trans_task.src_page_index],
                    remote_agent.page_xfer_handles,
                    [trans_task.dst_page_index],
                    b"",
                )
                if not handle:
                    raise RuntimeError(f"make_prepped_xfer failed for task: {trans_task.to_str()}")
                self.nixl_agent.transfer(handle)
                self._active_xfers[id(handle)] = (handle, decode_agent_name, generation)
                return handle
            except BaseException:
                if handle:
                    try:
                        self.nixl_agent.release_xfer_handle(handle=handle)
                    except BaseException as release_error:
                        logger.error(f"release failed transfer handle for remote agent {decode_agent_name}")
                        logger.exception(str(release_error))
                self._mark_remote_agent_broken_locked(decode_agent_name, generation)
                raise

    def check_task_status(self, trans_task: PDChunckedTransTask) -> str:
        assert trans_task.xfer_handle is not None
        handle = trans_task.xfer_handle
        with self._nixl_lock:
            active_xfer = self._active_xfers.get(id(handle))
            try:
                xfer_state = self.nixl_agent.check_xfer_state(handle)
            except BaseException:
                if active_xfer is not None:
                    _, peer_name, generation = active_xfer
                    self._mark_remote_agent_broken_locked(peer_name, generation)
                raise
            if xfer_state == "ERR":
                logger.warning(f"Transfer failed with trans task {trans_task.to_str()} for handle {handle}")
                if active_xfer is not None:
                    _, peer_name, generation = active_xfer
                    self._mark_remote_agent_broken_locked(peer_name, generation)
            return xfer_state

    def release_xfer_handle(self, handle):
        with self._nixl_lock:
            active_xfer = self._active_xfers.get(id(handle))
            self.nixl_agent.release_xfer_handle(handle=handle)
            self._active_xfers.pop(id(handle), None)
            if active_xfer is not None:
                _, peer_name, generation = active_xfer
                if peer_name in self._broken_remote_agents:
                    self._remove_remote_agent_locked(peer_name, expected_generation=generation)
        return

    def get_xfer_telemetry(self, handle):
        with self._nixl_lock:
            return self.nixl_agent.get_xfer_telemetry(handle)

    def query_xfer_backend(self, handle):
        with self._nixl_lock:
            return self.nixl_agent.query_xfer_backend(handle)

    def shutdown(self):
        with self._nixl_lock:
            for handle, _, _ in list(self._active_xfers.values()):
                try:
                    self.nixl_agent.release_xfer_handle(handle=handle)
                except BaseException as e:
                    logger.error("release active transfer handle during NIXL shutdown failed")
                    logger.exception(str(e))
            self._active_xfers.clear()
            for agent_name in list(self.remote_agents.keys()):
                generation = self._peer_generations.get(agent_name)
                self._broken_remote_agents.add(agent_name)
                self._remove_remote_agent_locked(agent_name, expected_generation=generation)
            self.nixl_agent.deregister_memory(self.page_reg_desc)
            self.nixl_agent.release_dlist_handle(self.page_local_xfer_handles)
        return
