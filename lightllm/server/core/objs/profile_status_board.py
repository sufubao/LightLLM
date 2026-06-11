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
        row = self.arr[slot].copy()
        return {
            "state": STATE_NAMES.get(int(row[_FIELD_STATE]), "unknown"),
            "profile_id": int(row[_FIELD_PROFILE_ID]),
            "forward_ct": int(row[_FIELD_FORWARD_CT]),
            "target_ct": int(row[_FIELD_TARGET_CT]),
            "error_code": int(row[_FIELD_ERROR_CODE]),
        }
