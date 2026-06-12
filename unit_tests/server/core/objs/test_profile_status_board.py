from lightllm.server.core.objs.profile_status_board import (
    ProfileStatusBoard,
    STATE_IDLE,
    STATE_RUNNING,
    STATE_ERROR,
    ERROR_NONE,
    ERROR_START_FAILED,
)


def make_board(num_worker_slots, name):
    board = ProfileStatusBoard(num_worker_slots=num_worker_slots, name=name)
    board.arr[:] = 0  # 清掉可能残留的上次运行数据
    return board


def test_set_and_get_slot():
    board = make_board(4, "test_profile_status_board_a")
    board.set_slot(0, state=STATE_RUNNING, profile_id=99, forward_ct=7, target_ct=10, error_code=ERROR_NONE)
    slot = board.get_slot(0)
    assert slot == {"state": "running", "profile_id": 99, "forward_ct": 7, "target_ct": 10, "error_code": 0}


def test_partial_update_preserves_other_fields():
    board = make_board(4, "test_profile_status_board_b")
    board.set_slot(1, state=STATE_RUNNING, profile_id=5, forward_ct=1, target_ct=9, error_code=ERROR_NONE)
    board.set_slot(1, forward_ct=3)
    slot = board.get_slot(1)
    assert slot["forward_ct"] == 3
    assert slot["profile_id"] == 5
    assert slot["state"] == "running"


def test_router_slot_index():
    board = make_board(4, "test_profile_status_board_c")
    assert board.router_slot == 4
    board.set_slot(board.router_slot, state=STATE_ERROR, error_code=ERROR_START_FAILED)
    assert board.get_slot(board.router_slot)["state"] == "error"


def test_two_instances_share_memory():
    writer = make_board(2, "test_profile_status_board_d")
    reader = ProfileStatusBoard(num_worker_slots=2, name="test_profile_status_board_d")
    writer.set_slot(0, state=STATE_IDLE, profile_id=777)
    assert reader.get_slot(0)["profile_id"] == 777
