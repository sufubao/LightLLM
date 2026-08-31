from functools import lru_cache
import time

from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.envs_utils import get_unique_server_name


@lru_cache(maxsize=1)
def _server_ready_flag():
    return SharedInt(f"{get_unique_server_name()}_server_ready")


def set_server_ready(ready: bool):
    _server_ready_flag().set_value(int(ready))


def is_server_ready() -> bool:
    return bool(_server_ready_flag().get_value())


@lru_cache(maxsize=1)
def _cpu_cache_ready_flag():
    return SharedInt(f"{get_unique_server_name()}_cpu_cache_ready")


def set_cpu_cache_ready(ready: bool):
    _cpu_cache_ready_flag().set_value(int(ready))


def wait_cpu_cache_ready(timeout_seconds: float = 600):
    deadline = time.monotonic() + timeout_seconds
    while not _cpu_cache_ready_flag().get_value():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the CPU cache manager")
        time.sleep(0.01)
