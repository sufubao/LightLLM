import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args

if TYPE_CHECKING:
    from lightllm.server.core.objs.shm_req_manager import ShmReqManager

logger = init_logger(__name__)


def scan_visual_unhealthy_sentinels(args) -> List[str]:
    """Return descriptions of visual worker watchdog sentinel files."""

    visual_dp = int(getattr(args, "visual_dp", 0) or 0)
    visual_tp = int(getattr(args, "visual_tp", 0) or 0)
    if visual_dp <= 0 or visual_tp <= 0:
        return []

    server_name = get_unique_server_name()
    dead = []
    for dp in range(visual_dp):
        for tp in range(visual_tp):
            path = f"/tmp/lightllm_visual_unhealthy_{server_name}_dp{dp}_tp{tp}"
            if not os.path.exists(path):
                continue
            try:
                with open(path) as fh:
                    info = fh.read().strip()
            except Exception:
                info = "<read failed>"
            dead.append(f"dp={dp} tp={tp}: {info}")
    return dead


@dataclass
class HealthObj:
    grace_timeout: int = int(os.getenv("HEALTH_TIMEOUT", "200"))

    def __post_init__(self):
        uid = get_unique_server_name()
        self.latest_success_infer_time_mark = SharedInt(f"{uid}_latest_success_infer_time_mark")
        self.run_reqs_count_mark = SharedInt(f"{uid}_run_reqs_count_mark")

    def check(self, shm_req_manager: "ShmReqManager") -> bool:
        """On-the-fly health check: recent success is ok; otherwise require no in-flight shm requests."""
        try:
            args = get_env_start_args()
            visual_dead = scan_visual_unhealthy_sentinels(args)
            if visual_dead:
                logger.error(f"Health check failed: visual worker watchdog reports dead: {visual_dead}")
                return False

            now = time.time()
            last_success_time = self.latest_success_infer_time_mark.get_value()

            # 如果最近一次成功推理的时间距离现在小于 grace_timeout，则认为系统健康
            if now - last_success_time <= self.grace_timeout:
                return True
            elif self.run_reqs_count_mark.get_value() == 0 and shm_req_manager.is_idle():
                # 如果最近一次成功推理的时间距离现在大于 grace_timeout，并且没有在推理的请求，则认为系统健康
                return True
            else:
                logger.warning(
                    "Health check failed: no success for %ss and in-flight shm requests remain",
                    int(now - last_success_time),
                )
                return False
        except Exception as e:
            logger.exception(str(e))
            return False


health_obj = HealthObj()


def health_check(shm_req_manager: "ShmReqManager") -> bool:
    return health_obj.check(shm_req_manager)
