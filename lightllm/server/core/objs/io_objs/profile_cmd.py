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
    # 0 表示 "停止当前任意 capture": 无状态的 /stop_profile 接口不知道原始 profile_id。
    profile_id: int = 0


@dataclass
class ProfileControlReq:
    """httpserver -> router 的 profile 控制消息, router 转换为 worker cmd 后经 ShmObjsIOBuffer 广播。"""

    action: str  # "start" or "stop"
    profile_id: int
    # router 层路由用, 不会转发给 worker cmd
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
        if self.action == "stop":
            return StopProfileCmd(profile_id=self.profile_id)
        raise ValueError(f"unknown profile action: {self.action!r}")
