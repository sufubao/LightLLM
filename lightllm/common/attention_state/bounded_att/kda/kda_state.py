from functools import partial
import torch
from lightllm.common.attention_state.state_ops.conv_ssm import ConvSSMRuntime
from ..conv_ssm_types import ConvSSMStateConfig
from ..conv_ssm_state import ConvSSMState
from lightllm.common.attention_state.bounded_att.conv_ssm_formats import conv_ssm_spec
from lightllm.common.attention_state.storage.page_pool import KDAPagesPool


class KDAState(ConvSSMState):
    def __init__(
        self, config: ConvSSMStateConfig, request_capacity: int, mtp_step: int = 0, device: str | torch.device = "cuda"
    ):
        assert isinstance(config, ConvSSMStateConfig), "config must be ConvSSMStateConfig"
        super().__init__(
            config=config,
            spec=conv_ssm_spec(config=config, request_capacity=request_capacity, mtp_step=mtp_step, algorithm="kda"),
            runtime=ConvSSMRuntime(config),
            request_capacity=request_capacity,
            mtp_step=mtp_step,
            device=device,
            page_pool_factory=partial(KDAPagesPool, config=config),
        )


def create_kda_state(
    config: ConvSSMStateConfig,
    request_capacity: int,
    mtp_step: int = 0,
    device: str | torch.device = "cuda",
) -> KDAState:
    """创建 KDA 请求运行态，大小页池由后续容量分配流程初始化。"""
    return KDAState(
        config=config,
        request_capacity=request_capacity,
        mtp_step=mtp_step,
        device=device,
    )
