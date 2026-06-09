import types
import torch

# NOTE: importing lightllm.common.req_manager *first* trips a pre-existing circular import
# (req_manager line-8 imports gen_sampling_params -> basemodel -> infer_struct, which re-enters
# the half-initialized req_manager before ReqManager is defined). Importing basemodel first
# fully resolves that chain, after which ReqManagerForMamba imports cleanly. This is an
# import-ordering fix only; it does not alter the method-under-test or the duck-typed call below.
import lightllm.common.basemodel  # noqa: F401  (resolves circular import; must precede req_manager)
from lightllm.common.req_manager import ReqManagerForMamba


class _Buf:
    def __init__(self, t):
        self.buffer = t


def test_init_zeros_full_ssm_block():
    mtp_step = 3
    layer, n_req = 2, 4
    conv_dim, width = 8, 3
    conv_buf = torch.ones(layer, n_req, conv_dim, width)
    ssm_buf = torch.ones(layer, n_req * (mtp_step + 1), 5)

    dummy = types.SimpleNamespace(
        mtp_step=mtp_step,
        req_to_conv_state=_Buf(conv_buf),
        req_to_ssm_state=_Buf(ssm_buf),
    )
    req = types.SimpleNamespace(req_idx=2, mtp_accept_len=None)

    ReqManagerForMamba.init_linear_att_state(dummy, req)

    start = 2 * (mtp_step + 1)
    block = ssm_buf[:, start : start + (mtp_step + 1), ...]
    assert torch.count_nonzero(block) == 0, "all S+1 SSM rows of the block must be zeroed on init"
    # other requests' rows must be untouched
    assert torch.count_nonzero(ssm_buf[:, :start, ...]) > 0
    # conv slot for this request zeroed; canonical accept-len reset
    assert torch.count_nonzero(conv_buf[:, 2, ...]) == 0
    assert req.mtp_accept_len == 1
