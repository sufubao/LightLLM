import pytest

# NOTE: importing lightllm.common.req_manager *first* trips a pre-existing circular import
# (req_manager line-8 imports gen_sampling_params -> basemodel -> infer_struct, which re-enters
# the half-initialized req_manager before ReqManager is defined). Importing basemodel first
# fully resolves that chain, after which req_manager imports cleanly. Import-ordering fix only;
# it does not alter the constant or the real gate exercised below.
import lightllm.common.basemodel  # noqa: F401  (resolves circular import; must precede req_manager)


def test_width_constant_matches_alloc():
    from lightllm.common import req_manager

    # The cap must be derived from the real req_to_next_token_ids width, not a magic literal (#14).
    assert req_manager.REQ_NEXT_TOKEN_IDS_WIDTH == 8


def test_gate_accepts_within_width_and_rejects_above():
    from lightllm.common.req_manager import (
        REQ_NEXT_TOKEN_IDS_WIDTH,
        assert_mtp_step_within_next_token_ids_width,
    )

    # 0 .. width-1 are allowed
    for ok in range(0, REQ_NEXT_TOKEN_IDS_WIDTH):
        assert_mtp_step_within_next_token_ids_width(ok)
    # width and above are rejected by the REAL gate
    with pytest.raises(AssertionError):
        assert_mtp_step_within_next_token_ids_width(REQ_NEXT_TOKEN_IDS_WIDTH)
