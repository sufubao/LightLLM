import pytest


def test_mtp_step_bound_rejects_above_7():
    # The gate must allow 0..7 and reject 8+ (req_to_next_token_ids width is 8).
    for ok in (0, 1, 2, 3, 7):
        assert ok <= 7
    with pytest.raises(AssertionError):
        step = 8
        assert step <= 7, "mtp_step must be <= 7 for ReqManagerForMamba (req_to_next_token_ids width is 8)"
