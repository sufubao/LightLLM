import pytest

from lightllm.server.core.objs.sampling_params import InvalidTokenIds


@pytest.mark.parametrize("token_id", [-1, 1 << 31])
def test_invalid_token_ids_reject_unsafe_value(token_id):
    token_ids = InvalidTokenIds()

    with pytest.raises(ValueError):
        token_ids.initialize([token_id])
