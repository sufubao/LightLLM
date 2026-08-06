import pytest
from pydantic import ValidationError

from lightllm.server.api_models import ChatCompletionRequest, CompletionRequest
from lightllm.server.core.objs.py_sampling_params import SamplingParams as PySamplingParams
from lightllm.server.core.objs.sampling_params import SamplingParams


def _chat_request(**kwargs):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **kwargs)


def _completion_request(**kwargs):
    return CompletionRequest(model="test-model", prompt="hi", **kwargs)


@pytest.mark.parametrize("request_factory", [_chat_request, _completion_request])
def test_request_seed_defaults_to_none(request_factory):
    assert request_factory().seed is None
    assert request_factory(seed=None).seed is None


@pytest.mark.parametrize("request_factory", [_chat_request, _completion_request])
@pytest.mark.parametrize("seed", [-1, 0, 42])
def test_request_seed_accepts_supported_values(request_factory, seed):
    assert request_factory(seed=seed).seed == seed


@pytest.mark.parametrize("request_factory", [_chat_request, _completion_request])
def test_request_seed_rejects_values_below_random_sentinel(request_factory):
    with pytest.raises(ValidationError, match="greater than or equal to -1"):
        request_factory(seed=-2)


def test_sampling_params_normalize_none_seed_to_random_sentinel():
    params = SamplingParams()
    params.init(tokenizer=None, seed=None)
    assert params.seed == -1

    py_params = PySamplingParams(seed=None)
    assert py_params.seed == -1


def test_sampling_params_reject_values_below_random_sentinel():
    with pytest.raises(ValueError, match="seed must be -1"):
        params = SamplingParams()
        params.init(tokenizer=None, seed=-2)

    py_params = PySamplingParams(seed=-2)
    with pytest.raises(ValueError, match="seed must be -1"):
        py_params.verify()
