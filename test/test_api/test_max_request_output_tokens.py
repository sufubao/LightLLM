from types import SimpleNamespace

import lightllm.server.core.objs.sampling_params as sampling_params_module
from lightllm.server.core.objs.sampling_params import SamplingParams


def test_max_request_output_tokens_is_default_and_hard_limit(monkeypatch):
    monkeypatch.setattr(
        sampling_params_module,
        "get_env_start_args",
        lambda: SimpleNamespace(max_request_output_tokens=1024),
    )

    for requested, expected in ((None, 1024), (256, 256), (2048, 1024)):
        params = SamplingParams()
        kwargs = {} if requested is None else {"max_new_tokens": requested}
        params.init(None, **kwargs)
        assert params.max_new_tokens == expected
