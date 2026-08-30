from lightllm.server.core.objs.sampling_params import SamplingParams


def test_sampling_params_combines_explicit_and_logit_bias_invalid_tokens():
    params = SamplingParams()
    params.init(
        None,
        invalid_token_ids=[271, 42],
        logit_bias={"42": -100, "99": -100},
    )

    assert params.invalid_token_ids.to_list() == [271, 42, 99]
