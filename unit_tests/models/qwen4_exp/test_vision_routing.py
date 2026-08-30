from transformers.configuration_utils import PretrainedConfig

from lightllm.utils import config_utils


def test_qwen4_reports_embedded_vision_module(monkeypatch):
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda _: (
            {
                "model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "vision_config": {"out_hidden_size": 2560},
            },
            {},
        ),
        raising=False,
    )
    config_utils.has_vision_module.cache_clear()

    assert config_utils.has_vision_module("/models/qwen38") is True
