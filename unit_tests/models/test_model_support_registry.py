from dataclasses import replace

import pytest

from lightllm.models.registry import (
    ModelContext,
    ModelRegistryStore,
    ModelSupport,
    ModelSupportConflictError,
    UnsupportedModelError,
    get_model_support,
)


class DefaultModel:
    pass


class SpecializedModel:
    pass


def test_model_context_exposes_normalized_nested_configs():
    context = ModelContext.from_config(
        {
            "model_type": "omni",
            "architectures": ["OmniForConditionalGeneration"],
            "thinker_config": {
                "text_config": {"model_type": "text"},
                "vision_config": {"model_type": "vision"},
                "audio_config": {"model_type": "audio"},
            },
        },
        model_dir="/models/omni",
    )

    assert context.model_type == "omni"
    assert context.architectures == ("OmniForConditionalGeneration",)
    assert context.text_config["model_type"] == "text"
    assert context.vision_config["model_type"] == "vision"
    assert context.audio_config["model_type"] == "audio"
    assert context.model_dir == "/models/omni"


def test_legacy_conditional_registration_overrides_default():
    registry = ModelRegistryStore()
    registry("example")(DefaultModel)
    registry("example", condition=lambda config: config.get("variant") == "special")(SpecializedModel)

    assert registry.resolve({"model_type": "example"}).text_model is DefaultModel
    assert registry.resolve({"model_type": "example", "variant": "special"}).text_model is SpecializedModel


def test_equal_priority_matches_report_support_names():
    registry = ModelRegistryStore()
    base = ModelSupport(name="first", model_types=("example",), text_model=DefaultModel)
    registry.register_support(base)
    registry.register_support(replace(base, name="second", text_model=SpecializedModel))

    with pytest.raises(ModelSupportConflictError, match="first, second"):
        registry.resolve({"model_type": "example"})


def test_lazy_module_is_loaded_only_when_its_model_type_is_resolved(monkeypatch):
    registry = ModelRegistryStore()
    loaded = []

    def import_fake_module(module_name):
        loaded.append(module_name)
        registry.register_support(ModelSupport(name="lazy", model_types=("lazy_type",), text_model=SpecializedModel))

    monkeypatch.setattr("lightllm.models.registry.importlib.import_module", import_fake_module)
    registry.register_lazy("lazy_type", "external_models.lazy_support")

    with pytest.raises(UnsupportedModelError):
        registry.resolve({"model_type": "different_type"})
    assert loaded == []

    assert registry.resolve({"model_type": "lazy_type"}).name == "lazy"
    assert loaded == ["external_models.lazy_support"]


@pytest.mark.parametrize(
    "config, expected_name",
    [
        (
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "text_config": {"model_type": "qwen3"},
                "vision_config": {},
            },
            "qwen3_vl",
        ),
        (
            {
                "model_type": "llava",
                "architectures": ["TarsierForConditionalGeneration"],
                "text_config": {"model_type": "qwen2"},
            },
            "tarsier2_qwen2",
        ),
    ],
)
def test_migrated_builtin_supports_declare_complete_multimodal_bundle(config, expected_name):
    support = get_model_support(config)

    assert support.name == expected_name
    assert support.is_multimodal
    assert support.tokenizer_factory is not None
    assert support.vision_factory is not None
