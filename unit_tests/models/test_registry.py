"""Model selection, lazy imports, and external registration compatibility."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_registry_import_and_selection_without_model_implementations():
    # The top-level lightllm package imports torch for device adaptation.
    # Registry metadata must not go on to import implementations or initialize CUDA.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class RejectModelDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split(".")[0] in ("triton", "transformers")
                or fullname.startswith("lightllm.models.") and fullname.endswith(".model")):
            raise ImportError("Registry imported model dependency: " + fullname)
sys.meta_path.insert(0, RejectModelDependencies())
import lightllm.models as models
from lightllm.models.registry import ModelRegistry
config = ModelRegistry.get_model_config({"model_type": "qwen3"})
assert config.model_class == "lightllm.models.qwen3.model:Qwen3TpPartModel"
assert not config.is_multimodal
config = ModelRegistry.get_model_config({"model_type": "llava", "text_config": {"model_type": "llama"}})
assert config.model_class == "lightllm.models.llava.model:LlavaTpPartModel"
config = ModelRegistry.get_model_config({
    "model_type": "llava", "architectures": ["TarsierForConditionalGeneration"],
    "text_config": {"model_type": "qwen2_vl"},
})
assert config.model_class == "lightllm.models.tarsier2.model:Tarsier2Qwen2VLTpPartModel"
assert config.is_multimodal
try:
    models.get_model_class({"model_type": "unsupported"})
except ValueError:
    pass
else:
    raise AssertionError("unsupported model accepted")
import torch
assert not torch.cuda.is_initialized()
assert "transformers" not in sys.modules
assert not any(name.startswith("lightllm.models.") and name.endswith(".model") for name in sys.modules)
""",
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def registry():
    from lightllm.models.registry import _ModelRegistries

    return _ModelRegistries()


@pytest.fixture
def model_module(tmp_path, monkeypatch):
    name = "registry_test_model"
    (tmp_path / f"{name}.py").write_text(
        "class ExampleModel:\n" "    def __init__(self, kvargs):\n" "        self.kvargs = kvargs\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, name, None)
    del sys.modules[name]
    return name


def test_selected_model_loads_lazily_and_receives_original_kwargs(registry, model_module):
    registry.register("example", "missing_default_dependency:Default")
    registry.register(
        "example", f"{model_module}:ExampleModel", is_multimodal=True, condition=lambda cfg: cfg.get("visual")
    )
    config = {"model_type": "example", "visual": True}
    descriptor = registry.get_model_config(config)
    assert descriptor.is_multimodal
    assert model_module not in sys.modules
    kwargs = {"weight_dir": "/example", "main_model": object()}
    model, is_multimodal = registry.get_model(config, kwargs)
    assert model.kvargs is kwargs
    assert is_multimodal
    assert type(model) is registry.get_model_class(config)


@pytest.mark.parametrize("guarded_fallback", [False, True])
def test_decorator_registration_and_default_fallback(registry, guarded_fallback):
    condition = (lambda cfg: cfg.get("allow_default", True)) if guarded_fallback else None

    @registry(["example", "alias"], condition=condition, is_fallback=guarded_fallback)
    class DefaultModel:
        def __init__(self, kvargs):
            self.kvargs = kvargs

    @registry("example", condition=lambda cfg: cfg.get("reward", False))
    class RewardModel(DefaultModel):
        pass

    assert registry.get_model_class({"model_type": "alias"}) is DefaultModel
    assert registry.get_model_class({"model_type": "example"}) is DefaultModel
    assert registry.get_model_class({"model_type": "example", "reward": True}) is RewardModel
    if guarded_fallback:
        with pytest.raises(ValueError, match="not supported"):
            registry.get_model_config({"model_type": "example", "allow_default": False})


@pytest.mark.parametrize("conditional", [False, True])
def test_ambiguous_registration_fails_before_import(registry, conditional):
    condition = (lambda cfg: True) if conditional else None
    registry.register("example", "unavailable.first:First", condition=condition)
    registry.register("example", "unavailable.second:Second", condition=condition)
    with pytest.raises(ValueError, match="Ambiguous.*example.*First.*Second"):
        registry.get_model_class({"model_type": "example"})


def test_selected_dependency_failure_is_not_hidden_by_fallback(registry, tmp_path, monkeypatch):
    (tmp_path / "registry_broken_model.py").write_text("import missing_registry_test_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    registry.register("example", "builtins:dict")
    registry.register("example", "registry_broken_model:Model", condition=lambda cfg: True)
    with pytest.raises(ModuleNotFoundError) as error:
        registry.get_model_class({"model_type": "example"})
    assert error.value.name == "missing_registry_test_dependency"


def test_draft_registration_keeps_modes_separate_and_rejects_duplicates(model_module):
    from lightllm.models.draft_registry import _DraftModelRegistry

    registry = _DraftModelRegistry()
    registry.register(("example", "alias"), ("with_att", "no_att"), f"{model_module}:ExampleModel")
    assert model_module not in sys.modules
    cls = registry.get_model_class({"model_type": "alias"}, "no_att")
    assert cls.__name__ == "ExampleModel"
    with pytest.raises(ValueError, match="Duplicate draft model registration"):
        registry("example", "with_att")(dict)
    with pytest.raises(ValueError, match="Unsupported speculative draft model"):
        registry.get_model_class({"model_type": "example"}, "other_mode")


@pytest.mark.parametrize("architectures", [None, ["Qwen2ForCausalLM"], ["Qwen2ForRewardModel"]])
def test_reward_architecture_overrides_default(architectures):
    from lightllm.models.registry import ModelRegistry

    selected = ModelRegistry.get_model_config({"model_type": "qwen2", "architectures": architectures})
    expected = (
        "qwen2_reward.model:Qwen2RewardTpPartModel"
        if architectures == ["Qwen2ForRewardModel"]
        else "qwen2.model:Qwen2TpPartModel"
    )
    assert selected.model_class == "lightllm.models." + expected


@pytest.mark.parametrize(
    "mode,class_name",
    [("dflash", "Qwen3DFlashModel"), ("dspark", "Qwen3DSparkModel"), ("eagle3", "Qwen3EagleModel")],
)
def test_builtin_target_and_draft_entry_points_load_real_classes(mode, class_name):
    import lightllm.models as models

    target = models.get_model_class({"model_type": "qwen3"})
    draft = models.get_draft_model_class({"model_type": "qwen3"}, mode)
    assert target.__name__ == "Qwen3TpPartModel"
    assert models.Qwen3TpPartModel is target
    assert draft.__name__ == class_name
    assert draft is not target


@pytest.mark.parametrize(
    "text_type,class_name",
    [
        ("llama", "Tarsier2LlamaTpPartModel"),
        ("qwen2", "Tarsier2Qwen2TpPartModel"),
        ("qwen2_vl", "Tarsier2Qwen2VLTpPartModel"),
    ],
)
def test_tarsier_architecture_selects_its_text_model(text_type, class_name):
    from lightllm.models.registry import ModelRegistry

    selected = ModelRegistry.get_model_config(
        {
            "model_type": "llava",
            "architectures": ["TarsierForConditionalGeneration"],
            "text_config": {"model_type": text_type},
        }
    )
    assert selected.model_class == f"lightllm.models.tarsier2.model:{class_name}"
    assert selected.is_multimodal


@pytest.mark.parametrize(
    "architectures", [[], None, ["LlavaForConditionalGeneration", "TarsierForConditionalGeneration"]]
)
def test_llava_is_not_tarsier_without_its_primary_architecture(architectures):
    from lightllm.models.registry import ModelRegistry

    selected = ModelRegistry.get_model_config(
        {"model_type": "llava", "architectures": architectures, "text_config": {"model_type": "llama"}}
    )
    assert selected.model_class == "lightllm.models.llava.model:LlavaTpPartModel"


@pytest.mark.parametrize(
    "nested_config",
    [
        {},
        {"text_config": None},
        {"text_config": {"model_type": "unsupported"}},
        {"llm_config": {"model_type": "llama"}},
    ],
)
def test_tarsier_without_supported_text_config_does_not_fall_back_to_llava(nested_config):
    from lightllm.models.registry import ModelRegistry

    config = {"model_type": "llava", "architectures": ["TarsierForConditionalGeneration"], **nested_config}
    with pytest.raises(ValueError, match="not supported"):
        ModelRegistry.get_model_config(config)


@pytest.mark.parametrize(
    "architecture,text_type",
    [("AcmeLlavaForConditionalGeneration", "llama"), ("TarsierForConditionalGeneration", "acme")],
)
def test_external_llava_variant_overrides_builtin_fallback(registry, architecture, text_type):
    from lightllm.models.registry import ModelRegistry

    registry._registry["llava"] = list(ModelRegistry._registry["llava"])

    @registry(
        "llava",
        condition=lambda cfg: cfg["architectures"][0] == architecture and cfg["text_config"]["model_type"] == text_type,
    )
    class ExternalModel:
        pass

    config = {"model_type": "llava", "architectures": [architecture], "text_config": {"model_type": text_type}}
    assert registry.get_model_class(config) is ExternalModel
