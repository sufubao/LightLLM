from types import SimpleNamespace

import pytest

from lightllm.common.basemodel.batch_objs import FinalHiddenOutputPolicy
import lightllm.common.basemodel.spec_collector as spec_collector_module
import lightllm.common.basemodel.spec_manager as spec_manager_module
from lightllm.common.basemodel.spec_collector import (
    FinalHiddenSpecCollector,
    LayerHiddenSpecCollector,
    MtpHeadCollector,
    NoopSpecCollector,
)
from lightllm.common.basemodel.spec_manager import SpecManager


@pytest.fixture(autouse=True)
def _reset_spec_manager():
    SpecManager._instance = None
    yield
    SpecManager._instance = None


def _decode_tokens_per_request(monkeypatch, spec_mode, *, is_draft_model, spec_step=7):
    args = SimpleNamespace(
        spec_mode=spec_mode,
        spec_step=spec_step,
        spec_dynamic_verify=False,
    )
    monkeypatch.setattr(spec_manager_module, "get_env_start_args", lambda: args)
    return SpecManager.get_instance().get_decode_tokens_per_request(is_draft_model)


@pytest.mark.parametrize(
    "spec_mode,is_draft_model,expected",
    [
        (None, False, 1),
        ("eagle3", False, 8),
        ("eagle3", True, 1),
        ("vanilla_with_att", True, 1),
        ("vanilla_no_att", True, 1),
        ("eagle_with_att", True, 1),
        ("eagle_no_att", True, 1),
        ("dspark", True, 7),
        ("dflash", True, 7),
    ],
)
def test_decode_tokens_per_request(monkeypatch, spec_mode, is_draft_model, expected):
    assert _decode_tokens_per_request(monkeypatch, spec_mode, is_draft_model=is_draft_model) == expected


@pytest.mark.parametrize(
    "spec_mode,dynamic_verify,is_draft_model,expected",
    [
        ("vanilla_with_att", False, False, 8),
        ("vanilla_with_att", True, False, 1),
        ("vanilla_with_att", False, True, 1),
        ("vanilla_with_att", True, True, 1),
        ("dspark", True, True, 7),
        ("dflash", True, True, 7),
    ],
)
def test_decode_batch_alignment(monkeypatch, spec_mode, dynamic_verify, is_draft_model, expected):
    args = SimpleNamespace(
        spec_mode=spec_mode,
        spec_step=7,
        spec_dynamic_verify=dynamic_verify,
    )
    monkeypatch.setattr(spec_manager_module, "get_env_start_args", lambda: args)

    assert SpecManager.get_instance().get_decode_batch_alignment(is_draft_model) == expected


@pytest.mark.parametrize(
    "spec_mode,is_draft_model,expected",
    [
        (None, False, 0),
        ("eagle3", False, 7),
        ("eagle3", True, 0),
        ("vanilla_with_att", True, 0),
        ("dspark", True, 6),
        ("dflash", True, 6),
    ],
)
def test_decode_draft_step(monkeypatch, spec_mode, is_draft_model, expected):
    args = SimpleNamespace(spec_mode=spec_mode, spec_step=7, spec_dynamic_verify=False)
    monkeypatch.setattr(spec_manager_module, "get_env_start_args", lambda: args)

    assert SpecManager.get_instance().get_decode_draft_step(is_draft_model) == expected


def test_get_instance_returns_singleton(monkeypatch):
    args = SimpleNamespace(spec_mode="eagle3", spec_step=7, spec_dynamic_verify=False)
    monkeypatch.setattr(spec_manager_module, "get_env_start_args", lambda: args)

    assert SpecManager.get_instance() is SpecManager.get_instance()


@pytest.mark.parametrize(
    "spec_mode,is_draft_model,expected_type",
    [
        (None, False, NoopSpecCollector),
        ("vanilla_with_att", False, FinalHiddenSpecCollector),
        ("eagle3", False, LayerHiddenSpecCollector),
        ("dspark", False, LayerHiddenSpecCollector),
        ("eagle3", True, FinalHiddenSpecCollector),
        ("dspark", True, MtpHeadCollector),
    ],
)
def test_create_spec_collector_selects_implementation(monkeypatch, spec_mode, is_draft_model, expected_type):
    args = SimpleNamespace(
        spec_mode=spec_mode,
        spec_step=7,
        spec_dynamic_verify=False,
        spec_draft_model_dir=["/models/draft"],
    )
    monkeypatch.setattr(spec_manager_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(spec_collector_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(
        spec_collector_module.PretrainedConfig,
        "get_config_dict",
        lambda _: ({"target_layer_ids": [0]}, {}),
    )
    model = SimpleNamespace(is_spec_draft_model=is_draft_model, layers_num=2)

    prototype = SpecManager.get_instance().create_spec_collector(model=model)
    collector = prototype.new_instance()

    assert isinstance(prototype, expected_type)
    assert isinstance(collector, expected_type)
    assert collector is not prototype


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "path,mode,expected",
    [
        ("deepseek2.model:Deepseek2TpPartModel", "vanilla_with_att", FinalHiddenOutputPolicy.POST_NORM),
        ("deepseek_mtp.model:Deepseek3MTPModel", "vanilla_with_att", FinalHiddenOutputPolicy.POST_NORM),
        ("glm4_moe_lite.model:Glm4MoeLiteTpPartModel", "vanilla_with_att", FinalHiddenOutputPolicy.POST_NORM),
        ("glm4_moe_lite_mtp.model:Glm4MoeLiteMTPModel", "vanilla_with_att", FinalHiddenOutputPolicy.PRE_NORM),
        ("glm5_next.model:Glm5NextTpPartModel", "eagle_with_att", FinalHiddenOutputPolicy.POST_NORM),
        ("glm5_next_mtp.model:Glm5NextMTPModel", "eagle_with_att", FinalHiddenOutputPolicy.POST_NORM),
        ("qwen3_5.model:Qwen3_5TpPartModel", "vanilla_with_att", FinalHiddenOutputPolicy.POST_NORM),
        ("qwen3_5_mtp.model:Qwen3_5MTPModel", "vanilla_with_att", FinalHiddenOutputPolicy.POST_NORM),
        ("qwen3_5_moe.model:Qwen3_5MOETpPartModel", "eagle_with_att", FinalHiddenOutputPolicy.POST_NORM),
        ("qwen3_5_moe_mtp.model:Qwen3_5MoeMTPModel", "eagle_with_att", FinalHiddenOutputPolicy.POST_NORM),
        ("mistral.model:MistralTpPartModel", "vanilla_no_att", FinalHiddenOutputPolicy.PRE_NORM),
        ("mistral_mtp.model:MistralMTPModel", "vanilla_no_att", FinalHiddenOutputPolicy.PRE_NORM),
        ("qwen3_moe.model:Qwen3MOEModel", "eagle_no_att", FinalHiddenOutputPolicy.PRE_NORM),
        ("qwen3_moe_mtp.model:Qwen3MOEMTPModel", "eagle_no_att", FinalHiddenOutputPolicy.PRE_NORM),
        ("qwen3_eagle.model:Qwen3EagleModel", "eagle3", FinalHiddenOutputPolicy.PRE_NORM),
        ("qwen3_dflash.model:Qwen3DFlashModel", "dflash", FinalHiddenOutputPolicy.NONE),
        ("qwen3_5_dflash.model:Qwen3_5DFlashModel", "dflash", FinalHiddenOutputPolicy.NONE),
        ("qwen3_dspark.model:Qwen3DSparkModel", "dspark", FinalHiddenOutputPolicy.NONE),
        ("qwen3_5_dspark.model:Qwen3_5DSparkModel", "dspark", FinalHiddenOutputPolicy.NONE),
    ],
)
def test_model_hidden_output(monkeypatch, path, mode, expected, enabled):
    from importlib import import_module

    module, name = path.split(":")
    cls = getattr(import_module("lightllm.models." + module), name)
    model = cls.__new__(cls)
    monkeypatch.setattr(
        spec_manager_module, "get_env_start_args", lambda: SimpleNamespace(spec_mode=mode if enabled else None)
    )
    policy = SpecManager.get_instance().get_final_hidden_output_policy(model)
    assert policy is (expected if enabled else FinalHiddenOutputPolicy.NONE)


@pytest.mark.parametrize("mode", ["eagle3", "dflash", "dspark"])
@pytest.mark.parametrize(
    "path",
    ["deepseek2.model:Deepseek2TpPartModel", "glm5_next.model:Glm5NextTpPartModel", "qwen3_5.model:Qwen3_5TpPartModel"],
)
def test_intermediate_hidden_output(monkeypatch, mode, path):
    from importlib import import_module

    args = SimpleNamespace(spec_mode=mode, spec_draft_model_dir=["/models/draft"])
    monkeypatch.setattr(spec_manager_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(spec_collector_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(
        spec_collector_module.PretrainedConfig, "get_config_dict", lambda _: ({"target_layer_ids": [0]}, {})
    )
    module, name = path.split(":")
    cls = getattr(import_module("lightllm.models." + module), name)
    model = cls.__new__(cls)
    model.layers_num = 2
    collector = SpecManager.get_instance().create_spec_collector(model).new_instance()
    assert isinstance(collector, LayerHiddenSpecCollector)
    assert SpecManager.get_instance().get_final_hidden_output_policy(model) is FinalHiddenOutputPolicy.NONE
