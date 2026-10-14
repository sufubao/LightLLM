from types import SimpleNamespace

import pytest

from lightllm.common.basemodel.batch_objs import FinalHiddenPolicy
import lightllm.common.basemodel.mtp_collector as mtp_collector_module
import lightllm.common.basemodel.mtp_manager as mtp_manager_module
from lightllm.common.basemodel.mtp_collector import (
    FinalHiddenMtpCollector,
    LayerHiddenMtpCollector,
    MtpHeadCollector,
    NoopMtpCollector,
)
from lightllm.common.basemodel.mtp_manager import MtpManager


@pytest.fixture(autouse=True)
def _reset_mtp_manager():
    MtpManager._instance = None
    yield
    MtpManager._instance = None


def _decode_tokens_per_request(monkeypatch, spec_mode, *, is_draft_model, mtp_step=7):
    args = SimpleNamespace(
        mtp_mode=spec_mode,
        mtp_step=mtp_step,
        mtp_dynamic_verify=False,
    )
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)
    return MtpManager.get_instance().get_decode_tokens_per_request(is_draft_model)


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
        mtp_mode=spec_mode,
        mtp_step=7,
        mtp_dynamic_verify=dynamic_verify,
    )
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)

    assert MtpManager.get_instance().get_decode_batch_alignment(is_draft_model) == expected


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
    args = SimpleNamespace(mtp_mode=spec_mode, mtp_step=7, mtp_dynamic_verify=False)
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)

    assert MtpManager.get_instance().get_decode_draft_step(is_draft_model) == expected


def test_get_instance_returns_singleton(monkeypatch):
    args = SimpleNamespace(mtp_mode="eagle3", mtp_step=7, mtp_dynamic_verify=False)
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)

    assert MtpManager.get_instance() is MtpManager.get_instance()


@pytest.mark.parametrize(
    "spec_mode,is_draft_model,expected_type",
    [
        (None, False, NoopMtpCollector),
        ("vanilla_with_att", False, FinalHiddenMtpCollector),
        ("eagle3", False, LayerHiddenMtpCollector),
        ("dspark", False, LayerHiddenMtpCollector),
        ("eagle3", True, FinalHiddenMtpCollector),
        ("dspark", True, MtpHeadCollector),
    ],
)
def test_create_mtp_collector_selects_implementation(monkeypatch, spec_mode, is_draft_model, expected_type):
    args = SimpleNamespace(
        mtp_mode=spec_mode,
        mtp_step=7,
        mtp_dynamic_verify=False,
        mtp_draft_model_dir=["/models/draft"],
    )
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(mtp_collector_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(
        mtp_collector_module.PretrainedConfig,
        "get_config_dict",
        lambda _: ({"target_layer_ids": [0]}, {}),
    )
    model = SimpleNamespace(is_mtp_draft_model=is_draft_model, layers_num=2)

    prototype = MtpManager.get_instance().create_mtp_collector(model=model)
    collector = prototype.new_instance()

    assert isinstance(prototype, expected_type)
    assert isinstance(collector, expected_type)
    assert collector is not prototype


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "path,mode,expected",
    [
        ("deepseek2.model:Deepseek2TpPartModel", "vanilla_with_att", FinalHiddenPolicy.POST_NORM),
        ("deepseek_mtp.model:Deepseek3MTPModel", "vanilla_with_att", FinalHiddenPolicy.POST_NORM),
        ("glm4_moe_lite.model:Glm4MoeLiteTpPartModel", "vanilla_with_att", FinalHiddenPolicy.POST_NORM),
        ("glm4_moe_lite_mtp.model:Glm4MoeLiteMTPModel", "vanilla_with_att", FinalHiddenPolicy.PRE_NORM),
        ("glm5_next.model:Glm5NextTpPartModel", "eagle_with_att", FinalHiddenPolicy.POST_NORM),
        ("glm5_next_mtp.model:Glm5NextMTPModel", "eagle_with_att", FinalHiddenPolicy.POST_NORM),
        ("qwen3_5.model:Qwen3_5TpPartModel", "vanilla_with_att", FinalHiddenPolicy.POST_NORM),
        ("qwen3_5_mtp.model:Qwen3_5MTPModel", "vanilla_with_att", FinalHiddenPolicy.POST_NORM),
        ("qwen3_5_moe.model:Qwen3_5MOETpPartModel", "eagle_with_att", FinalHiddenPolicy.POST_NORM),
        ("qwen3_5_moe_mtp.model:Qwen3_5MoeMTPModel", "eagle_with_att", FinalHiddenPolicy.POST_NORM),
        ("mistral.model:MistralTpPartModel", "vanilla_no_att", FinalHiddenPolicy.PRE_NORM),
        ("mistral_mtp.model:MistralMTPModel", "vanilla_no_att", FinalHiddenPolicy.PRE_NORM),
        ("qwen3_moe.model:Qwen3MOEModel", "eagle_no_att", FinalHiddenPolicy.PRE_NORM),
        ("qwen3_moe_mtp.model:Qwen3MOEMTPModel", "eagle_no_att", FinalHiddenPolicy.PRE_NORM),
        ("qwen3_eagle.model:Qwen3EagleModel", "eagle3", FinalHiddenPolicy.PRE_NORM),
        ("qwen3_dflash.model:Qwen3DFlashModel", "dflash", FinalHiddenPolicy.NONE),
        ("qwen3_5_dflash.model:Qwen3_5DFlashModel", "dflash", FinalHiddenPolicy.NONE),
        ("qwen3_dspark.model:Qwen3DSparkModel", "dspark", FinalHiddenPolicy.NONE),
        ("qwen3_5_dspark.model:Qwen3_5DSparkModel", "dspark", FinalHiddenPolicy.NONE),
    ],
)
def test_model_hidden_output(monkeypatch, path, mode, expected, enabled):
    from importlib import import_module

    module, name = path.split(":")
    cls = getattr(import_module("lightllm.models." + module), name)
    model = cls.__new__(cls)
    monkeypatch.setattr(
        mtp_manager_module, "get_env_start_args", lambda: SimpleNamespace(mtp_mode=mode if enabled else None)
    )
    policy = MtpManager.get_instance().get_final_hidden_policy(model)
    assert policy is (expected if enabled else FinalHiddenPolicy.NONE)


@pytest.mark.parametrize("mode", ["eagle3", "dflash", "dspark"])
@pytest.mark.parametrize(
    "path",
    ["deepseek2.model:Deepseek2TpPartModel", "glm5_next.model:Glm5NextTpPartModel", "qwen3_5.model:Qwen3_5TpPartModel"],
)
def test_intermediate_hidden_output(monkeypatch, mode, path):
    from importlib import import_module

    args = SimpleNamespace(mtp_mode=mode, mtp_draft_model_dir=["/models/draft"])
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(mtp_collector_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(
        mtp_collector_module.PretrainedConfig, "get_config_dict", lambda _: ({"target_layer_ids": [0]}, {})
    )
    module, name = path.split(":")
    cls = getattr(import_module("lightllm.models." + module), name)
    model = cls.__new__(cls)
    model.layers_num = 2
    collector = MtpManager.get_instance().create_mtp_collector(model).new_instance()
    assert isinstance(collector, LayerHiddenMtpCollector)
    assert MtpManager.get_instance().get_final_hidden_policy(model) is FinalHiddenPolicy.NONE
