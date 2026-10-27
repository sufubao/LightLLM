import sys
from types import SimpleNamespace

import pytest
import yaml

from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization import registry
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_server import launch_server
from lightllm.server.core.objs.start_args_type import StartArgs


INVALID_METHOD = "triton-fp8a8w8-pertensor"


@pytest.mark.parametrize("option", ["--quant_type", "--vit_quant_type"])
def test_cli_rejects_unknown_quantization(option, capsys):
    with pytest.raises(SystemExit) as exc:
        make_argument_parser().parse_args([option, INVALID_METHOD])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert option in error and INVALID_METHOD in error
    assert "fp8w8a8-pt-triton" in error


def test_lazy_method_is_imported_only_when_requested(monkeypatch):
    factory = registry.QuantMethodFactory()
    factory.register(["test", "alias"])("test_module.Method")
    imports = []

    class Method:
        pass

    def import_module(name, package):
        imports.append((name, package))
        return SimpleNamespace(Method=Method)

    monkeypatch.setattr(registry, "import_module", import_module)
    factory.validate("test")
    assert not imports
    assert isinstance(factory.get("test"), Method)
    assert isinstance(factory.get("test"), Method)
    assert isinstance(factory.get("alias"), Method)
    assert imports == [(".test_module", registry.__package__)] * 2


@pytest.mark.parametrize("option", ["quant_type", "vit_quant_type"])
def test_programmatic_launch_rejects_unknown_method_before_loading_startup(monkeypatch, option):
    monkeypatch.setitem(sys.modules, "lightllm.server.api_start", None)
    with pytest.raises(ValueError, match=f"--{option}.*{INVALID_METHOD}.*fp8w8a8-pt-triton"):
        launch_server(StartArgs(**{option: INVALID_METHOD}))


@pytest.mark.parametrize("option", ["quant_cfg", "vit_quant_cfg"])
@pytest.mark.parametrize("mixed", [False, True])
def test_launch_rejects_unknown_yaml_method_before_loading_startup(tmp_path, monkeypatch, option, mixed):
    path = tmp_path / "quant.yaml"
    data = {"quant_type": INVALID_METHOD}
    if mixed:
        data = {"quant_type": "none", "mix_bits": [{"name": "q_proj", "quant_type": INVALID_METHOD}]}
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setitem(sys.modules, "lightllm.server.api_start", None)
    with pytest.raises(ValueError) as exc:
        launch_server(StartArgs(**{option: str(path)}))
    error = str(exc.value)
    assert f"--{option}" in error and str(path) in error and INVALID_METHOD in error
    if mixed:
        assert "mix_bits[0]" in error


def test_quantcfg_rejects_unknown_method_before_constructing_layers():
    with pytest.raises(ValueError, match=INVALID_METHOD):
        Quantcfg({"n_layer": 1}, quant_type=INVALID_METHOD)


@pytest.mark.parametrize("expert_dtype", [None, "fp8"])
def test_mixed_config_preserves_quantization_precedence(tmp_path, expert_dtype):
    path = tmp_path / "quant.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "quant_type": "fp8w8a8-pt",
                "mix_bits": [{"name": "fused_moe", "quant_type": "none", "layer_nums": [0]}],
            }
        )
    )
    config = Quantcfg({"n_layer": 2, "expert_dtype": "fp8"}, "w8a8", str(path), expert_dtype=expert_dtype)
    assert config.get_quant_type(0, "q_proj") == "fp8w8a8-pt"
    assert config.get_quant_type(0, "fused_moe") == ("none" if expert_dtype is None else "fp8w8a8-b128-deepgemm")
    assert config.get_quant_type(1, "fused_moe") == "fp8w8a8-b128-deepgemm"


def test_no_quantization_still_loads_through_registry(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    method = Quantcfg({"n_layer": 1}).get_quant_method(0, "q_proj")
    assert method.method_name == "none"
    assert method.hf_quantization_config is None
