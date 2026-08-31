from types import SimpleNamespace

from lightllm.utils import backend_validator
from lightllm.utils import dist_check_utils


def test_jit_backends_are_skipped_without_cuda_compiler(monkeypatch):
    monkeypatch.setattr("lightllm.utils.device_utils.has_cuda_compiler", lambda: False)

    assert backend_validator._quick_validate("flashinfer") is False
    assert backend_validator._quick_validate("flashqla") is False
    assert backend_validator._quick_validate("fa3") is None
    assert backend_validator._quick_validate("triton") is None


def test_allreduce_check_uses_runtime_fallback(monkeypatch):
    monkeypatch.setattr(dist_check_utils, "has_cuda_compiler", lambda: False)
    args = SimpleNamespace(
        hardware_platform="cuda",
        disable_flashinfer_allreduce=False,
        disable_symm_mem_allreduce=False,
    )

    dist_check_utils.auto_configure_allreduce_flags_from_args(args)

    assert args.disable_flashinfer_allreduce is True
    assert args.disable_symm_mem_allreduce is False
