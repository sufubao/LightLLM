import argparse

import pytest

from lightllm.server.api_cli import add_cli_args
from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


def test_mtp_requires_cuda_graph(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    args = StartArgs(spec_mode="vanilla_no_att", disable_cudagraph=True)

    with pytest.raises(AssertionError, match="only supported on Prefill nodes"):
        _launch_subprocesses(args)


def test_mtp_prefill_still_requires_positive_step(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_fused_shared_experts", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.set_unique_server_name", lambda args: None)
    args = StartArgs(
        run_mode="prefill",
        spec_mode="dspark",
        spec_draft_model_dir=["draft"],
        spec_step=0,
        disable_cudagraph=True,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    with pytest.raises(AssertionError):
        _launch_subprocesses(args)


@pytest.mark.parametrize("prefix", ["spec", "mtp"])
def test_spec_cli_aliases_share_canonical_start_args(prefix):
    parser = add_cli_args(argparse.ArgumentParser())
    parsed = parser.parse_args(
        [
            f"--{prefix}_mode",
            "eagle3",
            f"--{prefix}_step",
            "3",
            f"--{prefix}_draft_model_dir",
            "draft",
            f"--{prefix}_dynamic_verify",
        ]
    )
    args = StartArgs(**vars(parsed))
    assert args.spec_mode == "eagle3"
    assert args.spec_step == 3
    assert args.spec_draft_model_dir == ["draft"]
    assert args.spec_dynamic_verify is True
    assert not any(name.startswith("mtp_") for name in vars(parsed))
