import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("run_mode", ["normal", "decode"])
def test_mtp_requires_cuda_graph_outside_prefill(monkeypatch, run_mode):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    args = StartArgs(run_mode=run_mode, mtp_mode="vanilla_no_att", disable_cudagraph=True)

    with pytest.raises(AssertionError, match="--disable_cudagraph is not supported"):
        _launch_subprocesses(args)


def test_mtp_prefill_allows_disabling_cuda_graph(monkeypatch):
    class ValidationPassed(Exception):
        pass

    def stop_after_mtp_validation(_args):
        raise ValidationPassed

    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", stop_after_mtp_validation)
    args = StartArgs(
        run_mode="prefill",
        mtp_mode="dspark",
        mtp_step=1,
        disable_cudagraph=True,
    )

    with pytest.raises(ValidationPassed):
        _launch_subprocesses(args)
