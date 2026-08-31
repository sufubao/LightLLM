from types import SimpleNamespace

import pytest
import torch

from lightllm.distributed import communication_op
from lightllm.distributed import flashinfer_all_reduce


def test_optional_allreduce_requires_every_rank(monkeypatch):
    group = communication_op.CustomProcessGroup.__new__(communication_op.CustomProcessGroup)
    group.optional_allreduce_group = object()

    def reject_on_peer(enabled, op, group):
        assert enabled.device.type == "cpu"
        assert op == communication_op.dist.ReduceOp.MIN
        enabled.zero_()

    monkeypatch.setattr(communication_op, "has_nvlink", lambda: True)
    monkeypatch.setattr(communication_op.dist, "all_reduce", reject_on_peer)

    assert group._support_custom_allreduce() is False


def test_optional_allreduce_skips_consensus_for_unsupported_world_size(monkeypatch):
    group = communication_op.CustomProcessGroup.__new__(communication_op.CustomProcessGroup)
    group.optional_allreduce_group = None
    monkeypatch.setattr(
        communication_op.dist,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("consensus should not run"),
    )

    assert group._support_custom_allreduce() is False


def test_flashinfer_workspace_failure_disables_every_rank(monkeypatch):
    destroyed = []

    class Workspace:
        def destroy(self):
            destroyed.append(True)

    reducer = flashinfer_all_reduce.FlashInferAllReduce.__new__(flashinfer_all_reduce.FlashInferAllReduce)
    reducer._workspace = None
    reducer._ws_hidden_dim = None
    reducer._ws_dtype = None
    reducer._ws_max_token_num = 0
    reducer.max_workspace_size = 1024
    reducer.world_size = 2
    reducer.rank = 0
    reducer.group = object()
    reducer.disabled = False

    fake_comm = SimpleNamespace(create_allreduce_fusion_workspace=lambda **kwargs: Workspace())

    def reject_on_peer(enabled, op, group):
        enabled.zero_()

    monkeypatch.setattr(flashinfer_all_reduce, "flashinfer_comm", fake_comm)
    monkeypatch.setattr(flashinfer_all_reduce, "TorchDistBackend", lambda group: object())
    monkeypatch.setattr(flashinfer_all_reduce.dist, "all_reduce", reject_on_peer)

    assert reducer._ensure_workspace(hidden_dim=16, dtype=torch.float16) is False
    assert reducer.disabled is True
    assert reducer._workspace is None
    assert destroyed == [True]
