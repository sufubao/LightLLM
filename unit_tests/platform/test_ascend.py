from lightllm.core.formats.kv.spec import DenseKVStateSpec
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
import torch

from lightllm.core.compute.attention.dense.ascend import (
    AscendAttentionBackend,
    AscendGraphBinding,
    BoundAscendAttention,
)
from lightllm.core.compute.attention.batch import AttentionBatch
from lightllm.core.compute.attention.dense.backend import DenseStaticParams
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.core.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.core.contracts.state import FormatSpec
from lightllm.core.device.graph import GraphExecution
from lightllm.platform.ascend import validate_config
from lightllm.server.core.objs.start_args_type import StartArgs


def config(**kwargs):
    return StartArgs(
        hardware_platform="ascend",
        page_size=32,
        enable_torch_fallback=True,
        disable_symm_mem_allreduce=True,
        disable_flashinfer_allreduce=True,
        **kwargs,
    )


def attention(phase="decode"):
    backend = AscendAttentionBackend(phase)
    spec = DenseAttentionSpec(4, 2, 64, dtype=torch.float16)
    fmt = FormatSpec("plain", torch.float16, "token_head_dim", ("k", "v"), 32)
    caps = DeviceCaps("npu", 0, platform="ascend", architecture="Ascend910B")
    limits = ExecutionLimits(32 * 1024 * 1024, 2, 64, graph_phases=("decode",))
    return backend, spec, fmt, caps, limits


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_native_attention_capabilities(phase):
    backend, spec, fmt, caps, limits = attention(phase)
    assert backend.probe(spec, fmt, caps, limits).offers
    cases = [
        (spec, fmt, DeviceCaps("cuda", 0, (9, 0)), limits),
        (spec, replace(fmt, encoding="int8"), caps, limits),
        (spec, replace(fmt, page_size=16), caps, limits),
        (replace(spec, window=(31, 0)), fmt, caps, limits),
    ]
    cases += [
        (spec, fmt, caps, replace(limits, **change))
        for change in (
            dict(max_verify_tokens=2),
            dict(graph_phases=("prefill",)),
            dict(transfer_paths=("cpu_cache",)),
            dict(microbatch_count=2),
            dict(expert_parallel=True),
            dict(dynamic_verify=True),
        )
    ]
    for values in cases:
        result = backend.probe(*values)
        assert result.reason and not result.offers


@pytest.mark.parametrize(
    "option",
    [
        "enable_cpu_cache",
        "enable_ep_moe",
        "enable_prefill_cudagraph",
        "enable_decode_microbatch_overlap",
        "enable_multimodal",
    ],
)
def test_unsupported_modes_are_rejected_before_vendor_import(option):
    with pytest.raises(ValueError, match=option):
        validate_config(config(**{option: True}))


def bound_attention():
    backend, spec, fmt, caps, limits = attention()
    (offer,) = backend.probe(spec, fmt, caps, limits).offers
    view = SimpleNamespace(format=fmt, fields=(torch.zeros(128, 2, 64), torch.zeros(128, 2, 64)))
    bound = BoundAscendAttention(
        offer, view, DenseStaticParams(spec, torch.empty), torch.empty(1024, dtype=torch.uint8)
    )
    return bound


def batch(lengths=(3, 5)):
    return AttentionBatch(
        2,
        1,
        max(lengths),
        0,
        torch.arange(128, dtype=torch.int32).view(2, 64),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        None,
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32),
    )


def test_graph_updates_host_lengths_after_submit_and_waits_before_copy(monkeypatch):
    bound = bound_attention()
    metadata = bound.create_metadata(batch())
    torch.testing.assert_close(metadata.page_table, torch.tensor([[0, 1], [2, 3]], dtype=torch.int32))
    order, capturing, observed = [], [False], []
    stream = Mock()
    stream.synchronize.side_effect = lambda: order.append("wait")
    event = Mock()
    event.record.side_effect = lambda s: order.append("event")
    api = SimpleNamespace(
        is_current_stream_capturing=lambda: capturing[0],
        current_stream=lambda: stream,
        ExternalEvent=lambda: event,
        graph_task_group_begin=lambda s: order.append("capture_begin"),
        graph_task_group_end=lambda s: "task",
        graph_task_update_begin=lambda s, h: order.append("update"),
        graph_task_update_end=lambda s: order.append("update_end"),
    )
    workspace = Mock(return_value=torch.empty(128, dtype=torch.uint8))

    def execute(**kwargs):
        kwargs["out"][0].copy_(kwargs["query"])
        observed.append(kwargs["actual_seq_lengths_kv"].tolist())

    vendor = SimpleNamespace(
        npu_fused_infer_attention_score=SimpleNamespace(out=execute),
        _npu_fused_infer_attention_score_get_max_workspace=workspace,
        _C=SimpleNamespace(_weak_ref_tensor=lambda t: t),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", vendor)
    monkeypatch.setattr(torch, "npu", api, raising=False)
    q = torch.randn(2, 4, 64)
    bound.run(q, metadata)
    assert workspace.call_args.kwargs["actual_seq_lengths_kv"].tolist() == [64, 64]
    capturing[0] = True
    bound.run(q, metadata)
    capturing[0] = False
    runtime = SimpleNamespace(
        api=api,
        create_stream=lambda: stream,
        use_stream=lambda s: nullcontext(),
        synchronize=lambda: order.append("sync"),
    )
    binding = bound.bind_graph(metadata, runtime)

    def copy(new):
        order.append("copy")
        metadata.copy_for_graph(new)

    graph = Mock()
    graph.replay.side_effect = lambda: order.append("submit")
    execution = GraphExecution(graph, (SimpleNamespace(copy_for_cuda_graph=copy),), (binding,))
    order.clear()
    execution.replay(bound.create_metadata(batch((6, 9))))
    assert order == ["wait", "copy", "submit", "update", "update_end", "event"]
    assert observed[-1] == [6, 9]
    assert workspace.call_count == 1
    execution.reset()
    assert not metadata.tasks
    graph.reset.assert_called_once()


def test_kv_writer_and_tp_embedding_use_native_views():
    from lightllm.core.formats.kv.ascend import AscendKVWriter
    from lightllm.common.basemodel.ascend_weights import embedding

    k, v, kv = torch.zeros(8, 2, 4), torch.zeros(8, 2, 4), torch.randn(3, 4, 4)
    writer = AscendKVWriter(SimpleNamespace(fields=(k, v)))
    writer(token_indices=torch.tensor([2, 5, 7], dtype=torch.int32), kv=kv)
    torch.testing.assert_close(k[[2, 5, 7]], kv[:, :2])
    torch.testing.assert_close(v[[2, 5, 7]], kv[:, 2:])
    weight, ids = torch.randn(8, 4), torch.tensor([0, 3, 4, 7])
    result = embedding(ids, weight=weight[:4], start=0, end=4) + embedding(ids, weight=weight[4:], start=4, end=8)
    torch.testing.assert_close(result, weight[ids])


def test_factory_injects_runtime_and_native_bindings(monkeypatch):
    from lightllm.core.device import ascend
    from lightllm.platform.ascend import create_context

    runtime = SimpleNamespace(caps=DeviceCaps("npu", 0, platform="ascend"))
    constructor = Mock(return_value=runtime)
    monkeypatch.setattr(ascend, "AscendRuntime", constructor)
    context = create_context(index=3, config=config())
    constructor.assert_called_once_with(3)
    assert context.device is runtime
    assert {b.name for b in context.providers.attention_backends(("prefill", "decode"))} == {
        "ascend.prefill",
        "ascend.decode",
    }
    assert set(context.providers.operations) == {"rms_norm", "rope", "swiglu"}
    assert context.providers.weight_finalizers


def test_runtime_uses_npu_graph_and_vendor_borrowing(monkeypatch):
    from lightllm.core.device import ascend

    api = Mock()
    api.get_device_name.return_value = "Ascend910B"
    api.mem_get_info.return_value = (800, 1000)
    weak = Mock()
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(_C=SimpleNamespace(_weak_ref_tensor=weak)))
    device = SimpleNamespace(type="npu", index=2)
    monkeypatch.setattr(ascend, "torch", SimpleNamespace(npu=api, device=lambda *args: device))
    runtime = ascend.AscendRuntime(2)
    assert runtime.caps.platform == "ascend" and not runtime.caps.is_nvidia
    assert runtime.distributed_backend == "hccl"
    assert runtime.workspace_budget(0.9) == 700
    assert runtime.decode_warmup_length(8192) == 8192
    graph, pool = runtime.create_graph(), runtime.graph_pool()
    runtime.capture_graph(graph, pool=pool)
    api.graph.assert_called_once_with(graph, pool=pool)
    tensor = object()
    assert runtime.borrow_tensor(tensor) is weak.return_value
    weak.assert_called_once_with(tensor)


def test_storage_initializes_padding_without_erasing_live_state():
    from lightllm.core.contracts.state import StateSpec
    from lightllm.core.storage.ascend import AscendStateStore
    from lightllm.common.attention_state import AttentionStateManager

    manager = AttentionStateManager(request_capacity=2)
    declaration = StateSpec("main", "kv", 0, (0,), DenseKVStateSpec(kv_heads=2, head_dim=64))
    store = AscendStateStore((declaration,), attention()[2], manager)
    manager.finish_state_registration()
    state = manager.get_full_attention_states()[0]
    state.device = torch.device("cpu")
    state.allocate_storage(64)
    view = store.views()[declaration.key]
    assert all(torch.count_nonzero(t) == 0 for t in view.fields)
    view.fields[0][0].fill_(5)
    assert store.views()[declaration.key].fields[0][0].eq(5).all()
