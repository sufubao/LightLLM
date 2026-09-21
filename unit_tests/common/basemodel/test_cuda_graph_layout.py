from types import SimpleNamespace

import pytest

import lightllm.common.basemodel.cuda_graph as cuda_graph_module
import lightllm.common.basemodel.mtp_manager as mtp_manager_module
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.basemodel.mtp_manager import MtpManager


@pytest.fixture(autouse=True)
def _graph_args(monkeypatch):
    args = SimpleNamespace(
        enable_decode_microbatch_overlap=False,
        enable_tpsp_mix_mode=False,
        enable_torch_memory_saver=False,
    )
    monkeypatch.setattr(cuda_graph_module, "get_env_start_args", lambda: args)
    return args


def _batch_sizes(max_batch_size, batch_stride=1):
    physical_max_batch_size = max_batch_size * batch_stride
    graph = CudaGraph(
        batch_step_size_before_split=batch_stride,
        split_batch_size=4 * batch_stride,
        batch_step_size_after_split=2 * batch_stride,
        max_batch_size=physical_max_batch_size,
    )
    return graph.cuda_graph_batch_sizes


def test_dynamic_schedule_uses_compacted_physical_rows(_graph_args):
    assert _batch_sizes(max_batch_size=128) == [1, 2, 3, 4, *range(6, 129, 2)]


def test_public_static_schedule_preserves_original_static_mtp_default(_graph_args):
    assert CudaGraph.gen_cuda_graph_batch_sizes(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=32,
    ) == [
        8,
        16,
        24,
        32,
    ]


def test_instance_and_public_static_schedule_match(_graph_args):
    graph = CudaGraph(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=128,
    )

    assert graph.cuda_graph_batch_sizes == CudaGraph.gen_cuda_graph_batch_sizes(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=graph.max_batch_size,
        tp_world_size=graph.tp_world_size,
    )


def test_batch_step_size_before_split_controls_capture_range(_graph_args):
    assert _batch_sizes(max_batch_size=4, batch_stride=8) == [8, 16, 24, 32]


def test_batch_step_size_after_split_controls_capture_range(_graph_args):
    assert _batch_sizes(max_batch_size=8, batch_stride=7) == [
        7,
        14,
        21,
        28,
        42,
        56,
    ]


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("mtp_step", [1, 2])
@pytest.mark.parametrize("dynamic,is_draft", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("overlap", [False, True])
def test_mtp_tpsp_layout(monkeypatch, _graph_args, tp_size, mtp_step, dynamic, is_draft, overlap):
    args = _graph_args
    args.enable_tpsp_mix_mode = True
    args.enable_decode_microbatch_overlap = overlap
    args.mtp_mode = "eagle_with_att"
    args.mtp_step = mtp_step
    args.mtp_dynamic_verify = dynamic
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args = args
    model.mtp_manager = MtpManager()
    model.tp_world_size_ = tp_size
    model.enable_tpsp_mix_mode = True
    model.is_mtp_draft_model = is_draft
    model._init_decode_batch_layout(max_requests=7)

    width = 1 if dynamic or is_draft else mtp_step + 1
    assert model.decode_batch_alignment % width == 0
    assert model.decode_batch_alignment % tp_size == 0
    if width == 1:
        assert model.decode_batch_alignment == tp_size
    assert model.graph_max_batch_size % model.decode_batch_alignment == 0
    logical_max = 7 // 2 if overlap else 7
    physical_max = logical_max * (1 if is_draft else mtp_step + 1)
    assert physical_max <= model.graph_max_batch_size < physical_max + model.decode_batch_alignment

    sizes = CudaGraph.gen_cuda_graph_batch_sizes(
        batch_step_size_before_split=width,
        split_batch_size=4 * width,
        batch_step_size_after_split=2 * width,
        max_batch_size=model.graph_max_batch_size,
        tp_world_size=tp_size,
    )
    assert sizes[-1] == model.graph_max_batch_size
    assert all(size % width == 0 and size % tp_size == 0 for size in sizes)
    if tp_size == 8 and width == 3:
        assert sizes == [24]
