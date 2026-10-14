from types import SimpleNamespace

import torch

import lightllm.common.basemodel.spec_collector as spec_collector_module
from lightllm.common.basemodel.spec_collector import LayerHiddenSpecCollector
from lightllm.common.basemodel.prefill_cuda_graph import PrefillCudaGraph


class _GraphInferState:
    def __init__(self, spec_collector):
        self.spec_collector = spec_collector
        self.input_ids = torch.empty(4, dtype=torch.int64)
        self.copied_from = None
        self.replayed_with = None

    def copy_for_prefill_cuda_graph(self, new_infer_state):
        self.copied_from = new_infer_state

    def prefill_replay(self, new_infer_state):
        self.replayed_with = new_infer_state


class _IdentityPreInfer:
    @staticmethod
    def _tpsp_allgather(input, infer_state):
        del infer_state
        return input


def _create_layer_collector(monkeypatch):
    monkeypatch.setattr(
        spec_collector_module,
        "get_env_start_args",
        lambda: SimpleNamespace(spec_draft_model_dir=["/models/draft"]),
    )
    monkeypatch.setattr(
        spec_collector_module.PretrainedConfig,
        "get_config_dict",
        lambda _: ({"target_layer_ids": [0]}, {}),
    )
    model = SimpleNamespace(layers_num=2, pre_infer=_IdentityPreInfer())
    return LayerHiddenSpecCollector(model=model)


def test_replay_restores_captured_hidden_state_into_request_collector(monkeypatch):
    graph_collector = _create_layer_collector(monkeypatch)
    graph_collector.add(layer_index=0, hidden=torch.randn(2, 3))
    request_collector = graph_collector.new_instance()
    graph_infer_state = _GraphInferState(spec_collector=graph_collector)
    request_infer_state = SimpleNamespace(
        input_ids=torch.empty(4, dtype=torch.int64),
        spec_collector=request_collector,
    )
    graph_output = torch.randn(2, 3)
    prefill_graph = PrefillCudaGraph.__new__(PrefillCudaGraph)
    prefill_graph.graph = {4: (graph_infer_state, [], [graph_output], graph_collector)}

    outputs = prefill_graph._replay(input_tensors=[], infer_state=request_infer_state)

    assert len(outputs) == 1
    assert outputs[0] is graph_output
    assert request_infer_state.spec_collector is request_collector
    assert request_collector.layer_hiddens is not graph_collector.layer_hiddens
    assert request_collector.layer_hiddens[0] is graph_collector.layer_hiddens[0]
    assert graph_infer_state.copied_from is request_infer_state
    assert graph_infer_state.replayed_with is request_infer_state


def test_first_replay_replaces_capture_collector_with_runtime_instance(monkeypatch):
    graph_collector = _create_layer_collector(monkeypatch)
    graph_collector.add(layer_index=0, hidden=torch.randn(2, 3))
    graph_infer_state = _GraphInferState(spec_collector=graph_collector)
    graph_output = torch.randn(2, 3)
    prefill_graph = PrefillCudaGraph.__new__(PrefillCudaGraph)
    prefill_graph.graph = {4: (graph_infer_state, [], [graph_output], graph_collector)}

    prefill_graph._replay(input_tensors=[], infer_state=graph_infer_state)

    assert graph_infer_state.spec_collector is not graph_collector
    assert graph_infer_state.spec_collector.layer_hiddens[0] is graph_collector.layer_hiddens[0]
