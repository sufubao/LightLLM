"""Run with exp -m 'attention state lifetime' python -m pytest -s <this file>."""

import copy
import gc
import weakref

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import BaseDecodeAttState, BasePrefillAttState
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.prefill_cuda_graph import PrefillCudaGraph


class _PrefillState(BasePrefillAttState):
    def init_state(self):
        pass

    def prefill_att(self, *args, **kwargs):
        pass


class _DecodeState(BaseDecodeAttState):
    def init_state(self):
        pass

    def decode_att(self, *args, **kwargs):
        pass


@pytest.fixture
def no_cycle_gc():
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if enabled:
            gc.enable()


@pytest.mark.parametrize("state_cls", [_PrefillState, _DecodeState])
def test_infer_state_assignment_preserves_identity_without_ownership(no_cycle_gc, state_cls):
    attention = state_cls()
    assert attention.infer_state is None
    owner = InferStateInfo()
    attention.infer_state = owner
    owner_ref = weakref.ref(owner)
    assert attention.infer_state is owner
    del owner
    assert owner_ref() is None
    with pytest.raises(ReferenceError):
        _ = attention.infer_state
    attention.infer_state = None
    assert attention.infer_state is None


@pytest.mark.parametrize("state_cls,slot", [(_PrefillState, "prefill_att_state"), (_DecodeState, "decode_att_state")])
def test_request_releases_state_and_tensor_without_cycle_gc(no_cycle_gc, state_cls, slot):
    owner = InferStateInfo()
    attention = state_cls(infer_state=owner)
    setattr(owner, slot, attention)
    owner.input_ids = torch.empty(4)
    owner_ref, attention_ref, tensor_ref = weakref.ref(owner), weakref.ref(attention), weakref.ref(owner.input_ids)
    assert attention.infer_state is owner
    assert copy.copy(attention).infer_state is owner
    del attention, owner
    assert owner_ref() is None
    assert attention_ref() is None
    assert tensor_ref() is None


@pytest.mark.parametrize(
    "graph_cls,state_cls,slot",
    [
        (CudaGraph, _DecodeState, "decode_att_state"),
        (PrefillCudaGraph, _PrefillState, "prefill_att_state"),
    ],
)
def test_graph_cache_owns_infer_state(no_cycle_gc, graph_cls, state_cls, slot):
    owner = InferStateInfo()
    attention = state_cls(infer_state=owner)
    setattr(owner, slot, attention)
    owner_ref = weakref.ref(owner)
    graph = graph_cls.__new__(graph_cls)
    # Match the cache layouts used by the real capture paths.
    graph.graph = {1: (None, owner, None) if graph_cls is CudaGraph else (owner, [], [], None)}
    del owner
    assert owner_ref() is not None
    assert attention.infer_state is owner_ref()
    graph.graph.clear()
    assert owner_ref() is None
    with pytest.raises(ReferenceError):
        _ = attention.infer_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("state_cls,slot", [(_PrefillState, "prefill_att_state"), (_DecodeState, "decode_att_state")])
def test_cuda_temporaries_do_not_accumulate(no_cycle_gc, state_cls, slot):
    torch.cuda.init()
    baseline = torch.cuda.memory_allocated()
    try:
        for _ in range(8):
            owner = InferStateInfo()
            setattr(owner, slot, state_cls(infer_state=owner))
            owner.input_ids = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            del owner
        retained = torch.cuda.memory_allocated() - baseline
        gc.collect()
        after_gc = torch.cuda.memory_allocated() - baseline
        print(f"{state_cls.__name__}: retained_before_gc={retained}, retained_after_gc={after_gc}")
        assert retained == 0
        assert after_gc == 0
    finally:
        gc.collect()
