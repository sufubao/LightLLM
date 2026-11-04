from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.core.compute.moe import flashinfer_trtllm
from lightllm.core.compute.moe.flashinfer_trtllm import FlashInferTrtllmMoeBackend
from lightllm.core.compute.moe.spec import MoeSpec
from lightllm.core.contracts.device import DeviceCaps
from lightllm.core.contracts.provider import MoeOffer
from unit_tests.common.execution.test_moe_compute import format_for, quantize_blocks, route_reference


@pytest.mark.parametrize(
    "caps", [DeviceCaps("cpu"), DeviceCaps("cuda", 0, (9, 0)), DeviceCaps("cuda", 0, (10, 0), "maca")]
)
def test_unsupported_device_rejects_before_importing_flashinfer(caps, monkeypatch):
    monkeypatch.setattr(flashinfer_trtllm, "import_module", lambda name: pytest.fail("optional dependency imported"))
    spec = MoeSpec(256, 256, 8, 2, torch.bfloat16)
    result = FlashInferTrtllmMoeBackend().probe(spec, format_for(spec.dtype), caps)
    assert not result.offers and "Blackwell" in result.reason


@pytest.mark.parametrize("quantized", [False, True])
def test_binding_converts_layout_once_and_keeps_inplace_execution(quantized, monkeypatch):
    spec = MoeSpec(256, 256, 2, 1, torch.bfloat16)
    offer = MoeOffer("flashinfer_trtllm", spec, format_for(spec.dtype, quantized))
    gate_up = torch.arange(2 * 512 * 256, dtype=torch.float32).reshape(2, 512, 256) % 11
    gate_up = gate_up.to(offer.format.dtype)
    down = torch.ones(2, 256, 256, dtype=offer.format.dtype)
    scales = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    prepared, calls = [], []

    def prepare(w1, w2, **kwargs):
        prepared.append((w1, w2))
        return dict(gemm1_weights=w1, gemm2_weights=w2)

    def kernel(**kwargs):
        calls.append(kwargs)
        assert kwargs["topk_ids"][0].dtype == torch.int32
        return torch.full((3, 256), 7, dtype=torch.bfloat16)

    api = SimpleNamespace(
        trtllm_bf16_routed_moe=kernel, trtllm_fp8_block_scale_routed_moe=kernel, prepare_trtllm_bf16_weights=prepare
    )
    monkeypatch.setattr(flashinfer_trtllm, "import_module", lambda name: api)
    bound = flashinfer_trtllm.BoundFlashInferTrtllmMoe(
        offer,
        gate_up,
        down,
        scales if quantized else None,
        torch.ones(2, 2, 2) if quantized else None,
        None,
        None,
    )
    monkeypatch.setattr(bound, "route", lambda *args: (torch.ones(3, 1), torch.zeros(3, 1, dtype=torch.int64)))
    if quantized:
        monkeypatch.setattr(
            flashinfer_trtllm,
            "quantize_fp8",
            lambda *args: (args[0].to(torch.float8_e4m3fn), torch.arange(6).float().reshape(3, 2)),
        )
    for _ in range(2):
        inputs = torch.ones(3, 256, dtype=spec.dtype)
        assert bound.run(inputs, torch.zeros(3, 2)) is inputs
        torch.testing.assert_close(inputs, torch.full_like(inputs, 7))
    assert len(prepared) == (0 if quantized else 1)
    for call in calls:
        torch.testing.assert_close(call["gemm1_weights"].float(), torch.cat(gate_up.float().chunk(2, 1)[::-1], 1))
        assert call["gemm2_weights"] is down
        if quantized:
            torch.testing.assert_close(call["gemm1_weights_scale"], torch.cat(scales.chunk(2, 1)[::-1], 1))
            torch.testing.assert_close(call["hidden_states_scale"], torch.arange(6).float().reshape(3, 2).T)


@pytest.mark.parametrize(
    "change", [dict(dtype=torch.float16), dict(round_intermediates=True), dict(hidden_size=192), dict(limit=7.0)]
)
def test_unsupported_semantics_reject_before_import(change, monkeypatch):
    monkeypatch.setattr(flashinfer_trtllm, "import_module", lambda name: pytest.fail("optional dependency imported"))
    spec = replace(MoeSpec(256, 256, 8, 2, torch.bfloat16), **change)
    backend = FlashInferTrtllmMoeBackend()
    assert not backend.probe(object(), format_for(spec.dtype), DeviceCaps("cuda", 0, (10, 0))).offers
    result = backend.probe(spec, format_for(spec.dtype), DeviceCaps("cuda", 0, (10, 0)))
    assert result.reason and not result.offers


def require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("TRT-LLM MoE numerical validation requires SM100/SM103")
    pytest.importorskip("flashinfer.fused_moe.prepare")


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("shared", [0, 1])
@torch.no_grad()
def test_blackwell_numerics_routing_tp_and_graph(quantized, shared):
    require_blackwell()
    torch.manual_seed(31)
    spec = MoeSpec(
        256,
        512,
        8,
        2,
        torch.bfloat16,
        scoring="sigmoid",
        renormalize=True,
        groups=2,
        selected_groups=1,
        group_score_topk=2,
        routed_scale=1.25,
        shared_experts=shared,
        gated_shared=bool(shared),
    )
    gate_up = torch.randn(8 + shared, 1024, 256, device="cuda") * 0.03
    gate_up[:, :512] *= 2
    down = torch.randn(8 + shared, 256, 512, device="cuda") * 0.04
    bias = torch.linspace(-0.1, 0.1, 8, device="cuda")
    expert_scale = torch.linspace(0.8, 1.2, 8, device="cuda")
    inputs = torch.randn(7, 256, dtype=spec.dtype, device="cuda")
    logits = torch.randn(7, 8, device="cuda")
    shared_gate = torch.randn(7, 1, dtype=spec.dtype, device="cuda") if shared else None
    bounds, references = [], []
    for rank in range(2):
        indices = torch.cat(
            (torch.arange(rank * 256, (rank + 1) * 256), torch.arange(512 + rank * 256, 512 + (rank + 1) * 256))
        ).to("cuda")
        w1, w2 = (
            gate_up[:, indices].to(spec.dtype),
            down[:, :, rank * 256 : (rank + 1) * 256].to(spec.dtype).contiguous(),
        )
        s1 = s2 = None
        if quantized:
            w1, s1, ref1 = quantize_blocks(w1.float())
            w2, s2, ref2 = quantize_blocks(w2.float())
        else:
            ref1, ref2 = w1.float(), w2.float()
        backend = FlashInferTrtllmMoeBackend()
        offer = backend.probe(
            spec,
            format_for(spec.dtype, quantized, (rank, 2)),
            DeviceCaps("cuda", 0, torch.cuda.get_device_capability()),
            fused_shared_experts=shared,
        ).offers[0]
        bounds.append(
            backend.bind(
                offer, w1, w2, gate_up_scale=s1, down_scale=s2, correction_bias=bias, expert_scale=expert_scale
            )
        )
        references.append((ref1, ref2))

    def reference():
        weights, ids = route_reference(spec, logits, bias, expert_scale, shared_gate, shared)
        result = torch.zeros_like(inputs, dtype=torch.float32)
        for w1, w2 in references:
            for token in range(len(inputs)):
                for slot, expert in enumerate(ids[token].tolist()):
                    gate, up = (inputs[token].float() @ w1[expert].T).chunk(2)
                    result[token] += (torch.nn.functional.silu(gate) * up) @ w2[expert].T * weights[token, slot]
        return result

    def execute():
        results = []
        for bound in bounds:
            buffer = inputs.clone()
            output = bound.run(buffer, logits, shared_gate=shared_gate)
            assert output is buffer
            results.append(output.float())
        return sum(results)

    for _ in range(3):
        actual = execute()
    tolerance = 0.08 if quantized else 0.03
    torch.testing.assert_close(actual, reference(), rtol=tolerance, atol=tolerance * 0.1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = execute()
    inputs.mul_(0.5)
    logits.neg_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference(), rtol=tolerance, atol=tolerance * 0.1)
    graph.reset()
