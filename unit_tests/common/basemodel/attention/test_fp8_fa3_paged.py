from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from lightllm.common.basemodel.attention.fa3.fp8 import (
    Fp8Fa3AttBackend,
    Fp8Fa3DecodeAttState,
    Fp8Fa3PrefillAttState,
)
from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_fp8_paged_attention_matches_token_pages_with_shared_partial_pages(is_prefill, kv_heads):
    torch.manual_seed(31)
    page_size = 128
    physical_tokens = 8192
    kv = torch.randn(physical_tokens, 2 * kv_heads, 256, dtype=torch.bfloat16, device="cuda")
    kv = kv.to(torch.float8_e4m3fn).view(torch.uint8)
    lengths = torch.tensor([257, 385, 513], dtype=torch.int32, device="cuda")
    blocks = torch.stack([torch.randperm(physical_tokens // page_size, device="cuda")[:5] for _ in range(3)])
    blocks[:, 0] = blocks[0, 0]
    token_map = (blocks[:, :, None] * page_size + torch.arange(page_size, device="cuda")).reshape(3, -1)
    req_ids = torch.arange(3, dtype=torch.int32, device="cuda")
    widths = [17, 31, 65] if is_prefill else [1, 2, 4]
    cu_q = torch.tensor([0, widths[0], sum(widths[:2]), sum(widths)], dtype=torch.int32, device="cuda")
    q = torch.randn(sum(widths), 6 * kv_heads, 256, dtype=torch.bfloat16, device="cuda")
    mem_manager = SimpleNamespace(
        kv_buffer=[kv],
        q_scales=torch.full((1, kv_heads), 0.25, device="cuda"),
        scales=torch.full((1, 2 * kv_heads), 0.5, device="cuda"),
    )
    infer_state = SimpleNamespace(mem_manager=mem_manager, max_q_seq_len=max(widths), b_seq_len=lengths)

    def make_state(infer_page_size):
        backend = object.__new__(Fp8Fa3AttBackend)
        backend.model = SimpleNamespace(args=SimpleNamespace(page_size=infer_page_size), mem_manager=mem_manager)
        backend._init_infer_page_size()
        state_class = Fp8Fa3PrefillAttState if is_prefill else Fp8Fa3DecodeAttState
        state = state_class(backend=backend, infer_state=infer_state)
        state.page_table = torch.empty(
            (3, (513 + infer_page_size - 1) // infer_page_size), dtype=torch.int32, device="cuda"
        )
        page_table_copy(state.page_table, token_map, req_ids, page_size=infer_page_size)
        state.cu_seqlens_q = cu_q
        state.cu_seqlens_k = cu_q
        state.k_descale = mem_manager.scales[:, :kv_heads, None].transpose(1, 2).expand(1, 3, kv_heads)
        state.v_descale = mem_manager.scales[:, kv_heads:, None].transpose(1, 2).expand(1, 3, kv_heads)
        state.causal = True
        if not is_prefill:
            state.b_att_seq_len = lengths
            state.decode_max_q_seq_len = max(widths)
        return state

    states = [make_state(size) for size in (1, page_size)]
    k, v = kv[:, :kv_heads], kv[:, kv_heads:]
    outputs = []
    for state in states:
        forward = state.prefill_att if is_prefill else state.decode_att
        outputs.append(forward(q, k, v))
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = forward(q, k, v)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, outputs[0], rtol=0, atol=0)
