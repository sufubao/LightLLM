import sys
from types import ModuleType

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.nsa.flashmla_sparse import NsaFlashMlaSparsePrefillAttState


@pytest.mark.parametrize("heads,padded_heads", [(16, 64), (64, 64), (128, 128)])
@pytest.mark.parametrize("indices_ndim", [2, 3])
def test_prefill_pads_query_heads_and_trims_kernel_output(monkeypatch, heads, padded_heads, indices_ndim):
    tokens, dim, value_dim = 3, 8, 4
    query = torch.arange(tokens * heads * dim, dtype=torch.bfloat16).reshape(tokens, heads, dim) + 1
    original = query.clone()
    kv_buffer = torch.zeros(7, 1, dim + 2, dtype=query.dtype)[:, :, :dim]
    indices = torch.tensor([[2, 0, -1, 1], [0, 2, -1, -1], [3, 4, 2, -1]], dtype=torch.int32)
    if indices_ndim == 3:
        indices = indices.unsqueeze(1)
    calls = []

    def sparse_prefill(*, q, kv, indices, sm_scale, d_v):
        assert q.shape == (tokens, padded_heads, dim) and q.is_contiguous()
        assert q.dtype == query.dtype and q.device == query.device
        torch.testing.assert_close(q[:, :heads], original, rtol=0, atol=0)
        if heads < padded_heads:
            assert torch.count_nonzero(q[:, heads:]) == 0
        else:
            assert q is query
        assert kv is kv_buffer
        assert sm_scale == 0.125 and d_v == value_dim
        expected_indices = control.nsa_prefill_dict["topk_mem_indices"]
        assert indices.shape == (tokens, 1, 4) and indices.dtype == torch.int32
        assert indices.data_ptr() == expected_indices.data_ptr()
        if indices_ndim == 3:
            assert indices is expected_indices
        torch.testing.assert_close(indices[:, 0], expected_indices.reshape(tokens, 4), rtol=0, atol=0)
        output = q.new_full((tokens, padded_heads, value_dim), -321)
        output[:, :heads] = q[:, :heads, :value_dim] + 2
        calls.append(output)
        return output, None, None

    module = ModuleType("sgl_kernel.flash_mla")
    module.flash_mla_sparse_fwd = sparse_prefill
    monkeypatch.setitem(sys.modules, "sgl_kernel.flash_mla", module)
    control = AttControl(
        nsa_prefill=True,
        nsa_prefill_dict={"topk_mem_indices": indices, "softmax_scale": 0.125, "kv_lora_rank": value_dim},
    )
    state = NsaFlashMlaSparsePrefillAttState()
    output = state.prefill_att(query, kv_buffer, None, control)

    assert len(calls) == 1 and output.shape == (tokens, heads, value_dim)
    assert output.data_ptr() == calls[0].data_ptr()
    assert not torch.any(output == -321)
    torch.testing.assert_close(output, original[:, :, :value_dim] + 2, rtol=0, atol=0)
    torch.testing.assert_close(query, original, rtol=0, atol=0)
