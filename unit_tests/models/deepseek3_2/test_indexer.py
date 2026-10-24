from types import SimpleNamespace

import pytest
import torch

from lightllm.models.deepseek3_2.layer_infer import transformer_layer_infer as indexer_module


@pytest.mark.parametrize("tp_world_size", [1, 2, 8])
def test_get_indices_gathers_query_heads_before_quantization(monkeypatch, tp_world_size):
    indexer = indexer_module.NsaInfer(
        0,
        {
            "index_topk": 2,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 0,
            "index_head_dim": 4,
            "rms_norm_eps": 1e-5,
            "quantization_config": {"weight_block_size": [4, 4], "scale_fmt": "ue8m0"},
            "index_n_heads": 2 * tp_world_size,
        },
        tp_world_size,
    )
    queries = [torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) + rank * 100 for rank in range(tp_world_size)]
    group = object()
    gathered = []
    monkeypatch.setattr(indexer, "_get_q_k_bf16", lambda *args: (queries[0], None))

    def gather(output_, input_, group, async_op):
        assert group is infer_state.dist_group and async_op is False
        assert output_.shape == (24 * tp_world_size,)
        assert output_.dtype == queries[0].dtype and output_.device == queries[0].device
        torch.testing.assert_close(input_, queries[0].flatten(), rtol=0, atol=0)
        output_.copy_(torch.cat([query.flatten() for query in queries]))
        gathered.append(True)

    def quantize(query, block_size, scale_fmt):
        assert block_size == 4 and scale_fmt == "ue8m0"
        assert query.shape == (3, 2 * tp_world_size, 4) and query.is_contiguous()
        torch.testing.assert_close(query, torch.cat(queries, dim=1), rtol=0, atol=0)
        raise RuntimeError("query reached quantization")

    monkeypatch.setattr(indexer_module, "all_gather_into_tensor", gather)
    monkeypatch.setattr(indexer_module, "act_quant", quantize)
    infer_state = SimpleNamespace(dist_group=group)
    with pytest.raises(RuntimeError, match="query reached quantization"):
        indexer._get_indices(None, None, infer_state, None, None)
    assert len(gathered) == int(tp_world_size > 1)
