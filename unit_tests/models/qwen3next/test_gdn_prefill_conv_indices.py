from types import SimpleNamespace

import torch


def test_gdn_prefill_uses_one_slot_conv_indices(monkeypatch):
    from lightllm.models.qwen3next.layer_infer import transformer_layer_infer as layer_mod

    layer = layer_mod.Qwen3NextTransformerLayerInfer.__new__(layer_mod.Qwen3NextTransformerLayerInfer)
    layer.activation = "silu"
    layer.needs_ssm_dtype_conversion = False

    captured = {}

    def fake_causal_conv1d_fn(mixed_qkv, *args, cache_indices=None, **kwargs):
        captured["cache_indices"] = cache_indices.detach().cpu().clone()
        return mixed_qkv

    def fake_fused_gdn_gating(*args, **kwargs):
        return torch.zeros(3, 1), torch.ones(3, 1)

    def fake_chunk_gated_delta_rule(*args, **kwargs):
        return torch.zeros(1, 3, 1, 1), torch.zeros(3, 1)

    def fake_rearrange_mixed_qkv(*args, **kwargs):
        return torch.zeros(1, 3, 1, 1), torch.zeros(1, 3, 1, 1), torch.zeros(1, 3, 1, 1)

    monkeypatch.setattr(layer_mod, "causal_conv1d_fn", fake_causal_conv1d_fn)
    monkeypatch.setattr(layer_mod, "fused_gdn_gating", fake_fused_gdn_gating)
    monkeypatch.setattr(layer_mod, "chunk_gated_delta_rule", fake_chunk_gated_delta_rule)
    layer._rearrange_mixed_qkv = fake_rearrange_mixed_qkv

    infer_state = SimpleNamespace(
        # SSM keeps an (S+1)-slot block per request; for S=1 these are 0,2,4.
        b_buffer_idx=torch.tensor([0, 2, 4], dtype=torch.int64),
        # Conv keeps one widened slot per request; prefill must write 0,1,2.
        b_conv_buffer_idx=torch.tensor([0, 1, 2], dtype=torch.int64),
        b1_cu_q_seq_len=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        b_ready_cache_len=torch.zeros(3, dtype=torch.int32),
    )
    layer_weight = SimpleNamespace(
        linear_conv1d=SimpleNamespace(mm_param=SimpleNamespace(weight=torch.zeros(1, 1)), bias=None),
        linear_A_log=SimpleNamespace(weight=torch.zeros(1)),
        linear_dt_bias=SimpleNamespace(weight=torch.zeros(1)),
    )

    layer._gdn_prefill_kernel(
        mixed_qkv=torch.zeros(3, 1),
        conv_states=torch.zeros(3, 1, 1),
        ssm_states=torch.zeros(6, 1),
        a=torch.zeros(3, 1),
        b=torch.zeros(3, 1),
        infer_state=infer_state,
        layer_weight=layer_weight,
    )

    assert captured["cache_indices"].tolist() == [0, 1, 2]
