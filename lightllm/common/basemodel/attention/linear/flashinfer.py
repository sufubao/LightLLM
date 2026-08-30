import torch

from lightllm.common.basemodel.attention.linear.gdn import LinearAttBackend
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.l2norm import (
    l2norm_fwd,
)
from lightllm.common.basemodel.triton_kernel.linear_att.fused_gdn_prefill_post_conv import (
    fused_gdn_prefill_post_conv,
)


def flashinfer_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    """Adapt FlashInfer's packed GDN prefill kernel to the FLA interface.

    LightLLM's linear-attention path represents ``g`` as the logarithm of the
    forget gate and keeps a leading singleton batch dimension. FlashInfer takes
    the materialized forget gate and packed tensors without that dimension.
    Keep the explicit L2-normalization used by vLLM's Qwen GDN integration so
    the two runtimes follow the same numerical path.
    """

    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

    # LightLLM stores recurrent states as [B, HV, K, V], matching its FLA
    # decode kernels. FlashInfer (and vLLM's GDN cache) use [B, HV, V, K].
    # Qwen3.8 happens to have K == V, so passing the cache through unchanged
    # is shape-correct but silently transposes the state semantics: prefill
    # logits look plausible, then the first decode step consumes a corrupted
    # state. Keep this layout conversion at the backend boundary.
    flashinfer_state = initial_state.transpose(-1, -2).contiguous().float()
    result = chunk_gated_delta_rule(
        q=q.squeeze(0).contiguous(),
        k=k.squeeze(0).contiguous(),
        v=v.squeeze(0).contiguous(),
        g=torch.exp(g.squeeze(0).contiguous().float()),
        beta=beta.squeeze(0).contiguous().float(),
        initial_state=flashinfer_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    if output_final_state:
        output, final_state = result
        lightllm_state = final_state.transpose(-1, -2).contiguous()
        return output.unsqueeze(0), lightllm_state
    return result.unsqueeze(0), None


class FlashInferLinearAttBackend(LinearAttBackend):
    def get_prefill_kernel(self):
        return flashinfer_chunk_gated_delta_rule

    def prepare_prefill_inputs(self, mixed_qkv, a, b, layer_weight):
        query, key, value, g, beta = fused_gdn_prefill_post_conv(
            conv_output=mixed_qkv,
            a=a,
            b=b,
            A_log=layer_weight.linear_A_log.weight,
            dt_bias=layer_weight.linear_dt_bias.weight,
            num_k_heads=self.tp_num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        return (
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            g.unsqueeze(0),
            beta.unsqueeze(0),
            False,
        )
