import torch

from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import (
    Qwen3VLMultimodalPreLayerInfer,
)
from lightllm.models.qwen4_exp.hyperconnection import grouped_gemma_rmsnorm
from lightllm.models.qwen4_exp_mtp.layer_weights.pre_and_post_layer_weight import (
    Qwen4ExpMTPPreAndPostLayerWeight,
)


class Qwen4ExpMTPPreLayerInfer(Qwen3VLMultimodalPreLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["rms_norm_eps"]
        self.hidden_size = network_config["hidden_size"]
        self.hc_count = network_config["hc_count"]

    def _mtp_fuse(
        self,
        input_embeddings: torch.Tensor,
        infer_state,
        layer_weight: Qwen4ExpMTPPreAndPostLayerWeight,
    ) -> torch.Tensor:
        target_hidden = infer_state.mtp_draft_input_hiddens
        if target_hidden is None:
            raise ValueError("Qwen4 MTP requires target multi-stream hidden states")
        expected_width = self.hc_count * self.hidden_size
        if input_embeddings.shape[0] != target_hidden.shape[0]:
            raise ValueError(
                f"token/hidden row mismatch: {input_embeddings.shape[0]} != "
                f"{target_hidden.shape[0]}"
            )
        if target_hidden.shape[-1] != expected_width:
            raise ValueError(
                f"Qwen4 MTP expects hidden width {expected_width}, got "
                f"{target_hidden.shape[-1]}"
            )

        normalized_embedding = grouped_gemma_rmsnorm(
            input_embeddings,
            layer_weight.pre_fc_norm_embedding.weight,
            hidden_size=self.hidden_size,
            eps=self.eps_,
        )
        projected_embedding = layer_weight.fc_embedding.mm(normalized_embedding)

        normalized_hidden = grouped_gemma_rmsnorm(
            target_hidden,
            layer_weight.pre_fc_norm_hidden.weight,
            hidden_size=self.hidden_size,
            eps=self.eps_,
        ).view(-1, self.hidden_size)
        projected_hidden = layer_weight.fc_hidden.mm(normalized_hidden).view(
            -1, self.hc_count, self.hidden_size
        )
        return (projected_hidden + projected_embedding.unsqueeze(1)).flatten(-2)

    def context_forward(self, input_ids, infer_state, layer_weight):
        input_embeddings = super().context_forward(input_ids, infer_state, layer_weight)
        return self._mtp_fuse(input_embeddings, infer_state, layer_weight)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embeddings = super().token_forward(input_ids, infer_state, layer_weight)
        return self._mtp_fuse(input_embeddings, infer_state, layer_weight)
