from lightllm.models.qwen4_exp.layer_weights.transformer_layer_weight import (
    Qwen4ExpTransformerLayerWeight,
)


def rename_qwen4_mtp_layer_weight_keys(weights: dict) -> None:
    """Expose the checkpoint's recurrent ``mtp.layers.0`` as decoder layer 0."""

    checkpoint_prefix = "mtp.layers.0."
    decoder_prefix = "model.layers.0."
    for name in list(weights):
        if name.startswith(checkpoint_prefix):
            weights[decoder_prefix + name[len(checkpoint_prefix) :]] = weights.pop(name)


class Qwen4ExpMTPTransformerLayerWeight(Qwen4ExpTransformerLayerWeight):
    def load_hf_weights(self, weights):
        rename_qwen4_mtp_layer_weight_keys(weights)
        return super().load_hf_weights(weights)
