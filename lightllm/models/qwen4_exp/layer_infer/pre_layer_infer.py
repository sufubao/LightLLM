from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import (
    Qwen3VLMultimodalPreLayerInfer,
)


class Qwen4ExpPreLayerInfer(Qwen3VLMultimodalPreLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        self.hc_count = network_config["hc_count"]

    def context_forward(self, input_ids, infer_state, layer_weight):
        embeddings = super().context_forward(input_ids, infer_state, layer_weight)
        return embeddings.repeat(1, self.hc_count)

    def token_forward(self, input_ids, infer_state, layer_weight):
        embeddings = super().token_forward(input_ids, infer_state, layer_weight)
        return embeddings.repeat(1, self.hc_count)
