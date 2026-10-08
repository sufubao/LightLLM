from lightllm.common.basemodel.hidden_collector import FinalHiddenCollector
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class Qwen35PostLayerInfer(LlamaPostLayerInfer):
    def token_forward(self, input_embdings, infer_state, layer_weight):
        output = super().token_forward(input_embdings, infer_state, layer_weight)
        if isinstance(infer_state.hidden_collector, FinalHiddenCollector):
            output.final_hidden = self._norm(input_embdings, infer_state, layer_weight)
        return output
