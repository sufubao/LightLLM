from lightllm.common.basemodel.hidden_collector import FinalHiddenCollector


class Qwen35HiddenCollector(FinalHiddenCollector):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def new_instance(self):
        return Qwen35HiddenCollector(self.model)

    def finish_output(self, infer_state):
        self.final_hidden = self.model.post_infer._norm(self.final_hidden, infer_state, self.model.pre_post_weight)
        return super().finish_output(infer_state)
