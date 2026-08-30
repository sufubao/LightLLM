import torch

from lightllm.distributed.communication_op import all_gather_into_tensor
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer

from ..hyperconnection import (
    grouped_gemma_rmsnorm,
    hyperconnection_mix,
    hyperconnection_silu,
)


class Qwen4ExpPostLayerInfer(LlamaPostLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        self.hidden_size = network_config["hidden_size"]
        self.hc_count = network_config["hc_count"]

    def _norm(self, input, infer_state, layer_weight):
        mixer = layer_weight.final_mixer
        normalized = grouped_gemma_rmsnorm(
            input,
            mixer.hc_norm.weight,
            hidden_size=self.hidden_size,
            eps=self.eps_,
        )
        lowrank = hyperconnection_silu(
            mixer.input_mix_weight_down.mm(normalized), self.hc_count
        )
        gate_logits = mixer.input_mix_weight_up.mm(lowrank)
        return hyperconnection_mix(normalized, gate_logits, hc_count=self.hc_count)

    def _lm_head_and_gather(self, hidden, token_num, layer_weight, infer_state):
        # Keep tokens as the leading dimension, matching torch.nn.functional.linear
        # and vLLM. For this model, the legacy W @ hidden.T path selects a different
        # BF16 cuBLAS reduction for a one-request versus packed prefill batch.
        normed = self._norm(hidden, infer_state, layer_weight)
        local_logits = layer_weight.lm_head_weight_.batch_major_forward(
            input=normed,
            alloc_func=self.alloc_tensor,
        )

        local_vocab_size = local_logits.shape[1]
        vocab_size = layer_weight.lm_head_weight_.vocab_size
        if self.tp_world_size_ == 1:
            rank_major_logits = local_logits.view(1, token_num, local_vocab_size)
        else:
            assert local_vocab_size * self.tp_world_size_ == vocab_size
            gathered_logits = self.alloc_tensor(
                (self.tp_world_size_ * token_num, local_vocab_size),
                dtype=hidden.dtype,
            )
            all_gather_into_tensor(
                gathered_logits,
                local_logits,
                group=infer_state.dist_group,
                async_op=False,
            )
            rank_major_logits = gathered_logits.view(
                self.tp_world_size_, token_num, local_vocab_size
            )

        logits = self.alloc_tensor((token_num, vocab_size), dtype=torch.float32)
        logits.view(token_num, self.tp_world_size_, local_vocab_size).copy_(
            rank_major_logits.permute(1, 0, 2)
        )
        return logits
