import torch

from lightllm.distributed.communication_op import all_gather_into_tensor
from lightllm.models.qwen4_exp.layer_infer.post_layer_infer import (
    Qwen4ExpPostLayerInfer,
)
from lightllm.utils.envs_utils import get_env_start_args


def select_global_argmax(gathered_winners: torch.Tensor) -> torch.Tensor:
    """Select global token ids from rank-major ``(score, token_id)`` pairs."""

    assert gathered_winners.ndim == 3 and gathered_winners.shape[-1] == 2
    token_rows = torch.arange(
        gathered_winners.shape[1],
        dtype=torch.long,
        device=gathered_winners.device,
    )
    winning_ranks = torch.argmax(gathered_winners[:, :, 0], dim=0)
    return gathered_winners[winning_ranks, token_rows, 1].long()


class Qwen4ExpMTPPostLayerInfer(Qwen4ExpPostLayerInfer):
    """Qwen4 recurrent-draft output layer with vocab-parallel argmax."""

    def __init__(self, network_config):
        super().__init__(network_config)
        # Dynamic verification needs full probabilities. Fixed-width EAGLE only
        # consumes greedy draft token ids, so gathering the complete vocabulary
        # would communicate O(batch * vocab_size) data unnecessarily.
        self.use_local_argmax_ = not get_env_start_args().mtp_dynamic_verify

    def _local_argmax_token_forward(self, input_embdings, infer_state, layer_weight):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        normed = self._norm(last_input, infer_state, layer_weight)
        local_logits = layer_weight.lm_head_weight_.batch_major_forward(
            input=normed,
            alloc_func=self.alloc_tensor,
        )
        local_max_values, local_max_indexes = torch.max(local_logits, dim=1)
        global_token_ids = (
            local_max_indexes + layer_weight.lm_head_weight_.tp_vocab_start_id
        )

        if self.tp_world_size_ == 1:
            draft_token_ids = global_token_ids
        else:
            local_winners = torch.stack(
                [local_max_values.float(), global_token_ids.float()],
                dim=-1,
            ).contiguous()
            gathered_winners = self.alloc_tensor(
                (self.tp_world_size_ * token_num, 2),
                dtype=torch.float32,
            )
            all_gather_into_tensor(
                gathered_winners,
                local_winners,
                group=infer_state.dist_group,
                async_op=False,
            )
            draft_token_ids = select_global_argmax(
                gathered_winners.view(self.tp_world_size_, token_num, 2)
            )

        infer_state.hidden_collector.add_mtp_outputs(
            draft_token_ids=draft_token_ids,
            confidence_logits=None,
        )
        # Decode graph unpadding only needs the leading row count when token ids
        # are returned through the MTP collector.
        return local_logits.new_empty((token_num, 1))

    def token_forward(self, input_embdings, infer_state, layer_weight):
        if infer_state.is_prefill or not self.use_local_argmax_:
            return super().token_forward(input_embdings, infer_state, layer_weight)
        return self._local_argmax_token_forward(
            input_embdings, infer_state, layer_weight
        )
