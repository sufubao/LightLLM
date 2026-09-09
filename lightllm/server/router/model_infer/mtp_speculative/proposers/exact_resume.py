"""Auxiliary-cache repair for a single Qwen3.5 attention draft module."""

import copy

import torch

from lightllm.common.basemodel.batch_objs import ModelInput


class Qwen35ExactResumeMixin:
    def supports_exact_prefix_resume(self) -> bool:
        """Do not apply a final-hidden recipe to multi-layer-hidden drafters."""
        from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel

        return (
            self.backend.args.mtp_mode in ("vanilla_with_att", "eagle_with_att")
            and len(self.backend.draft_models) == 1
            and isinstance(self.backend.draft_models[0], Qwen3_5MTPModel)
            and self.backend.model.supports_exact_output_seed()
        )

    def resume_auxiliary(
        self,
        resume_input: ModelInput,
        output_seed: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> None:
        """Rebuild draft[L-1] using H@L and the new token at logical index L.

        ``resume_input`` contains one decode row per request, b_seq_len=L and
        an independently owned target/draft packed KV slot at token L-1. The
        backend must copy the target layers to that slot before calling us;
        this forward changes only the draft layer. Draft KV for [0,L-1) must
        already be restored. Rebuilding a shared slot would corrupt another
        checkpoint and is forbidden by the caller's restore contract.

        The new token is either a fresh HEAD_ONLY sample or the first input
        suffix token, so the same repair applies to partial and complete hits.
        This does not advance the target recurrent state or generate an extra
        user-visible token. The normal first decode iteration seeds proposals.
        """
        if not self.supports_exact_prefix_resume():
            raise NotImplementedError("the configured proposer has no exact-prefix resume adapter")
        if resume_input.is_prefill:
            raise ValueError("auxiliary resume requires one decode row per restored request")
        batch_size = resume_input.batch_size
        if output_seed.ndim != 2 or output_seed.shape[0] != batch_size:
            raise ValueError("output seed rows must match resumed requests")
        if next_token_ids.shape != (batch_size,) or not next_token_ids.is_cuda:
            raise ValueError("resume tokens must be a CUDA vector with one token per request")
        if not output_seed.is_cuda or output_seed.device != next_token_ids.device:
            raise ValueError("resume hidden and tokens must use the same CUDA device")
        draft_input = copy.copy(resume_input)
        draft_input.input_ids = next_token_ids
        draft_input.b_mtp_index = torch.zeros_like(resume_input.b_req_idx)
        # The Qwen3.5 draft pre-layer normalizes this argument in-place.
        draft_input.mtp_draft_input_hiddens = output_seed.clone()
        self.backend.draft_models[0].forward(draft_input)
