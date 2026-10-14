from typing import ClassVar, Optional

from lightllm.common.basemodel.batch_objs import FinalHiddenOutputPolicy
from lightllm.common.basemodel.spec_collector import (
    FinalHiddenSpecCollector,
    SpecCollector,
    LayerHiddenSpecCollector,
    MtpHeadCollector,
    NoopSpecCollector,
)
from lightllm.utils.envs_utils import get_env_start_args


class SpecManager:
    """Manage speculative decoding layout policy and model-local helper construction."""

    _instance: ClassVar[Optional["SpecManager"]] = None
    _CHAINED_DRAFT_MODES = ("vanilla_with_att", "vanilla_no_att")
    _RECURRENT_DRAFT_MODES = ("eagle_with_att", "eagle_no_att", "eagle3")
    _BLOCK_DRAFT_MODES = ("dspark", "dflash")

    @classmethod
    def get_instance(cls) -> "SpecManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.args = get_env_start_args()

    def get_decode_tokens_per_request(self, is_draft_model: bool) -> int:
        """返回每请求的 decode token 数；动态 verify 返回压缩前的最大 token 数，用于容量规划。"""

        spec_mode = self.args.spec_mode
        if spec_mode is None:
            return 1

        verify_width = self.args.spec_step + 1

        # The main model verifies one target token plus spec_step draft tokens
        # for every logical request, regardless of how the draft is produced.
        if not is_draft_model:
            return verify_width

        # Chained MTP runs every draft module over the expanded verify layout.
        if spec_mode in self._CHAINED_DRAFT_MODES:
            return 1

        # Recurrent EAGLE draft models decode one row per logical request.
        if spec_mode in self._RECURRENT_DRAFT_MODES:
            return 1

        # Block draft models decode spec_step rows per logical request.
        if spec_mode in self._BLOCK_DRAFT_MODES:
            return self.args.spec_step

        return 1

    def get_decode_batch_alignment(self, is_draft_model: bool) -> int:
        """返回 decode/graph 的基础对齐粒度；动态主模型压缩后允许任意行数。"""
        if not is_draft_model and self.args.spec_dynamic_verify:
            return 1
        return self.get_decode_tokens_per_request(is_draft_model)

    def get_decode_draft_step(self, is_draft_model: bool) -> int:
        """Return the number of extra decode rows processed per request."""

        return self.get_decode_tokens_per_request(is_draft_model) - 1

    def get_final_hidden_output_policy(self, model) -> FinalHiddenOutputPolicy:
        spec_mode = self.args.spec_mode
        if spec_mode is None or spec_mode in self._BLOCK_DRAFT_MODES:
            return FinalHiddenOutputPolicy.NONE
        if spec_mode == "eagle3" and not model.is_spec_draft_model:
            return FinalHiddenOutputPolicy.NONE
        return getattr(model, "spec_final_hidden_output_policy", FinalHiddenOutputPolicy.PRE_NORM)

    def create_spec_collector(
        self,
        model,
    ) -> SpecCollector:
        """Create a model-local hidden-collector prototype for the configured speculative decoding mode."""

        spec_mode = self.args.spec_mode
        collector_kwargs = {}
        if spec_mode is None:
            collector_type = NoopSpecCollector
        elif model.is_spec_draft_model:
            if spec_mode == "dspark":
                collector_type = MtpHeadCollector
            elif spec_mode in self._BLOCK_DRAFT_MODES:
                collector_type = NoopSpecCollector
            else:
                collector_type = FinalHiddenSpecCollector
        elif spec_mode in ("eagle3", *self._BLOCK_DRAFT_MODES):
            collector_type = LayerHiddenSpecCollector
            collector_kwargs.update(model=model)
        else:
            collector_type = FinalHiddenSpecCollector

        return collector_type(**collector_kwargs)
