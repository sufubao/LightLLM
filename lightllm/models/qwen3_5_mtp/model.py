from typing import List

from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import Qwen35TransformerLayerInfer
from lightllm.models.qwen3_5_mtp.layer_weights.pre_and_post_layer_weight import Qwen3_5MTPPreAndPostLayerWeight
from lightllm.models.qwen3_5_mtp.layer_weights.transformer_layer_weight import Qwen3_5MTPTransformerLayerWeight
from lightllm.models.qwen3_5_mtp.layer_infer.pre_layer_infer import Qwen3_5MTPPreLayerInfer
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen3_5MTPModel(Qwen3_5TpPartModel):
    """Qwen3.5 MTP draft model (deepseek-MTP-style single-block speculator).

    The draft is ONE full-attention transformer layer that replicates Qwen3.5's mrope
    full-attention path. It shares the main model's embed_tokens + lm_head (config
    ``mtp_use_dedicated_embeddings: false``) and reuses the main model's req/mem managers
    and rotary cos/sin caches. It NEVER reads/writes conv/SSM (GDN) state -- it only uses
    its own full-attention KV.

    Inheritance design:
      - transformer_layer_infer_class is INHERITED (Qwen35TransformerLayerInfer): the draft
        layer takes the full-attn + mrope path, NOT the FFN-only path.
      - The base hybrid model decides GDN-vs-full-attn per layer via
        ``is_linear_attention_layer = (layer_num + 1) % full_attention_interval != 0``
        (computed in both Qwen3NextTransformerLayerWeight.__init__ and
        Qwen3NextTransformerLayerInfer.__init__). To force the single draft layer
        (layer_num == 0) to FULL attention we set ``full_attention_interval = 1`` in the
        draft's config: (0 + 1) % 1 == 0 -> not linear -> full attention. This forces
        full-attn for both the weight and the infer class in one place, without overriding
        the inherited transformer_layer_infer_class.

    NOTE: this model is intentionally NOT registered with @ModelRegistry -- MTP draft
    models are instantiated programmatically by the base backend (Phase 9).
    """

    pre_and_post_weight_class = Qwen3_5MTPPreAndPostLayerWeight
    pre_layer_infer_class = Qwen3_5MTPPreLayerInfer
    transformer_weight_class = Qwen3_5MTPTransformerLayerWeight
    # transformer_layer_infer_class intentionally inherited: Qwen35TransformerLayerInfer
    transformer_layer_infer_class = Qwen35TransformerLayerInfer

    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)
        return

    def _pre_init(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop("mtp_previous_draft_models")
        return

    def _init_config(self):
        super()._init_config()
        # The draft has a single layer that MUST be full-attention. The hybrid base keys
        # GDN-vs-full-attn on (layer_num + 1) % full_attention_interval; interval=1 makes
        # layer 0 full-attention. The draft only ever has one (full-attn) layer, so this
        # also makes the (unused) linear-layer count zero -- consistent with reusing the
        # main model's mem/req managers (the draft allocates no GDN state).
        self.config["full_attention_interval"] = 1
        self.config["num_hidden_layers"] = 1
        self.config["n_layer"] = 1
        return

    def _init_some_value(self):
        super()._init_some_value()
        self.layers_num = 1
        return

    def _init_custom(self):
        # Share rotary caches with the main model (mirrors deepseek_mtp). We deliberately
        # do NOT call super()._init_custom() so we skip rebuilding rotary tables and the
        # DeepEP group (the dense draft has no MoE experts anyway).
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached
        return

    def _init_req_manager(self):
        # Reuse the main model's req manager (holds the linear-attn state of the main model).
        self.req_manager = self.main_model.req_manager
        return

    def _init_mem_manager(self):
        # Reuse the main model's mem manager (KV + linear-attn buffers). The draft does NOT
        # rebuild the hybrid Qwen3NextMemManager and allocates no extra linear-attn state.
        self.mem_manager = self.main_model.mem_manager
        return

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type, network_config=self.config, quant_cfg=self.quant_cfg
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            for i in range(0, self.config["n_layer"])
        ]
        # Shared with the main Qwen3.5 model (mtp_use_dedicated_embeddings: false).
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        return

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        # Build the single draft layer with layer_num == 0 so that, with
        # full_attention_interval == 1, it takes the full-attention (mrope) path.
        super()._init_infer_layer(start_layer_index=0)
        self._assign_draft_kv_slot()
        return

    def _assign_draft_kv_slot(self):
        """Give the draft full-attn layer a DEDICATED KV slot that does NOT collide with the
        main model's full-attn slots.

        Decoupling weight-loading (needs layer_num=0) from KV-slot indexing:
          - WEIGHT loading already happened with the weight object's layer_num == 0, so it
            correctly resolves the ``mtp.layers.0.*`` keys. That is untouched here.
          - At RUNTIME the KV slot is derived ONLY from the INFER layer's ``layer_num_`` --
            it flows into ``mem_manager.get_att_input_params(layer_index=self.layer_num_)``
            (read) and ``operator.copy_kv_to_mem_manager(layer_index=self.layer_num_)``
            (write), and the SHARED Qwen3NextMemManager maps it via ``// full_attention_interval``
            (interval=4 on the main config). Nothing else in the inherited full-attn (mrope)
            path reads ``layer_num_`` -- conv/SSM are GDN-only and the draft layer is full-attn.
          - So we can set the INFER layer's ``layer_num_`` to a dedicated value INDEPENDENT of
            the weight object's layer_num. We pick ``(main_full_att + draft_idx) * interval`` so
            that the existing ``// interval`` math lands the draft at slot
            ``main_full_att + draft_idx`` (e.g. slot 16 for the first/only draft layer with
            main_full_att=16), past all main slots [0, main_full_att).
        """
        mem_manager = self.main_model.mem_manager
        main_full_att = getattr(mem_manager, "main_full_att_layer_num", None)
        interval = self.main_model.config["full_attention_interval"]
        if main_full_att is None:
            # Non-hybrid / unexpected mem_manager: nothing to remap.
            return
        # This draft block is one full-attn layer. In vanilla_with_att with mtp_step > 1,
        # init_mtp_draft_model creates `mtp_step` SEPARATE Qwen3_5MTPModel instances; the i-th
        # instance has i previously-created draft models in mtp_previous_draft_models, so we use
        # that length as this block's draft_idx (mirrors qwen3_moe_mtp / deepseek_mtp layer
        # offsetting). For eagle_with_att (one instance) draft_idx == 0. Each draft block thus
        # gets a DISTINCT slot main_full_att + draft_idx, all within the sized buffer.
        draft_idx = len(self.mtp_previous_draft_models)
        draft_full_att_layers = getattr(mem_manager, "draft_full_att_layers", None)
        if draft_full_att_layers is not None:
            assert draft_idx < draft_full_att_layers, (
                f"draft_idx {draft_idx} out of range for draft_full_att_layers "
                f"{draft_full_att_layers}; mem_manager not sized for this many MTP draft blocks"
            )
        draft_kv_slot = main_full_att + draft_idx
        layer_infer = self.layers_infer[0]
        layer_infer._draft_kv_slot = draft_kv_slot
        # Set the runtime KV-slot index so the shared mem_manager's `// interval` maps to the
        # dedicated slot. The weight object keeps layer_num == 0 (mtp.layers.0 already loaded).
        layer_infer.layer_num_ = draft_kv_slot * interval
        logger.info(
            f"Qwen3.5 MTP draft layer assigned dedicated full-attn KV slot {draft_kv_slot} "
            f"(layer_num_={layer_infer.layer_num_}, interval={interval}, main_full_att={main_full_att})"
        )
        return
