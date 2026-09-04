"""Complete model support declarations for Tarsier2 variants."""

from lightllm.models.registry import ModelContext, ModelSupport, ModelSupportRegistry
from lightllm.models.registry import TokenizerBuildContext, VisionBuildContext
from lightllm.models.tarsier2.model import (
    Tarsier2LlamaTpPartModel,
    Tarsier2Qwen2TpPartModel,
    Tarsier2Qwen2VLTpPartModel,
    Tarsier2Tokenizer,
)


def _is_tarsier_with_text_type(text_model_type):
    def condition(context: ModelContext) -> bool:
        return (
            "TarsierForConditionalGeneration" in context.architectures
            and context.text_config.get("model_type") == text_model_type
        )

    return condition


def _create_tokenizer(context: TokenizerBuildContext):
    from lightllm.models.qwen2_vl.vision_process import Qwen2VLImageProcessor

    image_processor = Qwen2VLImageProcessor.from_pretrained(context.tokenizer_name)
    return Tarsier2Tokenizer(
        tokenizer=context.tokenizer,
        image_processor=image_processor,
        model_cfg=context.model.raw_config,
    )


def _create_vision_model(context: VisionBuildContext):
    from lightllm.models.tarsier2.tarsier2_visual import TarsierVisionTransformerPretrainedModel

    return TarsierVisionTransformerPretrainedModel(**dict(context.model.raw_config)).eval().bfloat16()


for _name, _text_type, _model_class in (
    ("tarsier2_qwen2", "qwen2", Tarsier2Qwen2TpPartModel),
    ("tarsier2_qwen2_vl", "qwen2_vl", Tarsier2Qwen2VLTpPartModel),
    ("tarsier2_llama", "llama", Tarsier2LlamaTpPartModel),
):
    ModelSupportRegistry.register_support(
        ModelSupport(
            name=_name,
            model_types=("llava",),
            text_model=_model_class,
            condition=_is_tarsier_with_text_type(_text_type),
            tokenizer_factory=_create_tokenizer,
            vision_factory=_create_vision_model,
        )
    )
