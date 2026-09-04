"""Complete model support declaration for Qwen3-VL."""

from lightllm.models.qwen3_vl.model import QWen3VLTokenizer, Qwen3VLTpPartModel
from lightllm.models.registry import ModelSupport, ModelSupportRegistry, TokenizerBuildContext, VisionBuildContext


def _create_tokenizer(context: TokenizerBuildContext):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(context.tokenizer_name)
    return QWen3VLTokenizer(
        tokenizer=context.tokenizer,
        image_processor=processor.image_processor,
        model_cfg=context.model.raw_config,
    )


def _create_vision_model(context: VisionBuildContext):
    from lightllm.models.qwen3_vl.qwen3_visual import Qwen3VisionTransformerPretrainedModel

    return (
        Qwen3VisionTransformerPretrainedModel(
            dict(context.kvargs),
            **dict(context.model.vision_config),
        )
        .eval()
        .bfloat16()
    )


QWEN3_VL_SUPPORT = ModelSupportRegistry.register_support(
    ModelSupport(
        name="qwen3_vl",
        model_types=("qwen3_vl",),
        text_model=Qwen3VLTpPartModel,
        tokenizer_factory=_create_tokenizer,
        vision_factory=_create_vision_model,
    )
)
