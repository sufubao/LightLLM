"""Public model APIs with lazy compatibility exports."""

from importlib import import_module

from .draft_registry import get_draft_model_class
from .registry import (
    ModelContext,
    ModelSupport,
    ModelSupportConflictError,
    TokenizerBuildContext,
    UnsupportedModelError,
    VisionBuildContext,
    get_model,
    get_model_class,
    get_model_support,
)


_CLASS_EXPORTS = {
    "BloomTpPartModel": "lightllm.models.bloom.model",
    "Deepseek2TpPartModel": "lightllm.models.deepseek2.model",
    "Deepseek3_2TpPartModel": "lightllm.models.deepseek3_2.model",
    "Deepseek3MTPModel": "lightllm.models.deepseek_mtp.model",
    "Gemma3TpPartModel": "lightllm.models.gemma3.model",
    "Gemma4TpPartModel": "lightllm.models.gemma4.model",
    "Gemma_2bTpPartModel": "lightllm.models.gemma_2b.model",
    "Glm4MoeLiteMTPModel": "lightllm.models.glm4_moe_lite_mtp.model",
    "Glm4MoeLiteTpPartModel": "lightllm.models.glm4_moe_lite.model",
    "GptOssTpPartModel": "lightllm.models.gpt_oss.model",
    "InternVLLlamaTpPartModel": "lightllm.models.internvl.model",
    "InternVLPhi3TpPartModel": "lightllm.models.internvl.model",
    "InternVLQwen2TpPartModel": "lightllm.models.internvl.model",
    "InternVLDeepSeek2TpPartModel": "lightllm.models.internvl.model",
    "InternVLInternlm2TpPartModel": "lightllm.models.internvl.model",
    "Internlm2RewardTpPartModel": "lightllm.models.internlm2_reward.model",
    "Internlm2TpPartModel": "lightllm.models.internlm2.model",
    "InternlmTpPartModel": "lightllm.models.internlm.model",
    "LlamaTpPartModel": "lightllm.models.llama.model",
    "LlavaTpPartModel": "lightllm.models.llava.model",
    "MiniCPMTpPartModel": "lightllm.models.minicpm.model",
    "MistralMTPModel": "lightllm.models.mistral_mtp.model",
    "MistralTpPartModel": "lightllm.models.mistral.model",
    "MixtralTpPartModel": "lightllm.models.mixtral.model",
    "Phi3TpPartModel": "lightllm.models.phi3.model",
    "QWenTpPartModel": "lightllm.models.qwen.model",
    "QWenVLTpPartModel": "lightllm.models.qwen_vl.model",
    "Qwen2RewardTpPartModel": "lightllm.models.qwen2_reward.model",
    "Qwen2TpPartModel": "lightllm.models.qwen2.model",
    "Qwen2VLTpPartModel": "lightllm.models.qwen2_vl.model",
    "Qwen3DFlashModel": "lightllm.models.qwen3_dflash.model",
    "Qwen3DSparkModel": "lightllm.models.qwen3_dspark.model",
    "Qwen3EagleModel": "lightllm.models.qwen3_eagle.model",
    "Qwen3MOEModel": "lightllm.models.qwen3_moe.model",
    "Qwen3MOEMTPModel": "lightllm.models.qwen3_moe_mtp.model",
    "Qwen3NextTpPartModel": "lightllm.models.qwen3next.model",
    "Qwen3OmniMOETpPartModel": "lightllm.models.qwen3_omni_moe_thinker.model",
    "Qwen3TpPartModel": "lightllm.models.qwen3.model",
    "Qwen3VLMOETpPartModel": "lightllm.models.qwen3_vl_moe.model",
    "Qwen3VLTpPartModel": "lightllm.models.qwen3_vl.model",
    "Qwen3_5DFlashModel": "lightllm.models.qwen3_5_dflash.model",
    "Qwen3_5DSparkModel": "lightllm.models.qwen3_5_dspark.model",
    "Qwen3_5MOETpPartModel": "lightllm.models.qwen3_5_moe.model",
    "Qwen3_5MoeMTPModel": "lightllm.models.qwen3_5_moe_mtp.model",
    "Qwen3_5MTPModel": "lightllm.models.qwen3_5_mtp.model",
    "Qwen3_5TpPartModel": "lightllm.models.qwen3_5.model",
    "StablelmTpPartModel": "lightllm.models.stablelm.model",
    "Starcoder2TpPartModel": "lightllm.models.starcoder2.model",
    "StarcoderTpPartModel": "lightllm.models.starcoder.model",
    "Tarsier2LlamaTpPartModel": "lightllm.models.tarsier2.model",
    "Tarsier2Qwen2TpPartModel": "lightllm.models.tarsier2.model",
    "Tarsier2Qwen2VLTpPartModel": "lightllm.models.tarsier2.model",
}


def __getattr__(name):
    module_name = _CLASS_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "ModelContext",
    "ModelSupport",
    "ModelSupportConflictError",
    "TokenizerBuildContext",
    "UnsupportedModelError",
    "VisionBuildContext",
    "get_draft_model_class",
    "get_model",
    "get_model_class",
    "get_model_support",
] + sorted(_CLASS_EXPORTS)
