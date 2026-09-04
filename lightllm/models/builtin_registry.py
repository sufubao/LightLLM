"""Lightweight manifests for built-in model implementations.

Keep this module free of model imports. A module is imported only when its
``model_type`` or speculative mode is resolved.
"""


BUILTIN_MODEL_MODULES = {
    "bloom": ("lightllm.models.bloom.model",),
    "deepseek_v2": ("lightllm.models.deepseek2.model",),
    "deepseek_v3": ("lightllm.models.deepseek2.model",),
    "deepseek_v32": ("lightllm.models.deepseek3_2.model",),
    "gemma": ("lightllm.models.gemma_2b.model",),
    "gemma3": ("lightllm.models.gemma3.model",),
    "gemma4": ("lightllm.models.gemma4.model",),
    "glm4_moe_lite": ("lightllm.models.glm4_moe_lite.model",),
    "gpt_bigcode": ("lightllm.models.starcoder.model",),
    "gpt_oss": ("lightllm.models.gpt_oss.model",),
    "internlm": ("lightllm.models.internlm.model",),
    "internlm2": ("lightllm.models.internlm2.model", "lightllm.models.internlm2_reward.model"),
    "internvl_chat": ("lightllm.models.internvl.model",),
    "llama": ("lightllm.models.llama.model",),
    "llava": ("lightllm.models.llava.model", "lightllm.models.tarsier2.support"),
    "minicpm": ("lightllm.models.minicpm.model",),
    "mistral": ("lightllm.models.mistral.model",),
    "mixtral": ("lightllm.models.mixtral.model",),
    "phi3": ("lightllm.models.phi3.model",),
    "qwen": ("lightllm.models.qwen.model", "lightllm.models.qwen_vl.model"),
    "qwen2": ("lightllm.models.qwen2.model", "lightllm.models.qwen2_reward.model"),
    "qwen2_5_vl": ("lightllm.models.qwen2_vl.model",),
    "qwen2_vl": ("lightllm.models.qwen2_vl.model",),
    "qwen3": ("lightllm.models.qwen3.model",),
    "qwen3_5": ("lightllm.models.qwen3_5.model",),
    "qwen3_5_moe": ("lightllm.models.qwen3_5_moe.model",),
    "qwen3_moe": ("lightllm.models.qwen3_moe.model",),
    "qwen3_next": ("lightllm.models.qwen3next.model",),
    "qwen3_omni_moe": ("lightllm.models.qwen3_omni_moe_thinker.model",),
    "qwen3_vl": ("lightllm.models.qwen3_vl.support",),
    "qwen3_vl_moe": ("lightllm.models.qwen3_vl_moe.model",),
    "stablelm": ("lightllm.models.stablelm.model",),
    "starcoder2": ("lightllm.models.starcoder2.model",),
}


BUILTIN_DRAFT_MODULES = {
    ("deepseek_v3", "vanilla_with_att"): "lightllm.models.deepseek_mtp.model",
    ("deepseek_v3", "eagle_with_att"): "lightllm.models.deepseek_mtp.model",
    ("glm4_moe_lite", "vanilla_with_att"): "lightllm.models.glm4_moe_lite_mtp.model",
    ("glm4_moe_lite", "eagle_with_att"): "lightllm.models.glm4_moe_lite_mtp.model",
    ("mistral", "vanilla_no_att"): "lightllm.models.mistral_mtp.model",
    ("mistral", "eagle_no_att"): "lightllm.models.mistral_mtp.model",
    ("qwen3", "dflash"): "lightllm.models.qwen3_dflash.model",
    ("qwen3", "dspark"): "lightllm.models.qwen3_dspark.model",
    ("qwen3", "eagle3"): "lightllm.models.qwen3_eagle.model",
    ("qwen3_5", "dflash"): "lightllm.models.qwen3_5_dflash.model",
    ("qwen3_5_text", "dflash"): "lightllm.models.qwen3_5_dflash.model",
    ("qwen3_5", "dspark"): "lightllm.models.qwen3_5_dspark.model",
    ("qwen3_5_text", "dspark"): "lightllm.models.qwen3_5_dspark.model",
    ("qwen3_5", "vanilla_with_att"): "lightllm.models.qwen3_5_mtp.model",
    ("qwen3_5", "eagle_with_att"): "lightllm.models.qwen3_5_mtp.model",
    ("qwen3_5_text", "vanilla_with_att"): "lightllm.models.qwen3_5_mtp.model",
    ("qwen3_5_text", "eagle_with_att"): "lightllm.models.qwen3_5_mtp.model",
    ("qwen3_5_moe", "vanilla_with_att"): "lightllm.models.qwen3_5_moe_mtp.model",
    ("qwen3_5_moe", "eagle_with_att"): "lightllm.models.qwen3_5_moe_mtp.model",
    ("qwen3_5_moe_text", "vanilla_with_att"): "lightllm.models.qwen3_5_moe_mtp.model",
    ("qwen3_5_moe_text", "eagle_with_att"): "lightllm.models.qwen3_5_moe_mtp.model",
    ("qwen3_moe", "vanilla_no_att"): "lightllm.models.qwen3_moe_mtp.model",
    ("qwen3_moe", "eagle_no_att"): "lightllm.models.qwen3_moe_mtp.model",
}
