from lightllm.common.attention_state._exports import resolve_export

_EXPORTS = {
    "create_gqa_state": ("lightllm.common.attention_state.full_att.gqa.factory", "create_gqa_state"),
    "GQAAttentionConfig": ("lightllm.common.attention_state.full_att.gqa.formats", "GQAAttentionConfig"),
    "GQAFullAttentionState": ("lightllm.common.attention_state.full_att.gqa.gqa_state", "GQAFullAttentionState"),
    "GroupQuantizedAttentionConfig": (
        "lightllm.common.attention_state.full_att.gqa.formats",
        "GroupQuantizedAttentionConfig",
    ),
    "GroupQuantizedFullAttentionState": (
        "lightllm.common.attention_state.full_att.gqa.group_quantized_state",
        "GroupQuantizedFullAttentionState",
    ),
    "StaticFP8AttentionConfig": ("lightllm.common.attention_state.full_att.gqa.formats", "StaticFP8AttentionConfig"),
    "StaticFP8FullAttentionState": (
        "lightllm.common.attention_state.full_att.gqa.static_fp8_state",
        "StaticFP8FullAttentionState",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())
