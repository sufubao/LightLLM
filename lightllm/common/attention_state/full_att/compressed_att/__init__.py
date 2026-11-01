from lightllm.common.attention_state._exports import resolve_export

_EXPORTS = {
    "CompressedAttentionConfig": (
        "lightllm.common.attention_state.full_att.compressed_att.formats",
        "CompressedAttentionConfig",
    ),
    "CompressedAttentionLayerView": ("lightllm.common.attention_state.types", "CompressedAttentionLayerView"),
    "CompressedAttentionState": (
        "lightllm.common.attention_state.full_att.compressed_att.compressed_att_state",
        "CompressedAttentionState",
    ),
    "create_compressed_att_state": (
        "lightllm.common.attention_state.full_att.compressed_att.compressed_att_state",
        "create_compressed_att_state",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())
