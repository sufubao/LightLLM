from lightllm.common.attention_state._exports import resolve_export

_EXPORTS = {
    "create_packed_mla_state": (
        "lightllm.common.attention_state.full_att.packed_mla.factory",
        "create_packed_mla_state",
    ),
    "PackedMLAAttentionConfig": (
        "lightllm.common.attention_state.full_att.packed_mla.formats",
        "PackedMLAAttentionConfig",
    ),
    "PackedMLALogicalConfig": ("lightllm.common.attention_state.full_att.packed_mla.formats", "PackedMLALogicalConfig"),
    "PackedMLAFullAttentionState": (
        "lightllm.common.attention_state.full_att.packed_mla.packed_mla_state",
        "PackedMLAFullAttentionState",
    ),
    "FP8E8M0PackedMLAConfig": ("lightllm.common.attention_state.full_att.packed_mla.formats", "FP8E8M0PackedMLAConfig"),
    "FP8E8M0PackedMLAState": (
        "lightllm.common.attention_state.full_att.packed_mla.fp8_e8m0_packed_mla_state",
        "FP8E8M0PackedMLAState",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())
