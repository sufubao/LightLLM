from lightllm.common.attention_state._exports import resolve_export

_EXPORTS = {
    "create_gqa_state": ("lightllm.common.attention_state.full_att.gqa.factory", "create_gqa_state"),
    "create_mla_state": ("lightllm.common.attention_state.full_att.mla.mla_state", "create_mla_state"),
    "create_packed_mla_state": (
        "lightllm.common.attention_state.full_att.packed_mla.factory",
        "create_packed_mla_state",
    ),
    "FullAttentionState": ("lightllm.common.attention_state.full_att.base_state", "FullAttentionState"),
    "GQAFullAttentionState": ("lightllm.common.attention_state.full_att.gqa.gqa_state", "GQAFullAttentionState"),
    "GQAAttentionConfig": ("lightllm.common.attention_state.full_att.gqa.formats", "GQAAttentionConfig"),
    "MLAConfig": ("lightllm.common.attention_state.full_att.mla.formats", "MLAConfig"),
    "MLAFullAttentionState": ("lightllm.common.attention_state.full_att.mla.mla_state", "MLAFullAttentionState"),
    "GroupQuantizedFullAttentionState": (
        "lightllm.common.attention_state.full_att.gqa.group_quantized_state",
        "GroupQuantizedFullAttentionState",
    ),
    "GroupQuantizedAttentionConfig": (
        "lightllm.common.attention_state.full_att.gqa.formats",
        "GroupQuantizedAttentionConfig",
    ),
    "StaticFP8FullAttentionState": (
        "lightllm.common.attention_state.full_att.gqa.static_fp8_state",
        "StaticFP8FullAttentionState",
    ),
    "StaticFP8AttentionConfig": ("lightllm.common.attention_state.full_att.gqa.formats", "StaticFP8AttentionConfig"),
    "PackedMLAFullAttentionState": (
        "lightllm.common.attention_state.full_att.packed_mla.packed_mla_state",
        "PackedMLAFullAttentionState",
    ),
    "PackedMLAAttentionConfig": (
        "lightllm.common.attention_state.full_att.packed_mla.formats",
        "PackedMLAAttentionConfig",
    ),
    "PackedMLALogicalConfig": ("lightllm.common.attention_state.full_att.packed_mla.formats", "PackedMLALogicalConfig"),
    "PackedMLALayerView": ("lightllm.common.attention_state.types", "PackedMLALayerView"),
    "create_compressed_att_state": (
        "lightllm.common.attention_state.full_att.compressed_att.compressed_att_state",
        "create_compressed_att_state",
    ),
    "CompressedAttentionConfig": (
        "lightllm.common.attention_state.full_att.compressed_att.formats",
        "CompressedAttentionConfig",
    ),
    "CompressedAttentionState": (
        "lightllm.common.attention_state.full_att.compressed_att.compressed_att_state",
        "CompressedAttentionState",
    ),
    "CompressedAttentionLayerView": ("lightllm.common.attention_state.types", "CompressedAttentionLayerView"),
    "FP8E8M0PackedMLAConfig": ("lightllm.common.attention_state.full_att.packed_mla.formats", "FP8E8M0PackedMLAConfig"),
    "FP8E8M0PackedMLAState": (
        "lightllm.common.attention_state.full_att.packed_mla.fp8_e8m0_packed_mla_state",
        "FP8E8M0PackedMLAState",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())
