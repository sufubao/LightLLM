from lightllm.common.attention_state._exports import resolve_export

_EXPORTS = {
    "create_mla_state": ("lightllm.common.attention_state.full_att.mla.mla_state", "create_mla_state"),
    "MLAConfig": ("lightllm.common.attention_state.full_att.mla.formats", "MLAConfig"),
    "MLAFullAttentionState": ("lightllm.common.attention_state.full_att.mla.mla_state", "MLAFullAttentionState"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())
