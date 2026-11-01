from lightllm.common.attention_state._exports import resolve_export

_EXPORTS = {
    "create_kda_state": ("lightllm.common.attention_state.bounded_att.kda.kda_state", "create_kda_state"),
    "KDAPagesPool": ("lightllm.common.attention_state.storage.page_pool", "KDAPagesPool"),
    "KDAState": ("lightllm.common.attention_state.bounded_att.kda.kda_state", "KDAState"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())
