from lightllm.common.attention_state._exports import resolve_export

_EXPORTS = {
    "create_gdn_state": ("lightllm.common.attention_state.bounded_att.gdn.gdn_state", "create_gdn_state"),
    "GDNPagesPool": ("lightllm.common.attention_state.storage.page_pool", "GDNPagesPool"),
    "GDNState": ("lightllm.common.attention_state.bounded_att.gdn.gdn_state", "GDNState"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())
