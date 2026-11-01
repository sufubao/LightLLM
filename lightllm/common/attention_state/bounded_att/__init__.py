from lightllm.common.attention_state._exports import resolve_export

_EXPORTS = {
    "BoundedAttentionState": ("lightllm.common.attention_state.bounded_att.base_state", "BoundedAttentionState"),
    "ConvSSMStateConfig": ("lightllm.common.attention_state.bounded_att.conv_ssm_types", "ConvSSMStateConfig"),
    "GDNState": ("lightllm.common.attention_state.bounded_att.gdn.gdn_state", "GDNState"),
    "GDNPagesPool": ("lightllm.common.attention_state.storage.page_pool", "GDNPagesPool"),
    "create_gdn_state": ("lightllm.common.attention_state.bounded_att.gdn.gdn_state", "create_gdn_state"),
    "KDAState": ("lightllm.common.attention_state.bounded_att.kda.kda_state", "KDAState"),
    "KDAPagesPool": ("lightllm.common.attention_state.storage.page_pool", "KDAPagesPool"),
    "create_kda_state": ("lightllm.common.attention_state.bounded_att.kda.kda_state", "create_kda_state"),
    "FP8E8M0SlidingWindowConfig": (
        "lightllm.common.attention_state.bounded_att.sliding_window.formats",
        "FP8E8M0SlidingWindowConfig",
    ),
    "FP8E8M0SlidingWindowState": (
        "lightllm.common.attention_state.bounded_att.sliding_window.fp8_e8m0_swa_state",
        "FP8E8M0SlidingWindowState",
    ),
    "create_sliding_window_state": (
        "lightllm.common.attention_state.bounded_att.sliding_window.fp8_e8m0_swa_state",
        "create_sliding_window_state",
    ),
    "WindowCompressorStateConfig": (
        "lightllm.common.attention_state.bounded_att.window_compressor.formats",
        "WindowCompressorStateConfig",
    ),
    "WindowCompressorState": (
        "lightllm.common.attention_state.bounded_att.window_compressor.window_compressor_state",
        "WindowCompressorState",
    ),
    "create_window_compressor_state": (
        "lightllm.common.attention_state.bounded_att.window_compressor.window_compressor_state",
        "create_window_compressor_state",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())
