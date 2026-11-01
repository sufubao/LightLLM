from lightllm.common.attention_state._exports import resolve_export

_EXPORTS = {
    "create_window_compressor_state": (
        "lightllm.common.attention_state.bounded_att.window_compressor.window_compressor_state",
        "create_window_compressor_state",
    ),
    "WindowCompressorState": (
        "lightllm.common.attention_state.bounded_att.window_compressor.window_compressor_state",
        "WindowCompressorState",
    ),
    "WindowCompressorStateConfig": (
        "lightllm.common.attention_state.bounded_att.window_compressor.formats",
        "WindowCompressorStateConfig",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())
