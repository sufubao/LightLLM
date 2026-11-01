from lightllm.common.attention_state._exports import resolve_export

_EXPORTS = {
    "create_sliding_window_state": (
        "lightllm.common.attention_state.bounded_att.sliding_window.fp8_e8m0_swa_state",
        "create_sliding_window_state",
    ),
    "FP8E8M0SlidingWindowConfig": (
        "lightllm.common.attention_state.bounded_att.sliding_window.formats",
        "FP8E8M0SlidingWindowConfig",
    ),
    "FP8E8M0SlidingWindowState": (
        "lightllm.common.attention_state.bounded_att.sliding_window.fp8_e8m0_swa_state",
        "FP8E8M0SlidingWindowState",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())
