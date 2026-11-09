"""Isolated LightLLM v2 execution runtime.

The v2 package is additive. It owns its contracts, planner, providers,
storage, runner, attention-state manager, and context-parallel contracts
without changing the legacy ``lightllm.common`` execution path.
"""

from .execution import BuildConfig, ProviderRegistry, StateProvider, AttentionProvider, build, prepare

__all__ = [
    "AttentionProvider",
    "BuildConfig",
    "ProviderRegistry",
    "StateProvider",
    "build",
    "prepare",
    "V2ExecutionSession",
]


def __getattr__(name):
    if name == "V2ExecutionSession":
        from .runtime import V2ExecutionSession

        return V2ExecutionSession
    raise AttributeError(name)
