__all__ = ["KimiLinearTpPartModel"]


def __getattr__(name):
    if name == "KimiLinearTpPartModel":
        from lightllm.models.kimi_linear.model import KimiLinearTpPartModel

        return KimiLinearTpPartModel
    raise AttributeError(name)
