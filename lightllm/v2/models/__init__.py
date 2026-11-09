"""Model declaration helpers for the isolated v2 execution runtime."""
from .qwen38_flash_next import Qwen38FlashNextModelAdapter


def get_model_adapter(model_config, **kwargs):
    source = model_config.get("text_config", model_config)
    model_type = source.get("model_type")
    if model_type in {"qwen4_exp", "qwen4_exp_text", "qwen3_8_flash_next", "qwen3_8_flash_next_text"}:
        return Qwen38FlashNextModelAdapter(model_config, **kwargs)
    raise ValueError(f"unsupported v2 model type: {model_type}")


__all__ = ["Qwen38FlashNextModelAdapter", "get_model_adapter"]
