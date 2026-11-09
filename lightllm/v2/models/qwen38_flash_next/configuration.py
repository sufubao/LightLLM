from lightllm.common.build_utils import repair_config


def normalize_model_config(checkpoint):
    config = dict(checkpoint.get("text_config", checkpoint))
    for aliases in (
        ["num_attention_heads", "n_head"],
        ["num_key_value_heads", "n_kv_head"],
        ["hidden_size", "n_embd", "n_embed"],
        ["num_hidden_layers", "n_layer"],
    ):
        repair_config(config, same_names=aliases)
    if "full_attention_interval" not in config:
        raise ValueError("Qwen3.8-Flash-Next requires full_attention_interval")
    interval = config["full_attention_interval"]
    depth = config["num_hidden_layers"]
    if not isinstance(interval, int) or interval <= 0 or depth <= 0:
        raise ValueError("full_attention_interval and num_hidden_layers must be positive integers")
    expected = ["full_attention" if (index + 1) % interval == 0 else "linear_attention" for index in range(depth)]
    configured = config.get("layer_types", expected)
    if len(configured) != depth:
        raise ValueError("layer_types must describe every model layer")
    actual = ["full_attention" if kind == "indexed_attention" else kind for kind in configured]
    if actual != expected:
        raise ValueError("Qwen3.8-Flash-Next requires the configured periodic attention layout")
    config["layer_types"] = expected
    if "head_dim" not in config:
        config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]
    config.setdefault("norm_topk_prob", True)
    return config


def is_linear_attention_layer(config, layer_id):
    return config["layer_types"][layer_id] == "linear_attention"
