from lightllm.models.qwen4_exp.model import normalize_qwen4_text_config


def test_normalize_official_qwen38_flash_next_config():
    all_config = {
        "model_type": "qwen4_exp",
        "vision_config": {"out_hidden_size": 2560},
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "ple_layer_ids": [2],
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
            },
        },
    }

    config, vision_config = normalize_qwen4_text_config(all_config)

    assert config["model_type"] == "qwen4_exp_text"
    assert config["n_layer"] == 48
    assert config["n_embed"] == 2560
    assert config["partial_rotary_factor"] == 0.25
    assert config["rope_theta"] == 10_000_000
    assert config["norm_topk_prob"] is True
    assert config["seed"] == 1234
    assert config["split_ngram_parts"] == 128
    assert config["ple_padded_vocab_size"] == 320_001_536
    assert vision_config["out_hidden_size"] == config["hidden_size"]
