import torch

from lightllm.v2.execution.contracts.state import FormatSpec
from lightllm.v2.execution.build import BuildConfig


def default_formats(config, dtype=torch.bfloat16, tp_world_size=1, page_size=1, max_verify_tokens=1):
    return {
        "history": FormatSpec(
            "plain",
            dtype,
            "token_packed_mla",
            ("latent", "indexer"),
            page_size,
            shard=(0, tp_world_size),
            field_dtypes=(dtype, torch.uint8),
        ),
        "positions": FormatSpec(
            "plain",
            dtype,
            "token_head_dim",
            ("k", "v"),
            page_size,
            shard=(0, tp_world_size),
            field_dtypes=(dtype, dtype),
        ),
        "recurrent": FormatSpec(
            "conv_ssm",
            dtype,
            "request_conv_ssm",
            ("conv", "ssm"),
            1,
            shard=(0, tp_world_size),
            field_dtypes=(dtype, torch.float32),
            candidate_slots=max_verify_tokens,
        ),
        "bounded": FormatSpec(
            "lexical",
            dtype,
            "request_lexical",
            ("ngram", "convolution"),
            1,
            field_dtypes=(torch.int64, dtype),
        ),
    }


def requested_phases(config):
    return {
        "history": {"prefill": "triton", "decode": "triton"},
        "recurrent": {"prefill": "triton", "decode": "triton"},
    }


def build_config(adapter, device, limits, weights, *, page_size=1, max_verify_tokens=1):
    return BuildConfig(
        limits=limits,
        formats=default_formats(
            adapter.config,
            dtype=adapter.dtype,
            tp_world_size=adapter.tp_world_size,
            page_size=page_size,
            max_verify_tokens=max_verify_tokens,
        ),
        requested=requested_phases(adapter.config),
        static_params=adapter.attention_params(weights, device),
    )
