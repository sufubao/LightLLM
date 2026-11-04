from dataclasses import asdict, replace
from types import SimpleNamespace
from pathlib import Path
import json
import importlib
import hashlib
import os
import sys

import pytest
import torch

from lightllm.models.deepseek_v41.config import DeepseekV41Config


def tiny_config():
    return DeepseekV41Config(
        vocab_size=128,
        dim=128,
        moe_inter_dim=128,
        n_layers=6,
        n_heads=4,
        n_routed_experts=4,
        n_activated_experts=2,
        q_lora_rank=64,
        head_dim=64,
        rope_head_dim=32,
        o_groups=2,
        o_lora_rank=64,
        window_size=4,
        compress_ratios=(0, 2, 2, 1, 1, 1),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 5),
        index_n_heads=2,
        index_head_dim=64,
        index_topk=3,
        candidate_source_layer=3,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        original_seq_len=0,
        dtype="bf16",
        expert_dtype=None,
    )


def test_config_rejects_incompatible_shared_sources():
    config = tiny_config()
    assert config.sources(5) == (3, 5)
    with pytest.raises(ValueError, match="compatible shared"):
        replace(config, compress_ratios=(0, 2, 2, 1, 2, 1))
    with pytest.raises(ValueError, match="TP size"):
        config.validate_parallelism(3)


def make_engine(config, state_dict, directory="/nonexistent", max_seq_len=64):
    from lightllm.models.deepseek_v41.adapter import DeepseekV41Adapter
    from lightllm.core.device.cuda import CudaRuntime
    from lightllm.core.build import build, BuildConfig
    from lightllm.core.contracts.device import ExecutionLimits
    from lightllm.common.basemodel.weight_binding import bind_linears, bind_moe
    from lightllm.common.attention_state.full_att.base_state import FullAttentionState

    runtime = CudaRuntime()
    adapter = DeepseekV41Adapter(config, role="main", runtime=runtime, request_capacity=3, max_seq_len=max_seq_len)
    pre, layers = adapter.build_weights()
    adapter.load_weights((pre, layers), directory, state_dict)
    from lightllm.core.registry import default_providers

    providers = adapter.providers(default_providers())
    bind_linears((pre, *layers), torch.empty, providers=providers, runtime=runtime)
    bind_moe(adapter.moe_specs(), layers, adapter.recipes, providers=providers, runtime=runtime)
    states = []
    manager = SimpleNamespace(_request_capacity=3, register_state=lambda role, state, **kwargs: states.append(state))

    def allocate(execution):
        for state in states:
            if isinstance(state, FullAttentionState):
                state.device = runtime.device
                state.allocate_storage(3 * max_seq_len)

    execution = build(
        model_adapter=adapter,
        providers=providers,
        device=runtime,
        config=BuildConfig(
            adapter.recipes,
            manager,
            ExecutionLimits(2 ** 28, 3, max_seq_len),
            adapter.formats["history"],
            {"prefill": "auto", "decode": "auto"},
            static_params={
                i: adapter.static_params(spec, layers[i], device=runtime)
                for i, spec in adapter.attention_specs().items()
            },
            window_format=adapter.formats["window"],
            allocate_storage=allocate,
        ),
    )
    table = torch.arange(3 * max_seq_len, device="cuda", dtype=torch.int32).view(3, max_seq_len)
    return adapter, execution, table


def run_batch(adapter, table, requests, tokens, phase):
    from lightllm.core.compute.attention.shared_compressed.backend import SharedCompressedMetadata

    offsets, offset = [], 0
    for request, start, ids in requests:
        offsets.append((request, start, len(ids), offset))
        offset += len(ids)
    metadata = SharedCompressedMetadata(offsets, table)
    ids = torch.tensor(tokens, device="cuda", dtype=torch.long)
    return adapter.forward(ids, metadata, phase)


@pytest.fixture
def official_reference():
    root = os.getenv("DEEPSEEK_V41_REFERENCE")
    if not root:
        pytest.skip("set DEEPSEEK_V41_REFERENCE to the pinned official inference source")
    hashes = {
        "model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
        "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
        "engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    }
    for name, digest in hashes.items():
        assert hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() == digest
    sys.path.insert(0, root)
    try:
        yield importlib.import_module("model")
    finally:
        sys.path.remove(root)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA Hopper or newer")
@pytest.mark.parametrize("quantized", [False, True])
@torch.inference_mode()
def test_official_reference_variable_batches_and_reuse(tmp_path, quantized, official_reference):
    reference = official_reference
    config = tiny_config()
    tokenizer = None
    if quantized:
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast
        from engram import EngramLayout

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel({f"t{i}": i for i in range(config.vocab_size)}, unk_token="t0"))
        )
        tokenizer.save_pretrained(tmp_path)
        config = replace(
            config,
            dtype="fp8",
            expert_dtype="fp4",
            engram_layer_ids=(1, 3),
            engram_num_embeddings=(1, 1),
            engram_max_ngram_size=3,
            engram_vocab_size=17,
            engram_n_heads=2,
            engram_head_dim=64,
            engram_compressed_vocab_size=config.vocab_size,
        )
        layout = EngramLayout.from_args(config)
        config = replace(config, engram_num_embeddings=tuple(sum(sum(v) for v in layer) for layer in layout.primes))
    args = reference.ModelArgs(**asdict(config), max_batch_size=1, max_seq_len=64, n_mtp_layers=0, temperature=0)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            expected = reference.Transformer(args, tokenizer=tokenizer)
        torch.manual_seed(41)
        for name, parameter in expected.named_parameters():
            if "norm.weight" in name:
                parameter.fill_(1)
            elif name.endswith("attn_sink"):
                parameter.zero_()
            elif parameter.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2):
                continue
            else:
                parameter.normal_(0, 0.03)
        if quantized:
            from kernel import fp4_act_quant

            for module in expected.modules():
                weight = getattr(module, "weight", None)
                if weight is None:
                    continue
                if weight.dtype == torch.float4_e2m1fn_x2:
                    full = torch.randn(weight.shape[0], weight.shape[1] * 2, device="cuda", dtype=torch.bfloat16) * 0.03
                    packed, scale = fp4_act_quant(full, 32, False)
                    weight.copy_(packed)
                    module.scale.copy_(scale)
                elif weight.dtype == torch.float8_e4m3fn:
                    full = torch.randn(weight.shape, device="cuda", dtype=torch.bfloat16) * 0.03
                    module.scale.fill_(2 ** -12)
                    weight.copy_((full.float() / 2 ** -12).clamp(-448, 448).to(weight.dtype))
        checkpoint = dict(expected.state_dict())
        if quantized:
            for layer in expected.layers:
                matrix = layer.attn.wo_a.weight
                encoded = (matrix.float() / 2 ** -12).clamp(-448, 448).to(torch.float8_e4m3fn)
                scale = torch.full(
                    (matrix.shape[0] // 32, matrix.shape[1] // 32), 2 ** -12, device="cuda", dtype=torch.float8_e8m0fnu
                )
                prefix = f"layers.{layer.layer_id}.attn.wo_a"
                checkpoint[prefix + ".weight"], checkpoint[prefix + ".scale"] = encoded, scale
                matrix.copy_(encoded.float() * 2 ** -12)
        from safetensors.torch import save_file

        directory = Path(os.getenv("DEEPSEEK_V41_FIXTURE_DIR", str(tmp_path))) if quantized else tmp_path
        directory.mkdir(parents=True, exist_ok=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(directory)
        save_file(
            {
                key: (value.view(torch.int8) if value.dtype == torch.float4_e2m1fn_x2 else value)
                .detach()
                .cpu()
                .contiguous()
                for key, value in checkpoint.items()
            },
            str(directory / "model.safetensors"),
        )
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors" for key in checkpoint}})
        )
        adapter, execution, table = make_engine(config, None, str(directory))
        histories = {0: [3, 7, 4], 1: [5, 11, 13, 17, 19, 23]}
        orders = [("prefill", [0, 1]), ("decode", [1, 0]), ("decode", [0, 1]), ("decode", [1, 0])]
        for step, (phase, order) in enumerate(orders):
            requests, flat = [], []
            for req in order:
                if phase == "decode":
                    histories[req].append(29 + step + req)
                chunk = histories[req] if phase == "prefill" else histories[req][-1:]
                requests.append((req, len(histories[req]) - len(chunk), chunk))
                flat.extend(chunk)
            result = run_batch(adapter, table, requests, flat, phase)
            for row, req in enumerate(order):
                with torch.device("cuda"):
                    _, golden, _ = expected(torch.tensor([histories[req]], device="cuda"), 0)
                torch.testing.assert_close(result[row], golden[0], rtol=0.025, atol=0.02)
                relative = (result[row] - golden[0]).square().mean().sqrt() / golden[0].square().mean().sqrt()
                assert relative < 0.01, relative
        histories[0] = [41, 43, 47, 53, 59]
        result = run_batch(adapter, table, [(0, 0, histories[0])], histories[0], "prefill")
        with torch.device("cuda"):
            _, golden, _ = expected(torch.tensor([histories[0]], device="cuda"), 0)
        torch.testing.assert_close(result[0], golden[0], rtol=0.025, atol=0.02)
        for left, right in ((0, 1), (1, 3), (3, 5)):
            ids = histories[0][left:right]
            result = run_batch(adapter, table, [(0, left, ids)], ids, "prefill")
        torch.testing.assert_close(result[0], golden[0], rtol=0.025, atol=0.02)
        assert (
            adapter.recipes[2].bound_decode_attention.compressed.fields[0].data_ptr()
            == adapter.recipes[1].bound_decode_attention.compressed.fields[0].data_ptr()
        )
        assert (
            adapter.recipes[2].bound_decode_attention.window.data_ptr()
            != adapter.recipes[1].bound_decode_attention.window.data_ptr()
        )
        if quantized and os.getenv("DEEPSEEK_V41_FIXTURE_DIR"):
            hf = asdict(config)
            aliases = dict(
                dim="hidden_size",
                moe_inter_dim="moe_intermediate_size",
                n_layers="num_hidden_layers",
                n_heads="num_attention_heads",
                n_activated_experts="num_experts_per_tok",
                rope_head_dim="qk_rope_head_dim",
                window_size="sliding_window",
                norm_eps="rms_norm_eps",
                kv_source_layers="kv_source_layer_ids",
                index_source_layers="index_source_layer_ids",
                candidate_source_layer="candidate_source_layer_id",
                route_scale="routed_scaling_factor",
                score_func="scoring_func",
                engram_pad_id="engram_pad_token_id",
            )
            for old, new in aliases.items():
                hf[new] = hf.pop(old)
            hf.update(model_type="deepseek_v41_text", num_key_value_heads=1, max_position_embeddings=64)
            root_config = dict(
                model_type="deepseek_v41",
                architectures=["DeepseekV41ForCausalLM"],
                text_config=hf,
                bos_token_id=0,
                eos_token_id=1,
                pad_token_id=2,
                quantization_config=dict(
                    quant_method="fp8", weight_block_size=[32, 32], scale_fmt="ue8m0", expert_dtype="fp4"
                ),
            )
            (directory / "config.json").write_text(json.dumps(root_config))
            (directory / "generation_config.json").write_text(
                json.dumps(dict(bos_token_id=0, eos_token_id=1, pad_token_id=2))
            )
            golden_results = []
            for prompt in ([3, 7, 4], [5, 11, 13, 17, 19, 23]):
                sequence, generated = list(prompt), []
                for _ in range(4):
                    with torch.device("cuda"):
                        _, logits, _ = expected(torch.tensor([sequence], device="cuda"), 0)
                    generated.append(int(logits.argmax(-1)))
                    sequence.append(generated[-1])
                golden_results.append(dict(prompt=prompt, output=generated))
            (directory / "golden.json").write_text(json.dumps(golden_results))
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA Hopper or newer")
@torch.inference_mode()
def test_window_prefill_and_wrap(official_reference):
    config = replace(
        tiny_config(),
        n_layers=1,
        window_size=128,
        compress_ratios=(0,),
        kv_source_layers=(),
        index_source_layers=(),
        candidate_source_layer=-1,
    )
    args = official_reference.ModelArgs(**asdict(config), max_batch_size=1, max_seq_len=192, n_mtp_layers=0)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            expected = official_reference.Transformer(args)
        torch.manual_seed(42)
        for name, parameter in expected.named_parameters():
            if "norm.weight" in name:
                parameter.fill_(1)
            elif name.endswith("attn_sink"):
                parameter.zero_()
            else:
                parameter.normal_(0, 0.03)
        adapter, execution, table = make_engine(config, dict(expected.state_dict()), max_seq_len=192)
        for length in (3, 144):
            tokens = [i % config.vocab_size for i in range(length)]
            result = run_batch(adapter, table, [(0, 0, tokens)], tokens, "prefill")
            with torch.device("cuda"):
                _, golden, _ = expected(torch.tensor([tokens], device="cuda"), 0)
            torch.testing.assert_close(result, golden, rtol=0.025, atol=0.02)
            for start in range(length, length + 3):
                token = start % config.vocab_size
                result = run_batch(adapter, table, [(0, start, [token])], [token], "decode")
                with torch.device("cuda"):
                    _, golden, _ = expected(torch.tensor([[token]], device="cuda"), start)
                torch.testing.assert_close(result, golden, rtol=0.025, atol=0.02)
                relative = (result - golden).square().mean().sqrt() / golden.square().mean().sqrt()
                assert relative < 0.01, relative
    finally:
        torch.set_default_dtype(previous)
