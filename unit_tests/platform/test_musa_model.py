"""Run with torch_musa and the MUSA Triton build installed."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.no_grad()
def test_llama_prefill_prefix_decode_and_graph(dtype, tmp_path, monkeypatch):
    pytest.importorskip("torch_musa")
    if not torch.musa.is_available():
        pytest.skip("MUSA device required")
    from transformers import LlamaConfig, LlamaForCausalLM
    from lightllm.common.basemodel.batch_objs import ModelInput
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.core.runner.decode_graph import CudaGraph
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.models.llama.model import LlamaTpPartModel
    from lightllm.platform.musa import create_context
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type

    args = StartArgs(
        model_dir=str(tmp_path),
        hardware_platform="musa",
        data_type=str(dtype).split(".")[-1],
        page_size=1,
        llm_prefill_att_backend=["triton"],
        llm_decode_att_backend=["triton"],
        disable_symm_mem_allreduce=True,
        disable_flashinfer_allreduce=True,
        running_max_req_size=2,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    monkeypatch.setenv("LIGHTLLM_TORCH_DEVICE_TYPE", "musa")
    monkeypatch.setenv("LIGHTLLM_TRITON_AUTOTUNE_LEVEL", "0")
    for name in (
        "GLOBAL_RANK",
        "GLOBAL_DP_RANK",
        "DP_RANK_IN_NODE",
        "CURRENT_RANK_IN_DP",
        "CURRENT_RANK_IN_NODE",
        "CURRENT_DEVICE_ID",
    ):
        monkeypatch.setenv("LIGHTLLM_" + name, "0")
    for name in ("GLOBAL_WORLD_SIZE", "DP_SIZE", "DP_WORLD_SIZE", "NODE_WORLD_SIZE"):
        monkeypatch.setenv("LIGHTLLM_" + name, "1")
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])
    # Fail immediately if the model or Graph accidentally selects the CUDA runtime.
    for name in ("set_device", "current_stream", "synchronize", "mem_get_info", "get_device_name"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: pytest.fail("unexpected CUDA operation"))
    torch.manual_seed(41)
    configuration = LlamaConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    configuration._attn_implementation = "eager"
    reference = LlamaForCausalLM(configuration).eval()
    configuration.save_pretrained(tmp_path)
    weights = {name: value.to(dtype) for name, value in reference.state_dict().items()}
    context = create_context(config=args)
    runtime = context.device
    model = LlamaTpPartModel(
        dict(
            run_mode="normal",
            weight_dir=str(tmp_path),
            weight_dict=weights,
            max_total_token_num=64,
            batch_max_tokens=16,
            max_req_num=2,
            max_seq_length=16,
            disable_cudagraph=True,
            execution_device=runtime,
            execution_providers=context.providers,
        )
    )
    model.prepare_attention_states()
    model.att_state_manager.finish_state_registration()
    for state in model.att_state_manager.get_full_attention_states():
        state.device = runtime.device
        state.allocate_storage(64)
    model._init_execution_plan()
    model._init_hidden_collector()
    model.prefill_att_backend = model.decode_att_backend = None
    model.prefill_att_backend1 = model.decode_att_backend1 = None
    model.req_manager = SimpleNamespace(
        req_to_token_indexs=torch.arange(32, dtype=torch.int32, device=runtime.device).view(2, 16)
    )
    assert model.linear_plan.bindings and not model.linear_plan.legacy
    assert model.attention_execution is not None
    assert model.batch_runner.runtime is runtime

    def prepare(sequences, cached, prefill):
        lengths = [len(s) for s in sequences]
        new = [s[n:] for s, n in zip(sequences, cached)]
        tensor = lambda x, dtype=torch.int32: torch.tensor(x, dtype=dtype, device=runtime.device)
        inp = ModelInput(
            batch_size=2,
            total_token_num=sum(lengths),
            max_q_seq_len=max(map(len, new)),
            max_kv_seq_len=max(lengths),
            max_cache_len=max(cached),
            input_ids=tensor(sum(new, []), torch.int64),
            b_req_idx=tensor([0, 1]),
            b_seq_len=tensor(lengths),
            b_mtp_index=tensor([0, 0]),
            is_prefill=prefill,
            b_ready_cache_len=tensor(cached) if prefill else None,
            b_prefill_start_loc=tensor([0, len(new[0])]) if prefill else None,
            b_is_decode_req=tensor([False, False], torch.bool) if prefill else None,
            b_prefill_has_output_cpu=[True, True] if prefill else None,
            b_position_delta=None if prefill else tensor([0, 0]),
            b_shared_seq_len=None if prefill else tensor([0, 0]),
            b_shared_radix_node_id=None if prefill else tensor([0, 0]),
            multimodal_params=[{"images": [], "audios": []}, {"images": [], "audios": []}],
        )
        state = model._create_inferstate(inp)
        model.batch_metadata(state)
        state.init_att_state()
        return state

    def check(output, sequences):
        target = torch.cat([reference(torch.tensor([s])).logits[:, -1] for s in sequences])
        actual = output.logits.float().cpu()
        torch.testing.assert_close(actual, target, rtol=0.02, atol=0.012 if dtype == torch.bfloat16 else 0.004)
        assert torch.equal(actual.argmax(-1), target.argmax(-1))

    try:
        sequences = [[11, 12, 13], [21, 22]]
        check(model._context_forward(prepare(sequences, [0, 0], True)), sequences)
        sequences = [[11, 12, 13, 14, 15], [21, 22, 23]]
        check(model._context_forward(prepare(sequences, [3, 2], True)), sequences)
        sequences = [[11, 12, 13, 14, 15, 16], [21, 22, 23, 24]]
        state = prepare(sequences, [5, 3], False)
        check(model._token_forward(state), sequences)
        model.graph = CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=16, runtime=runtime)
        state.is_cuda_graph_capture = True
        captured = model.graph.capture_decode(model._token_forward, state)
        for index in range(3):
            cached = list(map(len, sequences))
            sequences = [s + [30 + index + i] for i, s in enumerate(sequences)]
            state = prepare(sequences, cached, False)
            state.is_cuda_graph_replay = True
            model.graph.replay(state)
            check(captured, sequences)
    finally:
        runtime.synchronize()
        model.graph_runner.close()
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()
