import os
import torch
import numpy as np
from multiprocessing import Queue
import multiprocessing
from transformers import PretrainedConfig
from lightllm.utils.dist_utils import init_distributed_env, get_current_rank_in_dp
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models import get_model
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.core.objs.start_args_type import StartArgs
from torch.profiler import profile, ProfilerActivity
from lightllm.utils.log_utils import init_logger
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.models.qwen3_moe_mtp.model import Qwen3MOEMTPModel
from lightllm.models.mistral_mtp.model import MistralMTPModel
from lightllm.models.glm4_moe_lite_mtp.model import Glm4MoeLiteMTPModel

logger = init_logger(__name__)


def init_mtp_model(args: StartArgs, kvargs, main_model):
    draft_models = []

    os.environ["DISABLE_CHECK_MAX_LEN_INFER"] = "1"

    if args.mtp_mode in ["vanilla_with_att", "vanilla_no_att"]:
        num_mtp_modules = args.mtp_step
    elif args.mtp_mode in ["eagle_with_att", "eagle_no_att"]:
        num_mtp_modules = 1
    else:
        assert False, f"error mtp mode {args.mtp_mode}"

    for i in range(num_mtp_modules):
        mtp_model_cfg, _ = PretrainedConfig.get_config_dict(args.mtp_draft_model_dir[i])
        model_type = mtp_model_cfg.get("model_type", "")
        mtp_model_kvargs = {
            "weight_dir": args.mtp_draft_model_dir[i],
            "max_total_token_num": main_model.mem_manager.size,
            "load_way": kvargs["load_way"],
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "is_token_healing": False,
            "return_all_prompt_logics": False,
            "disable_chunked_prefill": args.disable_chunked_prefill,
            "data_type": kvargs.get("data_type", "float16"),
            "graph_max_batch_size": kvargs.get("graph_max_batch_size", 16),
            "graph_max_len_in_batch": kvargs.get("graph_max_len_in_batch", 8196),
            "disable_cudagraph": kvargs.get("disable_cudagraph", False),
            "mem_fraction": kvargs["mem_fraction"],
            "batch_max_tokens": kvargs.get("batch_max_tokens", None),
            "quant_type": kvargs.get("quant_type", None),
            "quant_cfg": kvargs.get("quant_cfg", None),
            "run_mode": "normal",
            "llm_prefill_att_backend": kvargs.get("llm_prefill_att_backend", args.llm_prefill_att_backend),
            "llm_decode_att_backend": kvargs.get("llm_decode_att_backend", args.llm_decode_att_backend),
            "vit_att_backend": kvargs.get("vit_att_backend", args.vit_att_backend),
            "llm_kv_type": kvargs.get("llm_kv_type", args.llm_kv_type),
            "llm_kv_quant_group_size": kvargs.get("llm_kv_quant_group_size", args.llm_kv_quant_group_size),
            "main_model": main_model,
            "mtp_previous_draft_models": draft_models.copy(),
            "mtp_mode": args.mtp_mode,
        }

        if model_type == "deepseek_v3":
            assert args.mtp_mode in ["vanilla_with_att", "eagle_with_att"]
            draft_models.append(Deepseek3MTPModel(mtp_model_kvargs))
        elif model_type == "qwen3_moe":
            assert args.mtp_mode in ["vanilla_no_att", "eagle_no_att"]
            draft_models.append(Qwen3MOEMTPModel(mtp_model_kvargs))
        elif model_type == "mistral":
            assert args.mtp_mode in ["vanilla_no_att", "eagle_no_att"]
            draft_models.append(MistralMTPModel(mtp_model_kvargs))
        elif mtp_model_cfg["model_type"] == "glm4_moe_lite":
            assert args.mtp_mode in ["vanilla_with_att", "eagle_with_att"]
            draft_models.append(Glm4MoeLiteMTPModel(mtp_model_kvargs))
        elif model_type in ("qwen3_5", "qwen3_5_text"):
            assert args.mtp_mode in ["vanilla_with_att", "eagle_with_att"]
            from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel

            draft_models.append(Qwen3_5MTPModel(mtp_model_kvargs))
        elif model_type in ("qwen3_5_moe", "qwen3_5_moe_text"):
            assert args.mtp_mode in ["vanilla_with_att", "eagle_with_att"]
            from lightllm.models.qwen3_5_moe_mtp.model import Qwen3_5MoeMTPModel

            draft_models.append(Qwen3_5MoeMTPModel(mtp_model_kvargs))
        else:
            raise ValueError(f"Unsupported MTP model type: {model_type}")

        logger.info(f"loaded mtp model class {draft_models[i].__class__}")
    return draft_models


def test_model_inference_mtp(args):
    ans_queue = Queue()
    workers = []
    dp_size = args.get("dp", 1)

    for rank_id in range(args.node_rank * args.tp, (args.node_rank + 1) * args.tp):
        model_kvargs = {
            "args": args,
            "nccl_host": args.nccl_host,
            "data_type": args.data_type,
            "nccl_port": args.nccl_port,
            "rank_id": rank_id,
            "world_size": args.tp,
            "dp_size": dp_size,
            "weight_dir": args.model_dir,
            "quant_type": args.quant_type,
            "load_way": "HF",
            "max_total_token_num": args.max_total_token_num,
            "graph_max_len_in_batch": args.max_req_total_len,
            "graph_max_batch_size": args.graph_max_batch_size,
            "mem_fraction": args.mem_fraction,
            # Static bench runs explicit batch sizes (<= a few hundred). The hybrid Qwen3.5
            # GDN req-state cache is sized max_req_num * (mtp_step + 1) at ~34 MB/slot, so the
            # old default of 2000 alloc'd ~140 GB and OOM'd under MTP. 512 covers any realistic
            # static batch sweep while keeping the GDN cache small.
            "max_req_num": 512,
            "batch_max_tokens": 2048,
            "run_mode": "normal",
            "max_seq_length": args.max_req_total_len,
            "disable_cudagraph": args.disable_cudagraph,
            "quant_cfg": args.quant_cfg,
            "llm_prefill_att_backend": args.llm_prefill_att_backend,
            "llm_decode_att_backend": args.llm_decode_att_backend,
            "vit_att_backend": args.vit_att_backend,
            "llm_kv_type": args.llm_kv_type,
            "llm_kv_quant_group_size": args.llm_kv_quant_group_size,
        }
        proc = multiprocessing.Process(
            target=tppart_model_infer,
            args=(args, model_kvargs, args.batch_size, args.input_len, args.output_len, ans_queue),
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    assert not ans_queue.empty()
    while not ans_queue.empty():
        assert ans_queue.get()
    return


def torch_profile(fn, log_dir=None):
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    ) as prof:
        fn()
    if get_current_rank_in_dp() == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def run_forward_once(args, input_len, output_len, batch_size, main_model, draft_models, warmup=False):
    import time

    torch.cuda.synchronize()
    prefill_start_time = time.time()

    test_data = np.vstack([np.random.randint(0, 50256, input_len) for _ in range(batch_size)])
    test_data = test_data.reshape(-1)
    test_data = torch.from_numpy(test_data)

    b_req_idx = torch.tensor(
        [main_model.req_manager.alloc() for _ in range(batch_size)], dtype=torch.int32, device="cpu"
    )
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    for i in range(batch_size):
        b_seq_len[i] = input_len

    total_token_num = input_len * batch_size
    mem_indexes = main_model.req_manager.mem_manager.alloc(test_data.shape[0])
    b_mtp_index = torch.zeros(batch_size, dtype=torch.int32)
    b_prefill_start_loc = b_seq_len.cumsum(dim=0, dtype=torch.int32) - b_seq_len
    # Main model Prefill
    model_input = ModelInput(
        batch_size=batch_size,
        total_token_num=total_token_num,
        max_q_seq_len=input_len,
        max_kv_seq_len=input_len,
        max_cache_len=0,
        input_ids=test_data,
        mem_indexes_cpu=mem_indexes,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=b_seq_len,
        is_prefill=True,
        b_ready_cache_len=b_ready_cache_len,
        b_prefill_start_loc=b_prefill_start_loc,
        prefix_total_token_num=0,
        multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
    )

    model_output: ModelOutput = main_model.forward(model_input)
    prob_out = torch.softmax(model_output.logits, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    draft_ids = [predict_ids]

    # Draft model Prefill
    # For simplicity, we'll just take the input of main_model to draft model.
    model_input.mtp_draft_input_hiddens = model_output.mtp_main_output_hiddens
    for draft_model_id in range(len(draft_models)):
        draft_model = draft_models[draft_model_id]
        model_output = draft_model.forward(model_input)
        prob_out = torch.softmax(model_output.logits, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
        draft_ids.append(predict_ids)
        model_input.mtp_draft_input_hiddens = model_output.mtp_main_output_hiddens

    torch.cuda.synchronize()
    prefill_end_time = time.time()
    if get_current_rank_in_dp() == 0 and not warmup:
        print("prefill time cost:", (prefill_end_time - prefill_start_time) * 1000)
        print(
            f"Prefill throughput: {batch_size * input_len * args.dp / (prefill_end_time - prefill_start_time)} tokens/s"
        )

    torch.cuda.synchronize()

    # Speculative width = args.mtp_step in BOTH modes (mirrors base_backend: self.mtp_step =
    # args.mtp_step). The number of draft MODEL INSTANCES differs: vanilla loads mtp_step
    # instances (each forwarded once), eagle loads ONE instance forwarded mtp_step times
    # (chunked_prefill/impl.py: draft_models[_step % num_instances]). The verify batch always
    # expands to (mtp_step + 1) rows per request.
    spec_width = args.mtp_step
    num_instances = len(draft_models)
    # The draft prefill above produced (1 + num_instances) columns; pad/truncate to
    # (spec_width + 1) so the decode verify batch matches the server's expand width. Only the
    # SHAPE matters for throughput here (argmax over random inputs); token values do not.
    while len(draft_ids) < spec_width + 1:
        draft_ids.append(draft_ids[-1])
    draft_ids = draft_ids[: spec_width + 1]
    decode_input_ids = np.stack(draft_ids, axis=-1).reshape(-1)
    decode_input_ids = torch.from_numpy(decode_input_ids)
    mtp_step = spec_width

    # build main decode input:
    nopad_b_seq_idx = []
    nopad_b_seq_len = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0

    for i in range(batch_size):
        nopad_b_seq_idx.append(b_req_idx[i].item())
        seq_len = b_seq_len[i].item()
        nopad_b_seq_len.append(seq_len + 1)
        nopad_total_token_num += seq_len + 1
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len + 1)

        for step in range(mtp_step):
            nopad_b_seq_idx.append(b_req_idx[i].item())
            nopad_b_seq_len.append(seq_len + step + 2)
            nopad_total_token_num += seq_len + step + 2
            nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len + step + 2)

    nopad_b_seq_idx = torch.tensor(nopad_b_seq_idx, dtype=torch.int32, device="cpu")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cpu")
    b_mtp_index = torch.arange(mtp_step + 1, dtype=torch.int32).repeat(batch_size)
    mem_indexes = main_model.req_manager.mem_manager.alloc(batch_size * (mtp_step + 1))

    model_input = ModelInput(
        batch_size=batch_size * (mtp_step + 1),
        total_token_num=nopad_total_token_num,
        max_q_seq_len=1,
        max_kv_seq_len=nopad_max_len_in_batch,
        input_ids=decode_input_ids,
        mem_indexes_cpu=mem_indexes,
        b_req_idx=nopad_b_seq_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=nopad_b_seq_len,
        is_prefill=False,
        multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size * (mtp_step + 1))],
    )

    # Main decode
    for i in range(0, output_len, mtp_step + 1):
        torch.cuda.synchronize()
        step_start_time = time.time()
        model_output = main_model.forward(
            model_input,
        )
        prob_out = torch.softmax(model_output.logits, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)

        # draft decode: mtp_step forwards, reusing draft_models[_step % num_instances]
        # (eagle: one instance reused mtp_step times; vanilla: a distinct instance per step).
        model_input.input_ids = predict_ids.reshape(-1)
        model_input.mtp_draft_input_hiddens = model_output.mtp_main_output_hiddens

        for _step in range(mtp_step):
            draft_model = draft_models[_step % num_instances]
            model_output = draft_model.forward(
                model_input,
            )
            prob_out = torch.softmax(model_output.logits, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            model_input.input_ids = predict_ids.reshape(-1)
            model_input.mtp_draft_input_hiddens = model_output.mtp_main_output_hiddens

        # accept all draft ids by default.
        model_input.input_ids = predict_ids.reshape(-1)
        model_input.mtp_draft_input_hiddens = model_output.mtp_main_output_hiddens
        torch.cuda.synchronize()
        if i % 100 == 0 or i == output_len - 1:
            step_end_time = time.time()
            if get_current_rank_in_dp() == 0 and not warmup:
                step_time = step_end_time - step_start_time
                print(i, " step cost time:", step_time * 1000)
                print(f"Decode throughput: {batch_size * (mtp_step + 1) * args.dp / step_time} tokens/s")

    main_model.mem_manager.free_all()
    main_model.req_manager.free_all()


def tppart_model_infer(args, model_kvargs, batch_sizes, input_len, output_len, ans_queue):
    args = get_env_start_args()
    import triton.profiler as proton
    import torch
    from lightllm.distributed import dist_group_manager
    from lightllm.utils.dist_utils import set_current_device_id

    import torch.distributed as dist

    enable_decode_overlap = args.enable_decode_microbatch_overlap
    group_size = 1
    if enable_decode_overlap or args.enable_prefill_microbatch_overlap:
        group_size = 2
    init_distributed_env(model_kvargs)
    dist_group_manager.create_groups(group_size=group_size)
    model_cfg, _ = PretrainedConfig.get_config_dict(model_kvargs["weight_dir"])
    dist.barrier()

    torch.cuda.empty_cache()

    main_model, _ = get_model(model_cfg, model_kvargs)
    draft_models = init_mtp_model(args, model_kvargs, main_model)
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]

    for batch_size in batch_sizes:
        # warm up
        run_forward_once(args, input_len, output_len, batch_size, main_model, draft_models, warmup=True)
        torch.cuda.synchronize()
        run_forward_once(args, input_len, output_len, batch_size, main_model, draft_models, warmup=False)
        dist.barrier()

    ans_queue.put(True)

    try:
        ans_queue.close()
        ans_queue.join_thread()
    except Exception:
        pass
    os._exit(0)
