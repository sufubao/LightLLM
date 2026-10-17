"""Memory lifetime experiment. Always invoke through exp -m 'purpose' python ... .

Strong mode restores only the pre-fix ownership edge; model kernels and inputs
are identical. Individual invocations use fresh processes; --suite reuses
weights and clears GC between phases. No HTTP service is started.
"""
import argparse
import atexit
import gc
import hashlib
import json
import math
from importlib.metadata import version
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import weakref

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch

MIB = 1024**2
GIB = 1024**3


class StrongRef:
    """Restore the original strong attention -> InferState ownership edge."""

    def __get__(self, instance, owner=None):
        return None if instance is None else instance.__dict__.get("_strong_infer_state")

    def __set__(self, instance, value):
        instance.__dict__["_strong_infer_state"] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", action="store_true", help="Reuse one loaded model for six isolated ownership/GC phases"
    )
    parser.add_argument("--ownership", choices=["strong", "weak"], required=True)
    parser.add_argument("--workload", choices=["controlled", "qwen3", "model"], required=True)
    parser.add_argument("--gc-mode", choices=["default", "disabled", "periodic"], default="default")
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--gc-every", type=int, default=8)
    parser.add_argument("--temp-mib", type=int, default=256)
    parser.add_argument("--base-mib", type=int, default=4096)
    parser.add_argument("--budget-mib", type=int, default=8192)
    parser.add_argument(
        "--extra-budget-mib",
        type=int,
        default=0,
        help="For model: allocator limit = warmup max reserved + this amount; 0 = no limit",
    )
    parser.add_argument(
        "--model-limit-mib", type=int, default=0, help="Absolute process allocator budget for real forward"
    )
    parser.add_argument("--prompt-logprobs", action="store_true")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--model-dir", default="/mtc/models/Qwen3-4B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from lightllm.common.basemodel.attention.base_att import BasePrefillAttState, BaseDecodeAttState
    from lightllm.common.basemodel.attention.triton.fp import TritonPrefillAttState
    from lightllm.common.basemodel.infer_struct import InferStateInfo

    weak_descriptors = (BasePrefillAttState.__dict__["infer_state"], BaseDecodeAttState.__dict__["infer_state"])
    if args.suite and args.workload == "controlled":
        raise ValueError("--suite requires a model workload")
    if args.ownership == "strong":
        BasePrefillAttState.infer_state = StrongRef()
        BaseDecodeAttState.infer_state = StrongRef()

    # Verify that the only intended A/B variable actually controls lifetime.
    owner = InferStateInfo()
    owner.prefill_att_state = TritonPrefillAttState(infer_state=owner)
    ref = weakref.ref(owner)
    gc.disable()
    del owner
    assert (ref() is not None) == (args.ownership == "strong")
    gc.collect()
    gc.enable()
    torch.cuda.init()
    torch.cuda.set_device(0)
    metadata = {
        "args": vars(args),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "triton": version("triton"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gc_threshold": gc.get_threshold(),
        "source_manifest": json.loads((ROOT / "experiment_source.json").read_text())
        if (ROOT / "experiment_source.json").exists()
        else None,
        "base_att_sha256": hashlib.sha256(
            (ROOT / "lightllm/common/basemodel/attention/base_att.py").read_bytes()
        ).hexdigest(),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()[0],
        "service": "No HTTP service: controlled ownership harness or real LightLLM model.forward",
    }
    records, events = [], []
    t0 = time.perf_counter()
    step = 0

    def sample(phase):
        torch.cuda.synchronize()
        records.append(
            dict(
                step=step,
                seconds=time.perf_counter() - t0,
                phase=phase,
                allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
                peak=torch.cuda.max_memory_allocated(),
                gc_count=gc.get_count(),
            )
        )

    def gc_event(phase, info):
        events.append(
            dict(
                step=step,
                seconds=time.perf_counter() - t0,
                phase=phase,
                generation=info["generation"],
                allocated=torch.cuda.memory_allocated(),
                collected=info.get("collected", 0),
            )
        )

    if args.workload == "controlled":
        budget = args.budget_mib * MIB
        torch.cuda.set_per_process_memory_fraction(budget / metadata["gpu_total_bytes"])
        persistent = torch.empty(args.base_mib * MIB, dtype=torch.uint8, device="cuda")
        metadata["base_allocation_bytes"] = persistent.numel() * persistent.element_size()

        def forward():
            owner = InferStateInfo()
            owner.prefill_att_state = TritonPrefillAttState(infer_state=owner)
            owner.input_ids = torch.empty(args.temp_mib * MIB, dtype=torch.uint8, device="cuda")
            sample("in_forward")
            return None

    else:
        sys.path.insert(0, str(ROOT / "test/benchmark/static_inference"))
        import static_benchmark as sb
        from lightllm.distributed import dist_group_manager
        from transformers import PretrainedConfig

        model_parser = sb.make_argument_parser()
        sb.add_static_benchmark_args(model_parser)
        raw = model_parser.parse_args(
            [
                "--model_dir",
                args.model_dir,
                "--tp",
                "1",
                "--benchmark",
                "prefill",
                "--batch_size",
                "1",
                "--input_len",
                str(args.tokens),
                "--output_len",
                "1",
                "--chunked_prefill_sizes",
                "0",
                "--batch_max_tokens",
                str(args.tokens),
                "--max_total_token_num",
                str(args.tokens + 256),
                "--running_max_req_size",
                "4",
                "--disable_cudagraph",
                "--llm_prefill_att_backend",
                "triton",
                "triton",
                "--llm_decode_att_backend",
                "triton",
                "triton",
                "--linear_att_cache_size",
                "16",
                "--nccl_port",
                "29683",
            ]
        )
        cases = sb.build_cases(raw)
        config = sb.normalize_args(raw, cases)
        config.enable_prompt_logprobs = args.prompt_logprobs
        sb.set_env_start_args(config)
        from lightllm.utils.envs_utils import set_unique_server_name, get_unique_server_name

        set_unique_server_name(config)
        service_prefix = get_unique_server_name()
        metadata["service_namespace"] = service_prefix

        def cleanup_shm():
            for directory, pattern in [("/dev/shm", f"{service_prefix}_*"), ("/tmp", f"{service_prefix}_*.lock")]:
                for path in Path(directory).glob(pattern):
                    path.unlink(missing_ok=True)

        atexit.register(cleanup_shm)
        kvargs = sb.build_model_kvargs(config, 0)
        kvargs["return_all_prompt_logics"] = args.prompt_logprobs
        sb.init_distributed_env(kvargs)
        dist_group_manager.create_groups(group_size=1)
        cfg, _ = PretrainedConfig.get_config_dict(config.model_dir)
        model, _ = sb.get_model(cfg, kvargs)
        executor = sb.StaticBenchmarkExecutor(config, model, [], sb.TokenSource(config))
        req_idx = executor._alloc_req_indexes(1)
        token_rows = executor.token_source.batch(1, args.tokens)
        request_stub = sb.SimpleNamespace(req_idx=int(req_idx[0]))
        init_hybrid_state = getattr(model.req_manager, "init_hybrid_attention_state", None)
        metadata["hybrid_state_reset_each_request"] = init_hybrid_state is not None

        def forward():
            if init_hybrid_state is not None:
                init_hybrid_state(request_stub)
            model_input = executor._make_prefill_input(token_rows, req_idx, 0)
            output = model.forward(model_input)
            # Read one deterministic result without retaining logits on the host.
            token = int(output.logits.argmax(-1).item())
            if output.prompt_logics is not None:
                metadata["prompt_logits_shape"] = list(output.prompt_logics.shape)
                metadata["prompt_logits_bytes"] = output.prompt_logics.numel() * output.prompt_logics.element_size()
            return token

        metadata["model_config"] = cfg
        metadata["model_args"] = vars(config)
        metadata["model_config_sha256"] = hashlib.sha256(
            (Path(args.model_dir) / "config.json").read_bytes()
        ).hexdigest()
        budget = args.model_limit_mib * MIB if args.model_limit_mib else None
        if budget:
            torch.cuda.set_per_process_memory_fraction(budget / metadata["gpu_total_bytes"])

    base_metadata = json.loads(json.dumps(metadata))
    suite_output_dir = Path(args.output)
    suite_budget_mib = 0
    phases = [(args.ownership, args.prompt_logprobs, args.model_limit_mib)]
    if args.suite:
        phases = [
            ("weak", False, 0),
            ("strong", False, 0),
            ("weak", True, 0),
            ("strong", True, 0),
            ("weak", True, -1),
            ("strong", True, -1),
        ]
    for ownership, prompt_logprobs, limit_mib in phases:
        if args.suite:
            torch.cuda.set_per_process_memory_fraction(1.0)
            gc.collect()
            torch.cuda.empty_cache()
            args.ownership = ownership
            args.prompt_logprobs = prompt_logprobs
            args.model_limit_mib = suite_budget_mib if limit_mib == -1 else limit_mib
            suffix = ("logprobs_" if prompt_logprobs else "") + ("capped_" if args.model_limit_mib else "")
            args.output = str(suite_output_dir / f"model_{suffix}{ownership}_default.json")
            model.return_all_prompt_logics = prompt_logprobs
            config.enable_prompt_logprobs = prompt_logprobs
            for cls, descriptor in zip((BasePrefillAttState, BaseDecodeAttState), weak_descriptors):
                cls.infer_state = StrongRef() if ownership == "strong" else descriptor
            budget = args.model_limit_mib * MIB if args.model_limit_mib else None
            if budget:
                torch.cuda.set_per_process_memory_fraction(budget / base_metadata["gpu_total_bytes"])
        metadata = json.loads(json.dumps(base_metadata))
        metadata["args"] = vars(args).copy()
        metadata["process_model_reused_between_phases"] = args.suite
        if args.workload != "controlled":
            metadata["model_args"]["enable_prompt_logprobs"] = args.prompt_logprobs
        print(
            f"BEGIN PHASE: {args.ownership}, prompt_logprobs={args.prompt_logprobs}, budget={args.model_limit_mib} MiB",
            flush=True,
        )
        # Startup check and warmup always pass under the same ownership mode.
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        startup_results = []
        for _ in range(2):
            gc.collect()
            startup_results.append(forward())
        torch.cuda.synchronize()
        metadata["startup_passed"] = True
        metadata["startup_results"] = startup_results
        metadata["startup_peak_bytes"] = torch.cuda.max_memory_allocated()
        metadata["startup_reserved_bytes"] = torch.cuda.memory_reserved()
        gc.collect()
        if args.workload != "controlled" and args.extra_budget_mib:
            budget = metadata["startup_reserved_bytes"] + args.extra_budget_mib * MIB
            torch.cuda.set_per_process_memory_fraction(budget / metadata["gpu_total_bytes"])
        metadata["allocator_budget_bytes"] = budget
        metadata["baseline_allocated_bytes"] = torch.cuda.memory_allocated()
        records.clear()
        if args.gc_mode in ("disabled", "periodic"):
            gc.disable()
        else:
            gc.enable()
        # Start with clean generation counters for both variants.
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        events.clear()
        step = 0
        t0 = time.perf_counter()
        gc.callbacks.append(gc_event)
        sample("baseline")
        completed, oom_step = 0, None
        output_tokens = []
        try:
            for step in range(1, args.iterations + 1):
                torch.cuda.reset_peak_memory_stats()
                try:
                    result = forward()
                except torch.OutOfMemoryError as exc:
                    metadata["oom_message"] = str(exc)
                    oom_step = step
                    sample("oom")
                    break
                completed += 1
                if result is not None:
                    output_tokens.append(result)
                sample("after_forward")
                if args.gc_mode == "periodic" and step % args.gc_every == 0:
                    gc.collect()
                    sample("periodic_gc")
                if step % 25 == 0:
                    allocated_mib = torch.cuda.memory_allocated() / MIB
                    print(
                        f"{args.ownership}/{args.gc_mode}: step={step}, allocated={allocated_mib:.1f} MiB",
                        flush=True,
                    )
        finally:
            sample("before_final_gc")
            gc.collect()
            sample("after_final_gc")
            gc.callbacks.remove(gc_event)
            gc.enable()
        metadata.update(
            completed=completed,
            oom_step=oom_step,
            peak_allocated_bytes=max(x["peak"] for x in records),
            output_tokens_unique=sorted(set(output_tokens)),
            final_allocated_bytes=records[-1]["allocated"],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(dict(metadata=metadata, records=records, gc_events=events), indent=2))
        print(
            json.dumps(
                {
                    k: metadata[k]
                    for k in [
                        "args",
                        "startup_passed",
                        "completed",
                        "oom_step",
                        "baseline_allocated_bytes",
                        "peak_allocated_bytes",
                        "final_allocated_bytes",
                        "allocator_budget_bytes",
                        "output_tokens_unique",
                    ]
                },
                indent=2,
            ),
            flush=True,
        )
        if args.suite and ownership == "weak" and prompt_logprobs and not args.model_limit_mib:
            suite_budget_mib = math.ceil((metadata["peak_allocated_bytes"] + GIB) / GIB) * 1024
        gc.collect()
        torch.cuda.empty_cache()
    if args.workload != "controlled":
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
