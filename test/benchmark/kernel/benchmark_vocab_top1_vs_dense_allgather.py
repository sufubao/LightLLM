"""Compare the real vocabulary top-1 candidate path with dense all-gather.

Run with two or four ranks. Each rank owns an equal contiguous vocabulary shard.
The candidate operation is the repository's ``vocab_parallel_candidates`` with
``top_k=1``; it includes fused local top-1, one packed collective and unpacking.
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

from lightllm.common.basemodel.triton_kernel.vocab_parallel_sampling import vocab_parallel_candidates


DEFAULT_GLOBAL_VOCABS = [64, 128, 256, 1024, 8192, 32768, 131072, 524288, 1048576, 2097152, 4194304]
DEFAULT_BATCHES = [1, 8, 32, 64, 128]


def percentile(ordered, fraction):
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def measure_graph(fn, work_bytes, samples, control_group):
    unroll = max(1, min(128, (8 * 2 ** 20) // max(1, work_bytes)))
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=control_group)
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(4):
            fn()
        start.record()
        for _ in range(unroll):
            fn()
        end.record()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    local_times = []
    for _ in range(samples):
        graph.replay()
        end.synchronize()
        local_times.append(start.elapsed_time(end) * 1000 / unroll)
    graph.reset()
    rank_times = [None] * dist.get_world_size(control_group)
    dist.all_gather_object(rank_times, local_times, group=control_group)
    times = [max(sample) for sample in zip(*rank_times)]
    ordered = sorted(times)
    return {
        "median_us": statistics.median(times),
        "mean_us": statistics.mean(times),
        "std_us": statistics.stdev(times),
        "min_us": min(times),
        "p10_us": percentile(ordered, 0.1),
        "p90_us": percentile(ordered, 0.9),
        "p99_us": percentile(ordered, 0.99),
        "max_us": max(times),
        "graph_unroll": unroll,
        "samples_us": json.dumps(times),
        "rank_samples_us": json.dumps(rank_times),
    }


@torch.inference_mode()
def run(args):
    rank = int(os.environ["RANK"])
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", device_id=torch.device("cuda", device))
    world_size = dist.get_world_size()
    if world_size not in (2, 4):
        raise ValueError("This benchmark supports two or four ranks")
    if any(vocab % world_size for vocab in args.global_vocabs):
        raise ValueError("Every global vocabulary size must be divisible by world size")
    control = dist.new_group(backend="gloo")
    output_dir = Path(args.output_dir or f"artifacts/vocab_top1_vs_dense_{world_size}gpu_cuda_graph")
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "world_size": world_size,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "batches": args.batches,
        "global_vocabs": args.global_vocabs,
        "local_vocabs": [vocab // world_size for vocab in args.global_vocabs],
        "dtypes": args.dtypes,
        "samples": args.samples,
        "layout": "contiguous native projection layout [local_vocab, batch]",
        "operations": {
            "dense_all_gather_list": "current dense path: dist.all_gather into views of [global_vocab, batch]",
            "dense_all_gather_into_tensor": "dist.all_gather_into_tensor into [global_vocab, batch]",
            "candidate_top1": (
                "repository vocab_parallel_candidates(top_k=1), including local top1, packed gather, unpack"
            ),
        },
        "timing": "CUDA Events inside Graph; each replay sample uses the slowest rank",
        "conditions": (
            "allocation, graph setup, validation and host coordination excluded; no cache flush or clock lock"
        ),
    }
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.limit",
                "--format=csv",
            ],
            text=True,
        )
        metadata["topology"] = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        count = len(args.batches) * len(args.global_vocabs) * len(args.dtypes) * 3
        print(f"Starting {count} measurements on {world_size} GPUs", flush=True)

    warm_input = torch.ones(2 ** 20, device="cuda")
    warm_output = torch.empty(world_size * 2 ** 20, device="cuda")
    for _ in range(100):
        dist.all_gather_into_tensor(warm_output, warm_input)
    torch.cuda.synchronize()
    del warm_input, warm_output

    result_file = (output_dir / "timings.csv").open("w", newline="") if rank == 0 else None
    writer = None
    completed = 0
    started = time.monotonic()
    stream = torch.cuda.Stream()
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for global_vocab in args.global_vocabs:
            local_vocab = global_vocab // world_size
            with torch.cuda.stream(stream):
                torch.manual_seed(args.seed + rank * 1000003 + global_vocab)
                for batch in args.batches:
                    logits = torch.randn(local_vocab, batch, dtype=dtype, device="cuda")
                    dense_output = torch.empty(global_vocab, batch, dtype=dtype, device="cuda")
                    dense_chunks = list(dense_output.chunk(world_size, dim=0))
                    candidate_holder = [None, None]

                    def dense_list():
                        dist.all_gather(dense_chunks, logits)

                    def dense_into_tensor():
                        dist.all_gather_into_tensor(dense_output, logits)

                    def candidate_top1():
                        candidate_holder[0], candidate_holder[1] = vocab_parallel_candidates(
                            local_logits=logits,
                            vocab_start=rank * local_vocab,
                            vocab_size=global_vocab,
                            top_k=1,
                            group=dist.group.WORLD,
                            world_size=world_size,
                        )

                    operations = [
                        ("dense_all_gather_list", dense_list),
                        ("dense_all_gather_into_tensor", dense_into_tensor),
                        ("candidate_top1", candidate_top1),
                    ]
                    if batch % 2:
                        operations.reverse()
                    for operation, fn in operations:
                        row = {
                            "world_size": world_size,
                            "batch": batch,
                            "local_vocab": local_vocab,
                            "global_vocab": global_vocab,
                            "dtype": dtype_name,
                            "operation": operation,
                            "dense_rank_input_bytes": logits.numel() * logits.element_size(),
                            "candidate_rank_payload_bytes": batch
                            * 2
                            * torch.tensor([], dtype=torch.int64).element_size(),
                            **measure_graph(fn, logits.numel() * logits.element_size(), args.samples, control),
                        }
                        if rank == 0:
                            if writer is None:
                                writer = csv.DictWriter(result_file, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                        completed += 1

                    candidate_top1()
                    values, ids = candidate_holder
                    reference_values, reference_ids = torch.max(logits.float(), dim=0)
                    reference_ids += rank * local_vocab
                    gathered_reference_values = torch.empty(world_size * batch, dtype=torch.float32, device="cuda")
                    gathered_reference_ids = torch.empty(world_size * batch, dtype=torch.int64, device="cuda")
                    dist.all_gather_into_tensor(gathered_reference_values, reference_values)
                    dist.all_gather_into_tensor(gathered_reference_ids, reference_ids)
                    torch.testing.assert_close(
                        values, gathered_reference_values.view(world_size, batch).transpose(0, 1)
                    )
                    torch.testing.assert_close(ids, gathered_reference_ids.view(world_size, batch).transpose(0, 1))
                torch.cuda.synchronize()
            if rank == 0:
                result_file.flush()
                print(
                    f"{dtype_name} global_vocab={global_vocab}: {completed} rows, "
                    f"{time.monotonic() - started:.1f}s",
                    flush=True,
                )
    if rank == 0:
        result_file.close()
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["rows"] = completed
        metadata["correctness"] = "Candidate values and IDs checked against torch.max for every shape"
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    dist.barrier(group=control)
    dist.destroy_process_group(control)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-vocabs", type=int, nargs="+", default=DEFAULT_GLOBAL_VOCABS)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--dtypes", choices=["float32", "bfloat16"], nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--seed", type=int, default=1534)
    parser.add_argument("--output-dir")
    run(parser.parse_args())
