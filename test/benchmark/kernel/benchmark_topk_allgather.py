"""Benchmark local top-k followed by two-rank candidate all-gather.

Run with::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
        test/benchmark/kernel/benchmark_topk_allgather.py

Each rank owns a contiguous ``[batch, local_vocab]`` shard. The pipeline uses
``torch.topk(sorted=False)`` and then gathers candidate values and global int64
token IDs with ``all_gather_into_tensor``. Allocation, graph construction,
validation and host coordination are excluded from timings.
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist


DEFAULT_LOCAL_VOCABS = [1, 32, 64, 128, 512, 4096, 16384, 65536, 262144, 524288, 1048576, 2097152]


def percentile(ordered, fraction):
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def measure_graph(fn, work_bytes, samples, control_group, collective):
    """Time an unrolled graph and report the slower rank for every replay."""
    unroll = max(1, min(128, (8 * 2 ** 20) // max(1, work_bytes)))
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=control_group)
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if collective:
            # Absorb small CPU launch skew before the timed event pair.
            for _ in range(4):
                fn()
        start.record()
        for _ in range(unroll):
            fn()
        end.record()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    local_times = []
    for _ in range(samples):
        graph.replay()
        end.synchronize()
        local_times.append(start.elapsed_time(end) * 1000 / unroll)
    graph.reset()
    rank_times = [None, None]
    dist.all_gather_object(rank_times, local_times, group=control_group)
    times = [max(pair) for pair in zip(*rank_times)]
    ordered = sorted(times)
    return {
        "median_us": statistics.median(times),
        "min_us": min(times),
        "p10_us": percentile(ordered, 0.1),
        "p90_us": percentile(ordered, 0.9),
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
    if dist.get_world_size() != 2:
        raise ValueError("This benchmark requires exactly two ranks")
    control = dist.new_group(backend="gloo")
    batches = args.batches or list(range(1, 129))
    output_dir = Path(args.output_dir)
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "world_size": 2,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "batches": batches,
        "local_vocabs": args.local_vocabs,
        "global_vocabs": [2 * vocab for vocab in args.local_vocabs],
        "requested_ks": args.ks,
        "dtypes": args.dtypes,
        "layout": "contiguous logits [batch, local_vocab]; rank-major gathered candidates [2*batch, k]",
        "topk": "torch.topk(dim=-1, largest=True, sorted=False), followed by local-to-global ID addition",
        "all_gather": "two all_gather_into_tensor calls: candidate values then global int64 token IDs",
        "operations": ["topk", "candidate_all_gather", "topk_then_all_gather"],
        "timing": "CUDA events inside graph; median of 7 replay times after taking the slower rank",
        "conditions": "reused buffers; no cache flush or clock lock; allocation and validation excluded",
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
        cases = sum(sum(k <= 2 * vocab for k in args.ks) for vocab in args.local_vocabs)
        print(f"Starting {len(batches) * len(args.dtypes) * cases * 3} measurements", flush=True)

    warm_input = torch.ones(2 ** 20, device="cuda")
    warm_output = torch.empty(2 ** 21, device="cuda")
    for _ in range(100):
        dist.all_gather_into_tensor(warm_output, warm_input)
    torch.cuda.synchronize()
    del warm_input, warm_output

    started = time.monotonic()
    result_file = (output_dir / "timings.csv").open("w", newline="") if rank == 0 else None
    writer = None
    completed = 0
    stream = torch.cuda.Stream()
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for local_vocab in args.local_vocabs:
            with torch.cuda.stream(stream):
                torch.manual_seed(args.seed + rank * 1000003 + local_vocab)
                logits_buffer = torch.randn(max(batches), local_vocab, dtype=dtype, device="cuda")
                shuffled_batches = batches.copy()
                random.Random(args.seed + local_vocab).shuffle(shuffled_batches)
                requested_ks = [k for k in args.ks if k <= 2 * local_vocab]
                for batch in shuffled_batches:
                    logits = logits_buffer[:batch]
                    for requested_k in requested_ks:
                        k = min(requested_k, local_vocab)
                        values = torch.empty(batch, k, dtype=dtype, device="cuda")
                        local_ids = torch.empty(batch, k, dtype=torch.int64, device="cuda")
                        global_ids = torch.empty_like(local_ids)
                        gathered_values = torch.empty(2 * batch, k, dtype=dtype, device="cuda")
                        gathered_ids = torch.empty(2 * batch, k, dtype=torch.int64, device="cuda")

                        def topk():
                            torch.topk(logits, k, dim=-1, largest=True, sorted=False, out=(values, local_ids))
                            torch.add(local_ids, rank * local_vocab, out=global_ids)

                        def candidate_all_gather():
                            dist.all_gather_into_tensor(gathered_values, values)
                            dist.all_gather_into_tensor(gathered_ids, global_ids)

                        def combined():
                            topk()
                            candidate_all_gather()

                        topk()
                        candidate_all_gather()
                        operations = [
                            ("topk", topk, logits.numel() * logits.element_size(), False),
                            (
                                "candidate_all_gather",
                                candidate_all_gather,
                                values.numel() * values.element_size() + global_ids.numel() * global_ids.element_size(),
                                True,
                            ),
                            ("topk_then_all_gather", combined, logits.numel() * logits.element_size(), True),
                        ]
                        if (batch + k) % 2:
                            operations.reverse()
                        for operation, fn, work_bytes, collective in operations:
                            result = measure_graph(fn, work_bytes, args.samples, control, collective)
                            row = {
                                "batch": batch,
                                "local_vocab": local_vocab,
                                "global_vocab": 2 * local_vocab,
                                "requested_global_k": requested_k,
                                "k_per_rank": k,
                                "gathered_candidates": 2 * k,
                                "dtype": dtype_name,
                                "operation": operation,
                                "dense_rank_input_bytes": logits.numel() * logits.element_size(),
                                "candidate_rank_payload_bytes": values.numel() * values.element_size()
                                + global_ids.numel() * global_ids.element_size(),
                                **result,
                            }
                            if rank == 0:
                                if writer is None:
                                    writer = csv.DictWriter(result_file, fieldnames=list(row))
                                    writer.writeheader()
                                writer.writerow(row)
                            completed += 1

                        # Validate both values and the local-to-global token mapping outside timing.
                        for source_rank in range(2):
                            value_chunk = gathered_values[source_rank * batch : (source_rank + 1) * batch]
                            id_chunk = gathered_ids[source_rank * batch : (source_rank + 1) * batch]
                            assert bool(torch.all(id_chunk >= source_rank * local_vocab))
                            assert bool(torch.all(id_chunk < (source_rank + 1) * local_vocab))
                            if source_rank == rank:
                                torch.testing.assert_close(value_chunk, values)
                                torch.testing.assert_close(id_chunk, global_ids)
                torch.cuda.synchronize()
                del logits_buffer
            if rank == 0:
                result_file.flush()
                print(
                    f"{dtype_name} local_vocab={local_vocab}: {completed} rows, " f"{time.monotonic() - started:.1f}s",
                    flush=True,
                )
    if rank == 0:
        result_file.close()
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["rows"] = completed
        metadata["correctness"] = "Every local gathered chunk and every global token-ID range validated"
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    dist.barrier(group=control)
    dist.destroy_process_group(control)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-vocabs", type=int, nargs="+", default=DEFAULT_LOCAL_VOCABS)
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 64, 128])
    parser.add_argument("--dtypes", choices=["float32", "bfloat16"], nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1534)
    parser.add_argument("--output-dir", default="artifacts/topk_allgather_2gpu_cuda_graph")
    run(parser.parse_args())
