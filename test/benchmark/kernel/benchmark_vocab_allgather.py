"""Two-rank vocabulary all-gather benchmark with CUDA Graph internal events.

Run with: torchrun --standalone --nproc_per_node=2 \
    test/benchmark/kernel/benchmark_vocab_allgather.py

Vocab sizes passed to this script are PER RANK; global vocab is twice that.
Allocation, correctness checks, graph setup and host coordination are excluded.
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


DEFAULT_LOCAL_VOCABS = [1, 32, 64, 128, 512, 4096, 16384, 65536, 262144, 524288, 1048576, 2097152, 4194304]


def percentile(ordered, fraction):
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def measure(fn, input_bytes, samples, control_group):
    # Internal events and an untimed collective prefix absorb replay launch skew.
    unroll = max(1, min(128, (8 * 2 ** 20) // input_bytes))
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
    median = statistics.median(times)
    return {
        "median_us": median,
        "min_us": min(times),
        "max_us": max(times),
        "p10_us": percentile(ordered, 0.1),
        "p90_us": percentile(ordered, 0.9),
        "rank_payload_GBps": input_bytes / (median * 1000),
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
    output_dir = Path(args.output_dir)
    batches = args.batches or list(range(1, 129))
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "world_size": 2,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "batches": batches,
        "local_vocabs": args.local_vocabs,
        "global_vocabs": [2 * v for v in args.local_vocabs],
        "dtypes": args.dtypes,
        "apis": ["all_gather", "all_gather_into_tensor"],
        "samples": args.samples,
        "layout": "input contiguous [local_vocab, batch], output contiguous [global_vocab, batch]",
        "timing": "CUDA events inside graph; 4 untimed prefix collectives; median of per-sample rank max",
        "unroll": "clamp(floor(8 MiB / per-rank input bytes), 1, 128)",
        "conditions": "reused buffers; no cache flush or clock lock; no transpose or dtype cast",
    }
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata["gpu"] = subprocess.check_output(["nvidia-smi", "-q"], text=True)
        metadata["topology"] = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Starting {len(batches) * len(args.local_vocabs) * len(args.dtypes) * 2} measurements", flush=True)

    # Initialize NCCL and warm the GPUs with the same number of calls on both ranks.
    warm_input = torch.ones(2 ** 20, device="cuda")
    warm_output = torch.empty(2 ** 21, device="cuda")
    for _ in range(100):
        dist.all_gather_into_tensor(warm_output, warm_input)
    torch.cuda.synchronize()
    del warm_input, warm_output
    stream = torch.cuda.Stream()
    started = time.monotonic()
    file = (output_dir / "timings.csv").open("w", newline="") if rank == 0 else None
    writer = None
    completed = 0
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for vocab in args.local_vocabs:
            with torch.cuda.stream(stream):
                input_buffer = torch.full((vocab * max(batches),), rank + 1, dtype=dtype, device="cuda")
                output_buffer = torch.empty(2 * vocab * max(batches), dtype=dtype, device="cuda")
                shuffled = batches.copy()
                random.Random(1534 + vocab).shuffle(shuffled)
                for batch in shuffled:
                    n = vocab * batch
                    local_input = input_buffer[:n].view(vocab, batch)
                    output = output_buffer[: 2 * n].view(2 * vocab, batch)
                    chunks = list(output.chunk(2, dim=0))
                    apis = ["all_gather", "all_gather_into_tensor"]
                    if batch % 2:
                        apis.reverse()
                    for api in apis:

                        def fn(api=api, chunks=chunks, local_input=local_input, output=output):
                            if api == "all_gather":
                                dist.all_gather(chunks, local_input)
                            else:
                                dist.all_gather_into_tensor(output, local_input)

                        result = measure(fn, n * input_buffer.element_size(), args.samples, control)
                        # Check every shape/API outside timing; rank values detect wrong offsets or missing peers.
                        assert all(bool(torch.all(chunk == r + 1)) for r, chunk in enumerate(chunks))
                        row = {
                            "batch": batch,
                            "local_vocab": vocab,
                            "global_vocab": 2 * vocab,
                            "dtype": dtype_name,
                            "api": api,
                            "rank_input_bytes": n * input_buffer.element_size(),
                            "rank_output_bytes": 2 * n * input_buffer.element_size(),
                            **result,
                        }
                        if rank == 0:
                            if writer is None:
                                writer = csv.DictWriter(file, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                        completed += 1
                torch.cuda.synchronize()
                del input_buffer, output_buffer, local_input, output, chunks
            if rank == 0:
                file.flush()
                print(
                    f"{dtype_name} local_vocab={vocab}: {completed} rows, {time.monotonic() - started:.1f}s", flush=True
                )
    if rank == 0:
        file.close()
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["correctness"] = "Every output chunk equals its source rank's value for every shape/API"
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    dist.barrier(group=control)
    dist.destroy_process_group(control)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-vocabs", type=int, nargs="+", default=DEFAULT_LOCAL_VOCABS)
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--dtypes", choices=["float32", "bfloat16"], nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts/vocab_allgather_2gpu_cuda_graph")
    run(parser.parse_args())
