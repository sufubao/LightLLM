"""Measure FP32 softmax and topk using unrolled CUDA Graph replay.

Example:
    python test/benchmark/kernel/benchmark_softmax_topk.py \
        --output-dir artifacts/softmax_topk_cuda_graph

Use --no-sort to benchmark topk with sorted=False.

Outputs and inputs are preallocated, and the same buffers are reused during
replay. Timings exclude allocation, input generation and host/device copies.
"""

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch


DEFAULT_VOCAB_SIZES = [1, 64, 128, 1024, 8192, 32768, 131072, 524288, 1048576, 2097152, 4194304]


def gpu_snapshot(device):
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(device),
            "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,"
            "temperature.gpu,power.draw,power.limit,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def percentile(values, fraction):
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def measure_graph(fn, samples, target_graph_ms):
    """Amortize replay submission cost over an unrolled sequence of operations."""
    fn()
    fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        fn()
    end.record()
    end.synchronize()
    estimate_ms = max(start.elapsed_time(end) / 5, 0.0001)
    unroll = max(1, min(512, math.ceil(target_graph_ms / estimate_ms)))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    times_us = []
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000 / unroll)
    graph.reset()
    return {
        "median_us": statistics.median(times_us),
        "min_us": min(times_us),
        "p10_us": percentile(times_us, 0.1),
        "p90_us": percentile(times_us, 0.9),
        "graph_unroll": unroll,
        "samples_us": json.dumps(times_us),
    }


@torch.inference_mode()
def run(args):
    default_output = "artifacts/softmax_topk_cuda_graph" + ("_unsorted" if args.no_sort else "")
    output_dir = Path(args.output_dir or default_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    batches = list(range(args.batch_min, args.batch_max + 1, args.batch_step))
    metadata = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": args.device,
        "gpu": gpu_snapshot(args.device),
        "dtype": "float32",
        "layout": "contiguous [batch, vocab]",
        "batches": batches,
        "vocab_sizes": args.vocab_sizes,
        "ks": args.ks,
        "samples": args.samples,
        "target_graph_ms": args.target_graph_ms,
        "topk_largest": True,
        "topk_sorted": not args.no_sort,
        "seed": args.seed,
        "method": "CUDA Event timings of unrolled graph replay; reused buffers; no L2 flush or clock lock",
        "operations": {
            "softmax": "torch.softmax(logits, dim=-1, out=probs)",
            "topk": f"torch.topk(precomputed_probs, k, dim=-1, largest=True, sorted={not args.no_sort}, "
            "out=(values, ids))",
            "softmax_topk": "softmax followed by topk on its output in the same graph",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    warm_x = torch.randn(128, 32768, device="cuda")
    warm_y = torch.empty_like(warm_x)
    warm_until = time.monotonic() + 1
    while time.monotonic() < warm_until:
        for _ in range(50):
            torch.softmax(warm_x, dim=-1, out=warm_y)
        torch.cuda.synchronize()
    del warm_x, warm_y

    started = time.monotonic()
    fieldnames = [
        "batch",
        "vocab",
        "k",
        "operation",
        "input_mib",
        "median_us",
        "min_us",
        "p10_us",
        "p90_us",
        "graph_unroll",
        "samples_us",
    ]
    completed = 0
    stream = torch.cuda.Stream()
    with (output_dir / "timings.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for vocab in args.vocab_sizes:
            with torch.cuda.stream(stream):
                all_logits = torch.randn(max(batches), vocab, device="cuda", dtype=torch.float32)
                all_probs = torch.empty_like(all_logits)
                shuffled_batches = batches.copy()
                random.Random(args.seed + vocab).shuffle(shuffled_batches)
                valid_ks = sorted(set(k for k in args.ks if k <= vocab))
                for batch in shuffled_batches:
                    logits = all_logits[:batch]
                    probs = all_probs[:batch]

                    def softmax(logits=logits, probs=probs):
                        torch.softmax(logits, dim=-1, out=probs)

                    def record(operation, k, fn, input_mib=logits.numel() * logits.element_size() / 2 ** 20):
                        row = {
                            "batch": batch,
                            "vocab": vocab,
                            "k": k,
                            "operation": operation,
                            "input_mib": input_mib,
                        }
                        row.update(measure_graph(fn, args.samples, args.target_graph_ms))
                        writer.writerow(row)

                    record("softmax", 0, softmax)
                    for k in valid_ks:
                        values = torch.empty(batch, k, device="cuda", dtype=torch.float32)
                        ids = torch.empty(batch, k, device="cuda", dtype=torch.int64)

                        def topk(probs=probs, k=k, values=values, ids=ids):
                            torch.topk(probs, k, dim=-1, largest=True, sorted=not args.no_sort, out=(values, ids))

                        def combined():
                            softmax()
                            topk()

                        record("topk", k, topk)
                        record("softmax_topk", k, combined)
                        del values, ids
                    csv_file.flush()
                    completed += 1
                    if completed % 16 == 0:
                        elapsed = time.monotonic() - started
                        print(
                            f"shapes={completed}/{len(batches) * len(args.vocab_sizes)} "
                            f"vocab={vocab} batch={batch} elapsed={elapsed:.1f}s",
                            flush=True,
                        )
                softmax = topk = combined = record = None
                del logits, probs, all_logits, all_probs
            torch.cuda.synchronize()
            snapshot = {"vocab_completed": vocab, "gpu": gpu_snapshot(args.device)}
            with (output_dir / "gpu_snapshots.jsonl").open("a") as snapshot_file:
                snapshot_file.write(json.dumps(snapshot) + "\n")
            torch.cuda.empty_cache()
    metadata["elapsed_seconds"] = time.monotonic() - started
    metadata["gpu_final"] = gpu_snapshot(args.device)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Finished in {metadata['elapsed_seconds']:.1f}s: {output_dir / 'timings.csv'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-min", type=int, default=1)
    parser.add_argument("--batch-max", type=int, default=128)
    parser.add_argument("--batch-step", type=int, default=1)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=DEFAULT_VOCAB_SIZES)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 64, 128])
    parser.add_argument("--no-sort", action="store_true", help="Use torch.topk(sorted=False)")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-graph-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1534)
    parser.add_argument("--output-dir", help="Defaults to artifacts/softmax_topk_cuda_graph[_unsorted]")
    run(parser.parse_args())
