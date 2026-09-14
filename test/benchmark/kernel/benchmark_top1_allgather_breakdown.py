"""Detailed CUDA Graph timing for local top-1 followed by two-rank all-gather."""

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


DEFAULT_LOCAL_VOCABS = [1, 32, 64, 128, 512, 4096, 16384, 65536, 262144, 524288, 1048576, 2097152]
DEFAULT_BATCHES = [1, 8, 32, 64, 128]


def percentile(ordered, fraction):
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def measure_graph(fn, work_bytes, samples, control_group, collective):
    unroll = max(1, min(256, (8 * 2 ** 20) // max(1, work_bytes)))
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=control_group)
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if collective:
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
    rank_times = [None, None]
    dist.all_gather_object(rank_times, local_times, group=control_group)
    times = [max(pair) for pair in zip(*rank_times)]
    ordered = sorted(times)
    return {
        "median_us": statistics.median(times),
        "mean_us": statistics.mean(times),
        "std_us": statistics.stdev(times),
        "min_us": min(times),
        "p10_us": percentile(ordered, 0.1),
        "p25_us": percentile(ordered, 0.25),
        "p75_us": percentile(ordered, 0.75),
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
    if dist.get_world_size() != 2:
        raise ValueError("This benchmark requires exactly two ranks")
    control = dist.new_group(backend="gloo")
    output_dir = Path(args.output_dir)
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "world_size": 2,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "batches": args.batches,
        "local_vocabs": args.local_vocabs,
        "global_vocabs": [2 * vocab for vocab in args.local_vocabs],
        "dtypes": args.dtypes,
        "k": 1,
        "samples": args.samples,
        "layout": "contiguous logits [batch, local_vocab]",
        "topk": "torch.topk(k=1, dim=-1, largest=True, sorted=False)",
        "candidate_communication": "one values all_gather_into_tensor plus one int64 IDs all_gather_into_tensor",
        "timing": "CUDA Events inside Graph; each sample is max(rank 0, rank 1)",
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
        print(f"Starting {len(args.batches) * len(args.local_vocabs) * len(args.dtypes) * 7} measurements", flush=True)

    warm_input = torch.ones(2 ** 20, device="cuda")
    warm_output = torch.empty(2 ** 21, device="cuda")
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
        for local_vocab in args.local_vocabs:
            with torch.cuda.stream(stream):
                torch.manual_seed(args.seed + rank * 1000003 + local_vocab)
                logits_buffer = torch.randn(max(args.batches), local_vocab, dtype=dtype, device="cuda")
                for batch in args.batches:
                    logits = logits_buffer[:batch]
                    values = torch.empty(batch, 1, dtype=dtype, device="cuda")
                    local_ids = torch.empty(batch, 1, dtype=torch.int64, device="cuda")
                    global_ids = torch.empty_like(local_ids)
                    gathered_values = torch.empty(2 * batch, 1, dtype=dtype, device="cuda")
                    gathered_ids = torch.empty(2 * batch, 1, dtype=torch.int64, device="cuda")

                    def topk():
                        torch.topk(logits, 1, dim=-1, largest=True, sorted=False, out=(values, local_ids))

                    def id_offset():
                        torch.add(local_ids, rank * local_vocab, out=global_ids)

                    def gather_values():
                        dist.all_gather_into_tensor(gathered_values, values)

                    def gather_ids():
                        dist.all_gather_into_tensor(gathered_ids, global_ids)

                    def topk_and_id():
                        topk()
                        id_offset()

                    def candidate_all_gather():
                        gather_values()
                        gather_ids()

                    def total():
                        topk_and_id()
                        candidate_all_gather()

                    topk_and_id()
                    candidate_all_gather()
                    dense_bytes = logits.numel() * logits.element_size()
                    value_bytes = values.numel() * values.element_size()
                    id_bytes = global_ids.numel() * global_ids.element_size()
                    operations = [
                        ("topk", topk, dense_bytes, False),
                        ("global_id_offset", id_offset, id_bytes, False),
                        ("topk_and_id", topk_and_id, dense_bytes, False),
                        ("gather_values", gather_values, value_bytes, True),
                        ("gather_ids", gather_ids, id_bytes, True),
                        ("candidate_all_gather", candidate_all_gather, value_bytes + id_bytes, True),
                        ("total", total, dense_bytes, True),
                    ]
                    for operation, fn, work_bytes, collective in operations:
                        row = {
                            "batch": batch,
                            "local_vocab": local_vocab,
                            "global_vocab": 2 * local_vocab,
                            "dtype": dtype_name,
                            "operation": operation,
                            "dense_rank_input_bytes": dense_bytes,
                            "value_payload_bytes": value_bytes,
                            "id_payload_bytes": id_bytes,
                            **measure_graph(fn, work_bytes, args.samples, control, collective),
                        }
                        if rank == 0:
                            if writer is None:
                                writer = csv.DictWriter(result_file, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                        completed += 1

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
        metadata["correctness"] = "Candidate values and global token ID ranges validated"
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    dist.barrier(group=control)
    dist.destroy_process_group(control)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-vocabs", type=int, nargs="+", default=DEFAULT_LOCAL_VOCABS)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--dtypes", choices=["float32", "bfloat16"], nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--seed", type=int, default=1534)
    parser.add_argument("--output-dir", default="artifacts/top1_allgather_2gpu_breakdown")
    run(parser.parse_args())
