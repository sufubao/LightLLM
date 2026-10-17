"""Run sequential, alternating, exp-recorded paired trials on existing services."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time


def manifest(name):
    candidates = []
    for path in (Path.home() / "experiments/runs").glob("*/service-manifest.json"):
        data = json.loads(path.read_text())
        if data["configuration"]["name"] == name:
            candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        raise ValueError(f"no exp service manifest for {name}")
    return max(candidates, key=lambda item: item[0])[1:]


def timings(name, output):
    text = subprocess.check_output(["docker", "logs", name], stderr=subprocess.STDOUT, text=True)
    output.write_text(text)
    pattern = re.compile(r"X-Request-Id:(\S+) .*?first_token_cost:([0-9.eE+-]+)ms .*?gpu_prompt_cache_len:(\d+)")
    return {m[1]: {"ttft_ms": float(m[2]), "gpu_cached_tokens": int(m[3])} for m in pattern.finditer(text)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model-dir", default="/models/Qwen3-4B")
    parser.add_argument("--rounds", default="cachewise-analysis-v2.rounds.jsonl")
    parser.add_argument("--exp", default="/mtc/sufubao/shared_home/sufubao/.local/bin/exp")
    parser.add_argument(
        "--allow-different-source", action="store_true", help="Explicitly permit the original-base audit"
    )
    parser.add_argument("--no-reset", action="store_true", help="Warmup/exploration only: retain prior cache contents")
    args = parser.parse_args()
    destination = args.root / "results" / args.phase
    destination.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    services = {}
    for label, name in [("candidate", args.candidate), ("control", args.control)]:
        source, data = manifest(name)
        state = json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]
        if not state["State"]["Running"]:
            raise RuntimeError(f"service {name} is not running")
        services[label] = data
        shutil.copy2(source, destination / f"{label}-service.json")
    left, right = services["candidate"], services["control"]
    for key in ["image_id", "model_config_sha256"]:
        if left[key] != right[key]:
            raise ValueError(f"services differ in {key}")
    for key in [
        "model_dir",
        "model_name",
        "tp",
        "cache_tokens",
        "max_req_tokens",
        "batch_tokens",
        "max_requests",
        "disable_cudagraph",
    ]:
        if left["configuration"].get(key) != right["configuration"].get(key):
            raise ValueError(f"services differ in {key}")
    for flag in [
        "--tool_call_parser",
        "--linear_att_page_block_num",
        "--llm_prefill_att_backend",
        "--llm_decode_att_backend",
    ]:
        values = [
            service["command"][service["command"].index(flag) + 1] if flag in service["command"] else None
            for service in [left, right]
        ]
        if values[0] != values[1]:
            raise ValueError(f"services differ in {flag}")
    if left["source_revision"] != right["source_revision"] and not args.allow_different_source:
        raise ValueError("policy comparison requires identical source; use --allow-different-source for the base audit")
    harness_sha = hashlib.sha256((args.root / "replay.py").read_bytes()).hexdigest()
    shutil.copy2(args.root / "replay.py", destination / "replay.py")
    if not args.no_reset:
        shutil.copy2(args.root / "reset_cache.py", destination / "reset_cache.py")
    (destination / "protocol.json").write_text(
        json.dumps(
            {
                "cases": cases,
                "repeats": args.repeats,
                "harness_sha256": harness_sha,
                "cache_isolation": "none" if args.no_reset else "expire preferences, then independent capacity flood",
                "reset_sha256": (
                    None if args.no_reset else hashlib.sha256((destination / "reset_cache.py").read_bytes()).hexdigest()
                ),
                "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            indent=2,
        )
        + "\n"
    )
    summaries = []
    last_done = {label: time.monotonic() for label in services}
    for case in cases:
        for repeat in range(args.repeats):
            namespace = f"{args.phase}-{case['name']}-r{repeat}"
            outputs = {}
            order = ["control", "candidate"] if repeat % 2 == 0 else ["candidate", "control"]
            for label in order:
                service = services[label]
                output = destination / f"{case['name']}-r{repeat}-{label}.json"
                if not args.no_reset:
                    config = service["configuration"]
                    wait = max(0, config["retention"] + 0.25 - (time.monotonic() - last_done[label]))
                    reset = [
                        args.exp,
                        "-m",
                        f"Untimed isolated-cache prelude {namespace} {label}; "
                        f"service {service['source_revision']}; full manifest in {destination}",
                        "python3",
                        str(destination / "reset_cache.py"),
                        "--url",
                        f"http://127.0.0.1:{config['port']}",
                        "--model",
                        config["model_name"],
                        "--cache-tokens",
                        str(config["cache_tokens"]),
                        "--max-req-tokens",
                        str(config["max_req_tokens"]),
                        "--sleep-seconds",
                        str(wait),
                        "--seed",
                        str(int(hashlib.sha256(namespace.encode()).hexdigest(), 16)),
                    ]
                    print(f"Resetting cache before {namespace} {label}, lease wait {wait:.2f}s", flush=True)
                    with output.with_suffix(".reset.log").open("w") as log:
                        subprocess.run(reset, stdout=log, stderr=subprocess.STDOUT, check=True)
                command = [
                    args.exp,
                    "-m",
                    f"{namespace} {label}; service {service['source_revision']}; "
                    f"image {service['image_id']}; harness {harness_sha}; full manifests in {destination}",
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "host",
                    "--entrypoint",
                    "python",
                    "-e",
                    "HF_HOME=/tmp/hf",
                    "-v",
                    f"{args.root}:/bench",
                    "-v",
                    "/mtc/models:/models:ro",
                    service["image_id"],
                    "/bench/" + str((destination / "replay.py").relative_to(args.root)),
                    "--url",
                    f"http://127.0.0.1:{service['configuration']['port']}",
                    "--rounds",
                    f"/bench/{args.rounds}",
                    "--model-dir",
                    args.model_dir,
                    "--model",
                    service["configuration"]["model_name"],
                    "--namespace",
                    namespace,
                    "--seed",
                    str(42 + repeat),
                    "--output",
                    "/bench/" + str(output.relative_to(args.root)),
                ]
                for key, value in case["options"].items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                (destination / "progress.json").write_text(
                    json.dumps(
                        {
                            "case": case["name"],
                            "repeat": repeat,
                            "policy": label,
                            "command": command,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(f"Starting {namespace} {label}", flush=True)
                with output.with_suffix(".log").open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                last_done[label] = time.monotonic()
                if completed.returncode:
                    raise RuntimeError(f"trial failed, preserve logs: {output.with_suffix('.log')}")
                outputs[label] = json.loads(output.read_text())
            left = {(r["session"], r["turn"]): r for r in outputs["control"]["rows"]}
            right = {(r["session"], r["turn"]): r for r in outputs["candidate"]["rows"]}
            if left.keys() != right.keys():
                raise RuntimeError(f"paired requests differ: {namespace}")
            mismatches = []
            for key in left:
                for field in ["prompt_sha256", "finish_reason", "output_signature"]:
                    if left[key][field] != right[key][field]:
                        mismatches.append({"request": key, "field": field})
            summary = {"case": case["name"], "repeat": repeat, "order": order, "mismatches": mismatches}
            for label, result in outputs.items():
                summary[label] = {k: v for k, v in result.items() if k not in ["rows", "sessions"]}
            summaries.append(summary)
            (destination / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "case": case["name"],
                        "repeat": repeat,
                        "mismatches": len(mismatches),
                        **{
                            label: {
                                "wall_seconds": outputs[label]["wall_seconds"],
                                "cache_hit_ratio": outputs[label]["cache_hit_ratio"],
                                "prefilled_tokens": outputs[label]["prefilled_tokens"],
                            }
                            for label in outputs
                        },
                    }
                ),
                flush=True,
            )
            if mismatches:
                raise RuntimeError(
                    f"paired inputs/outputs differ; retain but do not interpret as matched performance: {namespace}"
                )
    for label, name in [("candidate", args.candidate), ("control", args.control)]:
        measured = timings(name, destination / f"{label}-server.log")
        for summary in summaries:
            data = json.loads((destination / f"{summary['case']}-r{summary['repeat']}-{label}.json").read_text())
            values = [measured[r["request_id"]]["ttft_ms"] for r in data["rows"]]
            if any(
                measured[r["request_id"]]["gpu_cached_tokens"] != r["usage"]["prompt_tokens_details"]["cached_tokens"]
                for r in data["rows"]
            ):
                raise RuntimeError("server/client cache metrics differ")
            summary[label]["server_ttft_ms"] = {
                "mean": statistics.mean(values),
                "p95": sorted(values)[int((len(values) - 1) * 0.95)],
            }
    (destination / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (destination / "progress.json").write_text(
        json.dumps({"status": "complete", "paired_trials": len(summaries)}) + "\n"
    )
    print(f"Completed {len(summaries)} paired trials: {destination}", flush=True)


if __name__ == "__main__":
    main()
