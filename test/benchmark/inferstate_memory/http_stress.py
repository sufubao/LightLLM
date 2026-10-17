"""Real HTTP request A/B coordinator. Invoke with exp -m 'purpose' python ... ."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

import aiohttp
import psutil


def set_phase(directory, phase):
    temp = directory / "phase.tmp"
    temp.write_text(phase)
    temp.replace(directory / "phase")


async def requests_load(url, directory, name, count, concurrency, input_len, output_len, prompt_logprobs=False):
    set_phase(directory, name)
    results = []
    semaphore = asyncio.Semaphore(concurrency)
    failure = asyncio.Event()
    begin = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as session:

        async def one(index):
            async with semaphore:
                if failure.is_set():
                    return
                payload = {
                    "inputs": [1000 + (i * 17 + index * 31) % 10000 for i in range(input_len)],
                    "parameters": {"do_sample": False, "ignore_eos": True, "max_new_tokens": output_len},
                }
                if prompt_logprobs:
                    payload["parameters"]["prompt_logprobs"] = 0
                start = time.time()
                row = dict(
                    index=index, scenario=name, start=start, input_len=input_len, requested_output_len=output_len
                )
                try:
                    async with session.post(url + "/generate", json=payload) as response:
                        body = await response.text()
                        row.update(status=response.status, end=time.time())
                        if response.status == 200:
                            data = json.loads(body)
                            row.update(
                                output_tokens=data.get("count_output_tokens"), prompt_tokens=data.get("prompt_tokens")
                            )
                            if row["output_tokens"] != output_len:
                                row["error"] = "unexpected output length"
                                failure.set()
                        else:
                            row["error"] = body[:1000]
                            failure.set()
                except Exception as exc:
                    row.update(status=0, end=time.time(), error=repr(exc))
                    failure.set()
                results.append(row)
                if len(results) % 8 == 0 or row.get("error"):
                    print(f'{directory.name}/{name}: {len(results)}/{count}, status={row["status"]}', flush=True)

        await asyncio.gather(*(one(i) for i in range(count)))
    summary = dict(
        scenario=name,
        count=count,
        concurrency=concurrency,
        input_len=input_len,
        output_len=output_len,
        started=begin,
        ended=time.time(),
        attempted=len(results),
        successes=sum(r.get("status") == 200 and not r.get("error") for r in results),
        errors=sum(bool(r.get("error")) for r in results),
    )
    (directory / f"{name}_requests.json").write_text(json.dumps(dict(summary=summary, requests=results), indent=2))
    set_phase(directory, "idle_after_" + name)
    await asyncio.sleep(2)
    return not failure.is_set()


async def run_variant(args, ownership, graph):
    directory = Path(args.output).resolve() / f"{ownership}_graph_{graph}"
    directory.mkdir(parents=True, exist_ok=True)
    set_phase(directory, "startup")
    command = [
        sys.executable,
        "-m",
        "lightllm.server.api_server",
        "--model_dir",
        args.model_dir,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--nccl_port",
        str(args.port + 1),
        "--tp",
        "1",
        "--data_type",
        "bfloat16",
        "--max_total_token_num",
        "65536",
        "--batch_max_tokens",
        "4096",
        "--max_req_total_len",
        "8192",
        "--running_max_req_size",
        "256",
        "--chunked_prefill_size",
        "4096",
        "--disable_vision",
        "--disable_audio",
        "--output_constraint_mode",
        "none",
        "--llm_prefill_att_backend",
        "triton",
        "triton",
        "--llm_decode_att_backend",
        "triton",
        "triton",
        "--linear_att_cache_size",
        "32",
        "--enable_prompt_logprobs",
        "--graph_max_batch_size",
        "8",
        "--graph_split_batch_size",
        "8",
        "--graph_max_len_in_batch",
        "8192",
    ]
    if graph == "off":
        command.append("--disable_cudagraph")
    environment = os.environ.copy()
    command_dir = Path.cwd() / "experiment_bin"
    command_dir.mkdir(exist_ok=True)
    hypercorn = command_dir / "hypercorn"
    hypercorn.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' -m hypercorn "$@"\n')
    hypercorn.chmod(0o700)
    environment["PATH"] = (
        str(command_dir) + os.pathsep + str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    environment.update(INFERSTATE_HTTP_PROBE_DIR=str(directory), INFERSTATE_HTTP_OWNERSHIP=ownership)
    metadata = dict(
        command=command,
        ownership=ownership,
        graph=graph,
        started=time.time(),
        model_dir=args.model_dir,
        cuda_visible_devices=environment.get("CUDA_VISIBLE_DEVICES"),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256((Path(__file__).parent / "service_probe.py").read_bytes()).hexdigest(),
        source=json.loads(Path("experiment_source.json").read_text())
        if Path("experiment_source.json").exists()
        else None,
    )
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log = (directory / "server.log").open("w")
    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    root = psutil.Process(process.pid)
    try:
        ready = False
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
            for attempt in range(300):
                if process.poll() is not None:
                    break
                try:
                    async with session.get(f"http://127.0.0.1:{args.port}/readiness") as response:
                        if response.status == 200:
                            ready = True
                            break
                except Exception:
                    pass
                if attempt % 15 == 0:
                    print(f"{directory.name}: waiting for HTTP readiness, {attempt * 2}s", flush=True)
                await asyncio.sleep(2)
        metadata["ready"] = ready
        metadata["ready_time"] = time.time()
        if not ready:
            print(f"{directory.name}: startup failed; see {directory}/server.log", flush=True)
            return
        url = f"http://127.0.0.1:{args.port}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url + "/get_server_info") as response:
                (directory / "server_info.json").write_text(await response.text())
        await requests_load(url, directory, "warmup", 2, 1, 32, 8)
        success = await requests_load(url, directory, "decode_heavy", 32, 8, 32, 256)
        if success:
            await requests_load(url, directory, "prefill_heavy", 64, 1, 4096, 1, True)
        await asyncio.sleep(2)
    finally:
        metadata["finished"] = time.time()
        metadata["server_returncode_before_stop"] = process.poll()
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        children = []
        try:
            children = root.children(recursive=True)
        except psutil.Error:
            pass
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        for child in children:
            try:
                if child.is_running():
                    child.kill()
            except psutil.Error:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        log.close()
        await asyncio.sleep(3)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/mtc/models/Qwen3.5-27B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=18761)
    parser.add_argument("--ownership", choices=["weak", "strong", "both"], default="both")
    parser.add_argument("--graph", choices=["on", "off", "both"], default="both")
    args = parser.parse_args()
    for graph in ["off", "on"] if args.graph == "both" else [args.graph]:
        for ownership in ["weak", "strong"] if args.ownership == "both" else [args.ownership]:
            await run_variant(args, ownership, graph)
            args.port += 10


if __name__ == "__main__":
    asyncio.run(main())
