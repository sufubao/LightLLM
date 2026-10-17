"""Controlled eviction probe, not a production-traffic throughput benchmark.

Uses real tool generation and independent length-limited requests to create KV
pressure. The return request repeats the initial prompt exactly, isolating its
residency from chat-template/history rewriting. Use smoke.py for actual tool
result continuation correctness. Run both policies with identical arguments.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import urllib.request


def post(base, endpoint, body):
    request = urllib.request.Request(
        base + endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix-words", type=int, default=4096)
    parser.add_argument("--churn-requests", type=int, default=6)
    parser.add_argument("--churn-tokens", type=int, default=8192)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--kind", choices=["tool", "normal", "length"], default="tool")
    args = parser.parse_args()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text.",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            },
        }
    ]
    rng = random.Random(args.seed)
    results = []
    for repeat in range(args.repeats):
        filler = " ".join(str(rng.randrange(1000, 9999)) for _ in range(args.prefix_words))
        instruction = (
            "Call echo with text hello. Use the tool now."
            if args.kind != "normal"
            else "Say OK. Do not call any tools."
        )
        body = {
            "model": "tool-cache-qwen4b",
            "messages": [{"role": "user", "content": filler + "\n" + instruction}],
            "tools": tools,
            "temperature": 0,
            "max_tokens": 1 if args.kind == "length" else 128,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        initial, initial_s = post(args.url, "/v1/chat/completions", body)
        expected = {"tool": "tool_calls", "normal": "stop", "length": "length"}[args.kind]
        if initial["choices"][0]["finish_reason"] != expected:
            raise RuntimeError(f"unexpected end state: {initial}")
        churn = []
        for _ in range(args.churn_requests):
            prompt = [rng.randrange(100, 140000) for _ in range(args.churn_tokens)]
            response, elapsed = post(
                args.url,
                "/v1/completions",
                {
                    "model": "tool-cache-qwen4b",
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            churn.append({"seconds": elapsed, "usage": response["usage"]})
        if args.wait_seconds:
            time.sleep(args.wait_seconds)
        returned, return_s = post(args.url, "/v1/chat/completions", body)
        if returned["choices"][0]["finish_reason"] != expected:
            raise RuntimeError(f"unexpected return end state: {returned}")
        results.append(
            {
                "repeat": repeat,
                "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                "initial_seconds": initial_s,
                "initial_usage": initial["usage"],
                "return_seconds": return_s,
                "return_usage": returned["usage"],
                "churn": churn,
            }
        )
        print(json.dumps(results[-1]), flush=True)
    result = {
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "results": results,
        "mean_return_seconds": statistics.mean(r["return_seconds"] for r in results),
        "cached_return_tokens": sum(r["return_usage"]["prompt_tokens_details"]["cached_tokens"] for r in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
