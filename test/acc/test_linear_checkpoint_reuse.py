"""Live hybrid-cache correctness probe. Run with exp, against a test server.

Use deterministic sampling, CPU cache, and sufficient CPU KV/state capacity
to retain the original prefix while --pressure evicts its GPU KV.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18867")
    parser.add_argument("--model", default="/mtc/models/Qwen3.5-0.8B")
    parser.add_argument("--pressure", action="store_true")
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument("--prompt-length", type=int, default=1103)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    unit = tokenizer.encode("The library contains books about science, art, and history. ", add_special_tokens=False)
    prompt = (unit * (args.prompt_length // len(unit) + 1))[: args.prompt_length]

    def generate(tokens, count=12, disable=False):
        response = requests.post(
            args.url + "/generate",
            json={
                "inputs": tokens,
                "parameters": {
                    "max_new_tokens": count,
                    "do_sample": False,
                    "ignore_eos": True,
                    "return_details": True,
                    "disable_prompt_cache": disable,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        ids = [token["id"] for token in data["tokens"]]
        cached = data["tokens"][0].get("prompt_cache_len", 0)
        print(json.dumps({"prompt": len(tokens), "cached": cached, "ids": ids, "disable": disable}), flush=True)
        time.sleep(2)  # allow first-use transfer compilation, publication, and request release
        return ids, cached

    original, _ = generate(prompt)
    repeat, repeat_hit = generate(prompt)
    assert repeat == original
    assert repeat_hit > 0
    extension = prompt + original + tokenizer.encode(" Continue describing the library.", add_special_tokens=False)
    continued, output_hit = generate(extension)
    assert output_hit >= len(prompt) + len(original) - 1, (output_hit, len(prompt), len(original))
    reference, disabled_hit = generate(extension, disable=True)
    assert disabled_hit == 0
    assert continued == reference, "cached output-state continuation differs from recomputation"
    if args.pressure:
        # With an 8192-token GPU cache these distinct prefixes evict the
        # original KV. Keep CPU capacity above 16K and state capacity >=64.
        for index in range(5):
            filler = tokenizer.encode(f"Document number {index}: ", add_special_tokens=False)
            filler = filler + (unit * 250)[:2200]
            generate(filler, count=2)
        restored, cpu_hit = generate(extension)
        assert cpu_hit >= len(prompt) + len(original) - 1
        assert restored == reference, "CPU checkpoint continuation differs from recomputation"
    if args.concurrent:
        branches = [extension + tokenizer.encode(f" Section {index}:", add_special_tokens=False) for index in range(4)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(generate, branches))
        for branch, (result, _) in zip(branches, results):
            recomputed, hit = generate(branch, disable=True)
            assert hit == 0
            assert result == recomputed, "concurrent extension changed a shared checkpoint"
    print("checkpoint reuse correctness passed", flush=True)


if __name__ == "__main__":
    main()
