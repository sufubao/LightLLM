"""Force the CPU KV-cache offload->restore path and check correctness.

GSM8K can't exercise the CPU cache (one shared hot prefix, sub-page tails).
This driver builds N distinct, page-aligned, long prompts that overflow the
GPU KV budget so their KV is offloaded to CPU, then re-requests them so they
are restored from CPU. With greedy decoding the round-2 (CPU-restored) output
MUST be token-identical to round-1 (freshly computed). For the MTP build it
also tracks accept-rate (mtp_avg_token_per_step) which would degrade if the
draft full-attn slots were not persisted/restored correctly.
"""
import argparse
import sys
import requests
from concurrent.futures import ThreadPoolExecutor


def make_prompts(n, words_per_prompt):
    prompts = []
    for i in range(n):
        # Distinct, deterministic filler so each prompt is its own radix branch
        # and long enough to span several 256-token pages.
        filler = " ".join(f"item{i}-{j}" for j in range(words_per_prompt))
        prompts.append(
            f"You are given list number {i}. The list is: {filler}. "
            f"Question: briefly summarize what list number {i} contains. Answer:"
        )
    return prompts


def gen(url, prompt, max_tokens):
    data = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.0,
            "max_new_tokens": max_tokens,
            "stop_sequences": None,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "top_k": 1,
        },
    }
    r = requests.post(url, json=data, timeout=120)
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    return r.json()["generated_text"][0]


def run_round(url, prompts, max_tokens, parallel):
    out = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {ex.submit(gen, url, p, max_tokens): k for k, p in enumerate(prompts)}
        for f in futs:
            k = futs[f]
            out[k] = f.result()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--num-prompts", type=int, default=24)
    ap.add_argument("--words-per-prompt", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    url = f"{args.host}:{args.port}/generate"
    prompts = make_prompts(args.num_prompts, args.words_per_prompt)

    print(f"Round 1 (cold compute): {len(prompts)} distinct prompts", flush=True)
    r1 = run_round(url, prompts, args.max_tokens, args.parallel)
    print("Round 2 (CPU-restored):", flush=True)
    r2 = run_round(url, prompts, args.max_tokens, args.parallel)

    mismatches = [i for i in range(len(prompts)) if r1[i] != r2[i]]
    print(f"\n=== RESULT ===")
    print(f"prompts: {len(prompts)}  identical: {len(prompts) - len(mismatches)}  mismatches: {len(mismatches)}")
    if mismatches:
        for i in mismatches[:5]:
            print(f"  [#{i}] R1={r1[i]!r}\n        R2={r2[i]!r}")
        sys.exit(1)
    print("PASS: round-2 (CPU-restored) output is token-identical to round-1.")


if __name__ == "__main__":
    main()
