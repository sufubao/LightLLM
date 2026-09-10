"""Native HTTP correctness and timing checks for exact Agent checkpoints.

Run through the experiment ledger, for example::

    exp -m "Agent checkpoint comparison" python test/benchmark/agent_checkpoint_cache.py \
      --baseline-url http://127.0.0.1:17880 --candidate-url http://127.0.0.1:17881 \
      --model-dir /models/Qwen3.5-0.8B --output /dev/shm/checkpoint-results

Each request records the exact input/output token IDs and parameters. Streaming
timings are client-observed; MTP can deliver several tokens in one burst. Older
native streams omit cache metadata: their hit length remains null, and a
separately labeled nonstream probe records cache hits without inventing TTFT.
This is a functional workload with short outputs, not a saturation benchmark.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid


CACHE_FIELDS = ("prompt_cache_len", "mtp_accepted_token_num", "mtp_verify_token_num", "mtp_verify_step_num")
DEFAULT_LENGTHS = (255, 256, 257, 8191, 8192, 8193, 12000)


def fetch_json(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError) as error:
        return {"unavailable": str(error)}


def git_metadata():
    result = {}
    for key, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "branch", "--show-current"]),
        ("status", ["git", "status", "--short"]),
    ):
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result[key] = completed.stdout.strip() if completed.returncode == 0 else None
    return result


def token_logprob(token):
    value = token.get("logprob")
    if value is None:
        values = token.get("logprobs", {})
        entry = values.get(str(token.get("id")), values.get(token.get("id"), {}))
        value = entry.get("logprob") if isinstance(entry, dict) else entry
    return float(value) if value is not None and math.isfinite(float(value)) else None


def request_once(url, case, parameters, phase, stream, timeout):
    started = time.perf_counter()
    record = {
        "case": case["name"],
        "kind": case["kind"],
        "phase": phase,
        "stream": stream,
        "url": url,
        "started_unix": time.time(),
        "input_len": len(case["tokens"]),
        "input_token_ids": case["tokens"],
        "input_sha256": hashlib.sha256(json.dumps(case["tokens"], separators=(",", ":")).encode()).hexdigest(),
        "parameters": parameters,
        "events": [],
        "output_token_ids": [],
        "output_logprobs": [],
        "cache_hit_len": None,
        "cache_hit_source": None,
        "ttft_ms": None,
        "tpot_ms": None,
        "error": None,
    }
    request = urllib.request.Request(
        url.rstrip("/") + ("/generate_stream" if stream else "/generate"),
        data=json.dumps({"inputs": case["tokens"], "parameters": parameters}).encode(),
        headers={"Content-Type": "application/json"},
    )
    tokens = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            record["http_status"] = response.status
            if stream:
                for line in response:
                    received = (time.perf_counter() - started) * 1000
                    line = line.decode("utf-8").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        continue
                    event = json.loads(body)
                    if "error" in event:
                        raise RuntimeError(f"SSE error: {event['error']}")
                    token = event.get("token")
                    if not isinstance(token, dict) or token.get("id") is None:
                        continue
                    tokens.append(token)
                    record["events"].append({"received_ms": received, **event})
                    if record["ttft_ms"] is None:
                        record["ttft_ms"] = received
                    if event.get("finished"):
                        record["finish_reason"] = event.get("finish_reason")
                if tokens and "finish_reason" not in record:
                    raise RuntimeError("stream ended without a finished event")
            else:
                body = json.load(response)
                record["response"] = body
                tokens = body.get("tokens", [])
                if tokens and isinstance(tokens[0], list):
                    raise RuntimeError("expected one sequence, received multiple outputs")
                record["finish_reason"] = body.get("finish_reason")
        if not tokens:
            raise RuntimeError("response contained no token IDs")
    except (OSError, ValueError, RuntimeError) as error:
        record["error"] = str(error)
        if isinstance(error, urllib.error.HTTPError):
            record["http_status"] = error.code
            record["error_body"] = error.read().decode("utf-8", errors="replace")[:4000]
    record["latency_ms"] = (time.perf_counter() - started) * 1000
    record["output_token_ids"] = [int(token["id"]) for token in tokens]
    record["output_logprobs"] = [token_logprob(token) for token in tokens]
    record["output_len"] = len(tokens)
    record["prompt_tokens"] = tokens[0].get("prompt_tokens") if tokens else None
    for field in CACHE_FIELDS:
        values = [token[field] for token in tokens if field in token]
        record[field] = max(values) if values else None
    record["cache_hit_len"] = record["prompt_cache_len"]
    if record["cache_hit_len"] is not None:
        record["cache_hit_source"] = "stream.token.prompt_cache_len" if stream else "generate.tokens.prompt_cache_len"
    if stream and len(tokens) > 1:
        record["tpot_ms"] = (record["events"][-1]["received_ms"] - record["ttft_ms"]) / (len(tokens) - 1)
    return record


def compare(reference, observed, tolerance):
    expected = reference["output_token_ids"]
    actual = observed["output_token_ids"]
    same_ids = expected == actual
    pairs = list(zip(reference["output_logprobs"], observed["output_logprobs"]))
    # Chosen-token probabilities are comparable only for the same sequence.
    # After a greedy divergence, later positions also have different prefixes.
    errors = [abs(a - b) for a, b in pairs if a is not None and b is not None] if same_ids else []
    complete_logprobs = bool(expected) and same_ids and len(errors) == len(expected)
    return {
        "case": observed["case"],
        "reference": reference["record_id"],
        "observed": observed["record_id"],
        "token_ids_equal": same_ids,
        "first_token_difference": next(
            (i for i, (a, b) in enumerate(zip(expected, actual)) if a != b),
            min(len(expected), len(actual)) if not same_ids else None,
        ),
        "max_logprob_abs_error": max(errors) if errors else None,
        "logprobs_complete": complete_logprobs,
        "logprob_atol": tolerance,
        "passed": (
            reference["error"] is None
            and observed["error"] is None
            and same_ids
            and complete_logprobs
            and max(errors) <= tolerance
        ),
    }


class Workload:
    def __init__(self, args):
        from transformers import AutoTokenizer

        self.args = args
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "requests.jsonl").exists():
            raise ValueError("output already contains requests.jsonl; choose a new experiment directory")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
        self.run_id = args.run_id or uuid.uuid4().hex[:12]
        self.endpoints = {
            name: url for name, url in (("baseline", args.baseline_url), ("candidate", args.candidate_url)) if url
        }
        self.server_info = {
            name: fetch_json(url.rstrip("/") + "/get_server_info", args.timeout) for name, url in self.endpoints.items()
        }
        self.records = []
        self.comparisons = []
        self.checks = []
        self.lock = threading.Lock()
        self.ordinal = 0
        self.manifest = {
            "run_id": self.run_id,
            "created_unix": time.time(),
            "script_version": 1,
            "git": git_metadata(),
            "arguments": vars(args),
            "server_info": self.server_info,
            "cases": [],
            "notes": [
                "TTFT and TPOT are observed by this HTTP client; MTP tokens may arrive in bursts.",
                "Missing streaming cache metadata is null; detail probes are different requests.",
                "Cold correctness controls use disable_prompt_cache=true; a rejected control is a failure.",
                "HTTP MTP counters prove activity, not the per-verify stopping row; kernel tests are also required.",
            ],
        }
        self.write_json("manifest.json", self.manifest)

    def write_json(self, name, value):
        (self.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def append(self, name, value):
        with (self.output / name).open("a") as file:
            file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def make_prompt(self, length, tag):
        prefix = self.tokenizer.encode(
            f"Checkpoint run {self.run_id}, case {tag}. Read this technical context carefully.\n",
            add_special_tokens=False,
        )
        filler = self.tokenizer.encode(
            "A request reads a shared prefix and then processes a new suffix. "
            "The cache stores attention keys and values, while recurrent state summarizes earlier tokens. ",
            add_special_tokens=False,
        )
        tail = self.tokenizer.encode("\nContinue the numbered list: 1, 2, 3, 4, 5,", add_special_tokens=False)
        remaining = length - len(prefix) - len(tail)
        if remaining < 1:
            raise ValueError(f"length {length} is too short for the reproducible prompt scaffold")
        return prefix + (filler * math.ceil(remaining / len(filler)))[:remaining] + tail

    def case(self, name, kind, tokens, parameters=None, **metadata):
        case = {"name": name, "kind": kind, "tokens": list(tokens), "parameters": parameters or {}, **metadata}
        self.manifest["cases"].append(case)
        self.write_json("manifest.json", self.manifest)
        return case

    def run_request(self, server, case, phase, *, stream=True, cold=False, concurrency_round=None):
        parameters = {
            "do_sample": False,
            "seed": self.args.seed,
            "max_new_tokens": self.args.max_new_tokens,
            "ignore_eos": True,
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "return_details": True,
            **case["parameters"],
            "disable_prompt_cache": cold,
        }
        record = request_once(self.endpoints[server], case, parameters, phase, stream, self.args.timeout)
        record["server"] = server
        record["concurrency_round"] = concurrency_round
        with self.lock:
            self.ordinal += 1
            record["record_id"] = f"{self.ordinal:05d}-{server}-{case['name']}-{phase}"
            self.records.append(record)
            self.append("requests.jsonl", record)
        print(
            f"{server:9s} {case['name']:28s} {phase:18s} input={record['input_len']} "
            f"output={record['output_len']} cache={record['cache_hit_len']} "
            f"ttft_ms={record['ttft_ms']} error={record['error']}",
            flush=True,
        )
        self.add_check(record, "input_length", record["prompt_tokens"] == len(case["tokens"]))
        if cold and record["cache_hit_len"] is not None:
            self.add_check(record, "cold_has_no_cache_hit", record["cache_hit_len"] == 0)
        return record

    def add_check(self, record, name, passed, detail=None):
        check = {"record_id": record["record_id"], "check": name, "passed": bool(passed), "detail": detail}
        with self.lock:
            self.checks.append(check)
            self.append("checks.jsonl", check)

    def compare(self, reference, observed):
        comparison = compare(reference, observed, self.args.logprob_atol)
        self.comparisons.append(comparison)
        self.append("comparisons.jsonl", comparison)
        if not comparison["passed"]:
            print("COMPARISON FAILED " + json.dumps(comparison), flush=True)

    def evaluate(self, case):
        references = {}
        for server in self.endpoints:
            # Legacy servers may still insert at request teardown when reads
            # are disabled. Probe a new branch before the cold control can
            # populate that branch and hide its initial matching behavior.
            seed = self.run_request(server, case, "cache_seed", stream=False)
            cold = self.run_request(server, case, "cold_reference", cold=True)
            references[server] = cold
            self.compare(cold, seed)
            if "max_initial_hit" in case and seed["cache_hit_len"] is not None:
                self.add_check(
                    seed, "branch_hit_does_not_cross_divergence", seed["cache_hit_len"] <= case["max_initial_hit"]
                )
            if self.args.require_exact_hits and server == "candidate" and "min_initial_hit" in case:
                self.add_check(
                    seed,
                    "output_prefix_reused",
                    seed["cache_hit_len"] is not None and seed["cache_hit_len"] >= case["min_initial_hit"],
                )
            time.sleep(self.args.settle_ms / 1000)
            for repeat in range(self.args.repeats):
                warm = self.run_request(server, case, f"warm_stream_{repeat}")
                self.compare(cold, warm)
                self.compare(seed, warm)
                if self.args.require_exact_hits and server == "candidate":
                    self.add_check(warm, "exact_full_hit", warm["cache_hit_len"] == len(case["tokens"]))
                if warm["cache_hit_len"] is None and not self.args.no_detail_probe:
                    detail = self.run_request(server, case, f"cache_detail_probe_{repeat}", stream=False)
                    self.compare(cold, detail)
            if case["kind"] == "eos":
                self.add_check(
                    seed,
                    "natural_eos_reached",
                    seed.get("finish_reason") == "stop"
                    and bool(seed["output_token_ids"])
                    and seed["output_token_ids"][-1] in case["eos_token_ids"],
                )
            if "stop_sequences" in case["parameters"]:
                self.add_check(seed, "token_stop_reached", seed.get("finish_reason") == "stop")
                if self.args.require_mtp_activity and server == "candidate":
                    self.add_check(
                        seed, "mtp_acceptance_observed_on_stop_request", (seed["mtp_accepted_token_num"] or 0) > 0
                    )
        if len(references) == 2:
            self.compare(references["baseline"], references["candidate"])
        return references

    def run(self):
        suites = set(self.args.suites.split(","))
        if "boundaries" in suites:
            for length in self.args.lengths:
                self.evaluate(self.case(f"boundary_{length}", "boundary", self.make_prompt(length, str(length))))
        anchor = None
        if suites.intersection({"agent", "branch", "stops"}):
            anchor = self.case("agent_anchor", "anchor", self.make_prompt(self.args.agent_input_len, "anchor"))
            refs = self.evaluate(anchor)
            reference = refs.get("baseline", next(iter(refs.values())))
            if reference["error"]:
                raise RuntimeError("anchor cold request failed; cannot construct trustworthy continuation cases")
            outputs = reference["output_token_ids"]
            if "agent" in suites:
                tool = self.tokenizer.encode(
                    '\nTool result: {"status":"ok","value":42}. Continue the answer.\n', add_special_tokens=False
                )
                self.evaluate(
                    self.case(
                        "agent_tool_suffix",
                        "agent",
                        anchor["tokens"] + outputs + tool,
                        source_case=anchor["name"],
                        source_output_token_ids=outputs,
                        min_initial_hit=len(anchor["tokens"]) + len(outputs) - 1,
                    )
                )
            if "branch" in suites:
                split = max(1, len(anchor["tokens"]) - 17)
                replacement = self.tokenizer.encode("Different branch. ", add_special_tokens=False)
                old = anchor["tokens"][split]
                different = next(token for token in replacement if token != old)
                branch = anchor["tokens"][:split] + [different] + anchor["tokens"][split + 1 :]
                self.evaluate(self.case("branch_before_tail", "branch", branch, max_initial_hit=split))
            if "stops" in suites:
                stop = outputs[: min(3, len(outputs))]
                stop_refs = self.evaluate(
                    self.case(
                        "accepted_prefix_token_stop",
                        "token_stop",
                        anchor["tokens"],
                        {"stop_sequences": [stop]},
                        stop_output_position=len(stop),
                    )
                )
                stopped = stop_refs.get("baseline", next(iter(stop_refs.values())))["output_token_ids"]
                stop_tool = self.tokenizer.encode("\nTool result: 7. Continue.\n", add_special_tokens=False)
                self.evaluate(
                    self.case(
                        "token_stop_tool_suffix",
                        "stopped_agent",
                        anchor["tokens"] + stopped + stop_tool,
                        source_case="accepted_prefix_token_stop",
                        source_output_token_ids=stopped,
                        max_initial_hit=len(anchor["tokens"]) + len(stopped),
                    )
                )
                eos_ids = next(iter(self.server_info.values())).get("eos_id") or []
                if isinstance(eos_ids, int):
                    eos_ids = [eos_ids]
                if eos_ids:
                    # allowed_token_ids is ignored unless the server uses outlines.
                    # A completed assistant answer exercises real EOS sampling.
                    eos_prompt = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": "Reply with exactly OK."},
                            {"role": "assistant", "content": "OK"},
                        ],
                        tokenize=False,
                        continue_final_message=True,
                        enable_thinking=False,
                    )
                    self.evaluate(
                        self.case(
                            "natural_eos",
                            "eos",
                            self.tokenizer.encode(eos_prompt, add_special_tokens=False),
                            {"ignore_eos": False, "max_new_tokens": max(16, self.args.max_new_tokens)},
                            eos_token_ids=eos_ids,
                        )
                    )
                else:
                    self.manifest["notes"].append("EOS case unavailable: server did not expose an eos_id.")
        if "concurrency" in suites:
            cases = [
                self.case(
                    f"concurrent_{i}",
                    "concurrency",
                    self.make_prompt(self.args.concurrency_input_len, f"parallel-{i}"),
                )
                for i in range(self.args.concurrency)
            ]
            for server in self.endpoints:
                refs = {}
                seeds = {}
                for case in cases:
                    refs[case["name"]] = self.run_request(server, case, "cold_reference", cold=True)
                    seeds[case["name"]] = self.run_request(server, case, "cache_seed", stream=False)
                    self.compare(refs[case["name"]], seeds[case["name"]])
                time.sleep(self.args.settle_ms / 1000)
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.concurrency) as executor:
                    for round_index in range(self.args.repeats):
                        pending = [
                            (
                                case,
                                executor.submit(
                                    self.run_request,
                                    server,
                                    case,
                                    "concurrent_warm",
                                    concurrency_round=round_index,
                                ),
                            )
                            for case in cases
                        ]
                        for case, future in pending:
                            warm = future.result()
                            self.compare(refs[case["name"]], warm)
                            self.compare(seeds[case["name"]], warm)
                            if self.args.require_exact_hits and server == "candidate":
                                self.add_check(
                                    warm, "concurrent_exact_full_hit", warm["cache_hit_len"] == len(case["tokens"])
                                )
        return self.finish()

    def finish(self, fatal_error=None):
        summary = {
            "run_id": self.run_id,
            "fatal_error": fatal_error,
            "servers": {},
            "comparison_count": len(self.comparisons),
        }
        for server in self.endpoints:
            records = [record for record in self.records if record["server"] == server]
            warm = [
                record
                for record in records
                if record["phase"].startswith("warm_stream") or record["phase"] == "concurrent_warm"
            ]
            timings = {}
            for key in ("ttft_ms", "tpot_ms", "latency_ms"):
                values = [record[key] for record in warm if record[key] is not None and record["error"] is None]
                timings[key] = {
                    "n": len(values),
                    "mean": statistics.mean(values) if values else None,
                    "median": statistics.median(values) if values else None,
                }
            info = self.server_info[server]
            accepted = [
                record["mtp_accepted_token_num"] for record in records if record["mtp_accepted_token_num"] is not None
            ]
            summary["servers"][server] = {
                "requests": len(records),
                "errors": sum(record["error"] is not None for record in records),
                "warm_timings": timings,
                "warm_case_timings": [
                    {
                        key: record[key]
                        for key in (
                            "case",
                            "phase",
                            "concurrency_round",
                            "input_len",
                            "output_len",
                            "ttft_ms",
                            "tpot_ms",
                            "latency_ms",
                            "cache_hit_len",
                        )
                    }
                    for record in warm
                ],
                "stream_cache_metadata_observed": sum(record["cache_hit_len"] is not None for record in warm),
                "mtp_enabled": bool(info.get("mtp_step", 0)),
                "mtp_accepted_tokens_observed": max(accepted) if accepted else None,
                "mtp_interior_stop_coverage": "requires per-verify trace; HTTP counters alone are insufficient"
                if info.get("mtp_step", 0)
                else "not applicable: MTP disabled",
            }
        failed = [comparison for comparison in self.comparisons if not comparison["passed"]]
        failed_checks = [check for check in self.checks if not check["passed"]]
        summary["failed_comparisons"] = failed
        summary["failed_checks"] = failed_checks
        summary["passed"] = not (
            fatal_error or failed or failed_checks or any(record["error"] for record in self.records)
        )
        self.write_json("manifest.json", self.manifest)
        self.write_json("summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if summary["passed"] else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-url")
    parser.add_argument("--candidate-url")
    parser.add_argument(
        "--baseline-revision", help="Revision of the deployed baseline; independent of the client checkout"
    )
    parser.add_argument("--candidate-revision", help="Revision/diff label of the deployed candidate")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", help="Stable case identity; default is unique to avoid previous-run cache hits")
    parser.add_argument("--seed", type=int, default=1558)
    parser.add_argument(
        "--lengths", type=lambda value: [int(item) for item in value.split(",")], default=list(DEFAULT_LENGTHS)
    )
    parser.add_argument("--suites", default="boundaries,agent,branch,stops,concurrency")
    parser.add_argument("--agent-input-len", type=int, default=12000)
    parser.add_argument("--concurrency-input-len", type=int, default=257)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--settle-ms", type=float, default=200)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--logprob-atol", type=float, default=0.03)
    parser.add_argument("--no-detail-probe", action="store_true")
    parser.add_argument(
        "--require-exact-hits",
        action="store_true",
        help="Assert candidate full and Agent continuation hits; requires streaming cache metadata",
    )
    parser.add_argument(
        "--require-mtp-activity",
        action="store_true",
        help="Require accepted MTP tokens on the candidate token-stop request; does not prove the exact stop row",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if not args.baseline_url and not args.candidate_url:
        parser.error("at least one of --baseline-url and --candidate-url is required")
    if set(args.suites.split(",")) - {"boundaries", "agent", "branch", "stops", "concurrency"}:
        parser.error("unknown suite")
    if (
        min(
            [
                args.max_new_tokens,
                args.repeats,
                args.concurrency,
                args.agent_input_len,
                args.concurrency_input_len,
                *args.lengths,
            ]
        )
        <= 0
    ):
        parser.error("lengths, repeats, and concurrency must be positive")
    if args.settle_ms < 0 or args.logprob_atol < 0 or args.timeout <= 0:
        parser.error("invalid timing or tolerance parameter")
    return args


def main():
    workload = Workload(parse_args())
    try:
        return workload.run()
    except Exception as error:
        return workload.finish(fatal_error=f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
