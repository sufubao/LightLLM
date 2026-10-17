"""Aggregate matched trial pairs without hiding neutral or negative cases.

Bootstrap independent workload seeds, not individual requests. Repeated seeds
across hardware crossover phases form one cluster. Preserve every raw pair.
"""

import argparse
import collections
import json
import math
from pathlib import Path
import random
import statistics


def confidence(values, seed=0):
    center = statistics.mean(values)
    if len(values) < 2:
        return {"mean": center, "ci95": None}
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return {"mean": center, "ci95": [samples[249], samples[9749]]}


def summarize(phases):
    groups = collections.defaultdict(list)
    for phase in phases:
        progress = json.loads((phase / "progress.json").read_text())
        protocol = json.loads((phase / "protocol.json").read_text())
        trials = json.loads((phase / "summary.json").read_text())
        if progress.get("status") != "complete" or len(trials) != len(protocol["cases"]) * protocol["repeats"]:
            raise ValueError(f"incomplete phase cannot enter final report: {phase}")
        candidate = json.loads((phase / "candidate-service.json").read_text())
        control = json.loads((phase / "control-service.json").read_text())
        for trial in trials:
            if trial["mismatches"]:
                raise ValueError(f"mismatched trial cannot enter matched report: {phase} {trial['case']}")
            if trial["candidate"]["unexpected_end_states"] or trial["control"]["unexpected_end_states"]:
                raise ValueError(f"invalid end states in {phase}")
            config = candidate["configuration"]
            key = (
                config["model_dir"],
                candidate["source_revision"],
                control["source_revision"],
                config["cache_tokens"],
                config["retention"],
                config["fraction"],
                trial["case"],
            )
            groups[key].append({"phase": phase.name, "trial": trial})
    result = []
    for key, pairs in sorted(groups.items()):
        per_seed = collections.defaultdict(list)
        for pair in pairs:
            trial = pair["trial"]
            candidate, control = trial["candidate"], trial["control"]
            metrics = {
                "log_wall_speedup": math.log(control["wall_seconds"] / candidate["wall_seconds"]),
                "log_mean_request_speedup": math.log(
                    control["request_seconds"]["mean"] / candidate["request_seconds"]["mean"]
                ),
                "cache_hit_gain_pp": 100 * (candidate["cache_hit_ratio"] - control["cache_hit_ratio"]),
                "prefill_reduction_percent": 100 * (1 - candidate["prefilled_tokens"] / control["prefilled_tokens"]),
            }
            if "server_ttft_ms" in candidate and "server_ttft_ms" in control:
                metrics["log_mean_ttft_speedup"] = math.log(
                    control["server_ttft_ms"]["mean"] / candidate["server_ttft_ms"]["mean"]
                )
            per_seed[candidate["configuration"]["seed"]].append(metrics)
        clustered = {
            name: [statistics.mean(m[name] for m in rows) for rows in per_seed.values()]
            for name in next(iter(per_seed.values()))[0]
        }
        summary = {
            "model": key[0],
            "candidate_revision": key[1],
            "control_revision": key[2],
            "cache_tokens": key[3],
            "retention_seconds": key[4],
            "fraction": key[5],
            "case": key[6],
            "pairs": len(pairs),
            "independent_seeds": len(per_seed),
            "metrics": {},
            "raw_pairs": pairs,
        }
        for name, values in clustered.items():
            estimate = confidence(values)
            if name.startswith("log_"):
                name = name[4:]
                estimate = {
                    "geomean": math.exp(estimate["mean"]),
                    "ci95": ([math.exp(v) for v in estimate["ci95"]] if estimate["ci95"] is not None else None),
                }
            summary["metrics"][name] = estimate
        result.append(summary)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phases", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "method": "Paired ratios; 95% descriptive bootstrap intervals clustered by workload seed; 10000 resamples.",
        "limitations": "Few seeds give limited uncertainty estimates. Request counts are not independent repetitions.",
        "groups": summarize(args.phases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "| Model | Case | TTL | Pairs / seeds | Wall speedup | Cache gain (pp) | Prefill reduction |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in result["groups"]:
        metric = group["metrics"]
        speed = metric["wall_speedup"]
        interval = "" if speed["ci95"] is None else f" [{speed['ci95'][0]:.3f}, {speed['ci95'][1]:.3f}]"
        lines.append(
            f"| {Path(group['model']).name} | {group['case']} | {group['retention_seconds']} | "
            f"{group['pairs']} / {group['independent_seeds']} | {speed['geomean']:.3f}x{interval} | "
            f"{metric['cache_hit_gain_pp']['mean']:+.2f} | {metric['prefill_reduction_percent']['mean']:+.2f}% |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
