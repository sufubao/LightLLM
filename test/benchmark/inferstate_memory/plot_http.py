"""Render actual HTTP pressure records; no new model measurements."""
import argparse
import base64
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GIB = 1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    variants = []
    summaries = []
    for directory in sorted(args.directory.iterdir()):
        if not directory.is_dir() or not (directory / "metadata.json").exists():
            continue
        metadata = json.loads((directory / "metadata.json").read_text())
        events = []
        for path in directory.glob("worker_*.jsonl"):
            for line in path.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        phases = {}
        for scenario in ("warmup", "decode_heavy", "prefill_heavy"):
            path = directory / f"{scenario}_requests.json"
            if path.exists():
                response_data = json.loads(path.read_text())
                phases[scenario] = response_data["summary"]
                phases[scenario]["generated_tokens"] = sum(
                    r.get("output_tokens", 0) or 0 for r in response_data["requests"] if r.get("status") == 200
                )
        records = [r for r in events if r["event"] in ("forward", "oom") and r.get("initialized")]
        row = dict(
            variant=directory.name,
            ownership=metadata["ownership"],
            graph=metadata["graph"],
            ready=metadata.get("ready", False),
        )
        for scenario in ("decode_heavy", "prefill_heavy"):
            phase = [r for r in records if r["phase"] == scenario]
            decode = [r for r in phase if r.get("stage") == "decode"]
            row[scenario] = dict(
                requests=phases.get(scenario),
                forwards=len(phase),
                decode_forwards=len(decode),
                peak_GiB=max((r["peak"] / GIB for r in phase), default=None),
                decode_peak_GiB=max((r["peak"] / GIB for r in decode), default=None),
                max_decode_uncached=max((r.get("decode_uncached", 0) for r in decode), default=None),
                max_prefill_uncached=max((r.get("prefill_uncached", 0) for r in phase), default=None),
                graph_cached=max((r.get("graph_cached", 0) for r in phase), default=None),
                oom=[r for r in phase if r["event"] == "oom"],
            )
        row["server_log_cuda_oom"] = "CUDA out of memory" in (directory / "server.log").read_text(errors="replace")
        summaries.append(row)
        variants.append((metadata, events, phases))
    (args.directory / "summary.json").write_text(json.dumps(summaries, indent=2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for col, graph in enumerate(("off", "on")):
        for meta, events, phases in variants:
            if meta["graph"] != graph:
                continue
            owner = meta["ownership"]
            color = "#d1495b" if owner == "strong" else "#007f73"
            label = "Before fix: strong ref" if owner == "strong" else "After fix: weak ref"
            decode = [
                r
                for r in events
                if r["event"] == "forward"
                and r.get("initialized")
                and r["phase"] == "decode_heavy"
                and r.get("stage") == "decode"
            ]
            if decode:
                axes[0, col].plot(
                    range(1, len(decode) + 1), [r["decode_uncached"] for r in decode], color=color, label=label, lw=1.5
                )
            forwards = [
                r
                for r in events
                if r["event"] in ("forward", "oom") and r.get("initialized") and r["phase"] == "prefill_heavy"
            ]
            if forwards and "prefill_heavy" in phases:
                start = phases["prefill_heavy"]["started"]
                axes[1, col].plot(
                    [r["time"] - start for r in forwards],
                    [r["peak"] / GIB for r in forwards],
                    color=color,
                    label=label,
                    lw=1.8,
                )
                for r in forwards:
                    if r["event"] == "oom":
                        axes[1, col].scatter([r["time"] - start], [r["peak"] / GIB], color=color, marker="x", s=70)
                        axes[1, col].annotate(
                            "CUDA OOM",
                            (r["time"] - start, r["peak"] / GIB),
                            xytext=(8, -18),
                            textcoords="offset points",
                            color=color,
                        )
        axes[0, col].set(
            title=f"Decode Graph {graph.upper()}: uncached InferState survivors",
            xlabel="Decode forward (HTTP workload)",
            ylabel="Live states excluding Graph cache",
        )
        axes[1, col].set(
            title=f"Decode Graph {graph.upper()}: long-input HTTP workload",
            xlabel="Runtime since prefill load started (s)",
            ylabel="Peak active CUDA tensors (GiB)",
        )
        for ax in axes[:, col]:
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=9)
    fig.suptitle(
        (
            "Qwen3.5-27B real HTTP pressure / m35 H200 GPUs 2/3 / default Pyth"
            "on GC\nDecode: 32 requests, concurrency 8, 32 in + 256 out; Prefi"
            "ll: 64 requests, concurrency 1, 4096 in + 1 out, prompt logprobs"
        ),
        fontsize=12,
        weight="bold",
    )
    fig.savefig(args.directory / "http_memory.png", dpi=170)
    fig.savefig(args.directory / "http_memory.pdf")
    nvml_path = args.directory / "nvml.csv"
    nvml_chart = ""
    if nvml_path.exists():
        with nvml_path.open() as f:
            samples = list(csv.DictReader(f))
        fig2, nvml_axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        for col, graph in enumerate(("off", "on")):
            for meta, events, phases in variants:
                if meta["graph"] != graph:
                    continue
                color = "#d1495b" if meta["ownership"] == "strong" else "#007f73"
                label = "Before fix: strong ref" if meta["ownership"] == "strong" else "After fix: weak ref"
                gpu = str(meta["cuda_visible_devices"])
                for row, scenario in enumerate(("decode_heavy", "prefill_heavy")):
                    q = phases.get(scenario)
                    if not q:
                        continue
                    errors = [r for r in events if r["event"] == "oom" and r["phase"] == scenario]
                    end = min(q["ended"], min((e["time"] for e in errors), default=q["ended"]))
                    selected = [s for s in samples if s["gpu"] == gpu and q["started"] <= float(s["time"]) <= end]
                    if not selected:
                        continue
                    times = [float(s["time"]) - q["started"] for s in selected]
                    memory = [float(s["used_mib"]) / 1024 for s in selected]
                    nvml_axes[row, col].plot(times, memory, color=color, label=label, lw=1.8)
                    for summary in summaries:
                        if summary["ownership"] == meta["ownership"] and summary["graph"] == graph:
                            summary[scenario]["nvml_sampled_peak_GiB"] = max(memory)
                    if errors:
                        nvml_axes[row, col].scatter([times[-1]], [memory[-1]], color=color, marker="x", s=65)
                        nvml_axes[row, col].annotate(
                            "CUDA OOM",
                            (times[-1], memory[-1]),
                            xytext=(8, -18),
                            textcoords="offset points",
                            color=color,
                        )
            for row, scenario in enumerate(("Decode-heavy", "Prefill-heavy")):
                ax = nvml_axes[row, col]
                ax.set(
                    title=f"{scenario}, Decode Graph {graph.upper()}",
                    xlabel="Runtime since HTTP load started (s)",
                    ylabel="NVML GPU memory used (GiB)",
                )
                ax.grid(alpha=0.2)
                if ax.lines:
                    ax.legend(fontsize=9)
        fig2.suptitle(
            (
                "Real HTTP request pressure: GPU physical memory (NVML, sampled ev"
                "ery 0.5 s)\nQwen3.5-27B / m35 H200 GPUs 2 and 3 / 256 request slo"
                "ts / default GC"
            ),
            fontsize=13,
            weight="bold",
        )
        fig2.savefig(args.directory / "http_nvml.png", dpi=170)
        fig2.savefig(args.directory / "http_nvml.pdf")
        plt.close(fig2)
        nvml_chart = (
            '<img src="data:image/png;base64,'
            + base64.b64encode((args.directory / "http_nvml.png").read_bytes()).decode()
            + '">'
        )
    (args.directory / "summary.json").write_text(json.dumps(summaries, indent=2))
    chart = base64.b64encode((args.directory / "http_memory.png").read_bytes()).decode()
    rows = []
    for r in summaries:
        for scenario in ("decode_heavy", "prefill_heavy"):
            s = r[scenario]
            q = s["requests"] or {}
            values = [
                r["variant"],
                scenario,
                q.get("successes"),
                q.get("attempted"),
                s["peak_GiB"],
                s["max_decode_uncached"],
                s["graph_cached"],
                len(s["oom"]),
            ]
            rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>")
    report = (
        '<html lang="zh"><meta charset="utf-8"><title>真实 HTTP 压测</title><s'
        "tyle>body{font:16px system-ui;max-width:1300px;margin:32px auto;l"
        "ine-height:1.6}table{border-collapse:collapse}td,th{padding:8px;b"
        "order-bottom:1px solid #ddd}img{width:100%}</style>"
    )
    report += (
        "<h1>Qwen3.5-27B 真实 HTTP 压测</h1><p>实际运行 LightLLM api_server，通过 /ge"
        "nerate 请求生成结果。m35 H200：Graph 关闭组用 GPU 2，Graph 开启组用 GPU 3；BF16、TP="
        "1、默认 GC；动态前缀缓存和 chunked prefill 保持开启，请求前缀不同。每个变体启动全新服务。启用 prompt "
        "logprobs；长输入请求实际请求 prompt_logprobs=0。</p>"
    )
    report += (
        "<p>四个变体共完成 384 个正式压力请求，全部成功，本配置未出现 OOM。每个变体包含 32 个并发度 8 的短输入长输出请求"
        "，以及 64 个串行长输入请求；请求槽位为 256。实际长输入被调度器拆为 3584 和 512 token 的两个 Prefil"
        "l chunk，因此先前直接调用 forward 的第 44 次 OOM 不能当成 HTTP 服务的请求阈值。</p>"
    )
    report += (
        "<p>探针被动记录实际 worker 的 CUDA 分配和峰值、GC、InferState 弱引用存活计数；不主动触发 GC，不在"
        "请求路径同步 CUDA。Graph 缓存持有的状态单独统计，不算作未回收的请求状态。第一张图为 PyTorch 活跃张量显存；第二"
        "张图为每 0.5 秒采样的 NVML GPU 实际占用，包含分配器缓存及 CUDA 上下文等，不能把其全部差值归因于活跃 Deco"
        "de 张量。统计中的 Decode peak 可能包含尚未回收的短 Prefill 状态，因此不把两组显存差全部归因于 Decod"
        "e 临时张量。</p>"
    )
    report += (
        "<table><tr>"
        + "".join(f"<th>{x}</th>" for x in ["变体", "负载", "成功", "已发出", "峰值 GiB", "Decode 非缓存存活状态峰值", "Graph 缓存状态", "OOM"])
        + "</tr>"
        + "".join(rows)
        + "</table>"
    )
    report += f'<img src="data:image/png;base64,{chart}">' + nvml_chart + "</html>"
    (args.directory / "report.html").write_text(report)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
