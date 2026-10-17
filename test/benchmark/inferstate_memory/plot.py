"""Render recorded experiments; no GPU execution or new measurements."""
import argparse
import base64
import csv
import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

GIB = 1024**3
COLORS = {"strong": "#d1495b", "weak": "#007f73"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    files = sorted(args.directory.glob("*.json"))
    runs = [json.loads(f.read_text()) for f in files if f.name != "summary.json"]
    runs = [r for r in runs if "metadata" in r]
    model_name = next(
        (
            Path(r["metadata"]["args"]["model_dir"]).name
            for r in runs
            if r["metadata"]["args"]["workload"] != "controlled"
        ),
        "model",
    )
    limits = [
        r["metadata"]["allocator_budget_bytes"] / GIB
        for r in runs
        if r["metadata"]["args"]["workload"] != "controlled" and r["metadata"].get("allocator_budget_bytes")
    ]
    limit_label = f"{limits[0]:g} GiB" if limits else "configured"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    for group in ("controlled", "model", "model_logprobs", "model_logprobs_uncapped"):
        workload = "controlled" if group == "controlled" else "model"
        data = [
            r
            for r in runs
            if (r["metadata"]["args"]["workload"] == "controlled") == (workload == "controlled")
            and bool(r["metadata"]["args"].get("prompt_logprobs")) == group.startswith("model_logprobs")
        ]
        if group.startswith("model_logprobs"):
            data = [
                r for r in data if bool(r["metadata"]["args"].get("model_limit_mib")) == (group == "model_logprobs")
            ]
        if not data:
            continue
        modes = [
            m for m in ("default", "disabled", "periodic") if any(r["metadata"]["args"]["gc_mode"] == m for r in data)
        ]
        fig, axes = plt.subplots(
            2, len(modes), figsize=(max(10, 7 * len(modes)), 9), squeeze=False, constrained_layout=True
        )
        for col, mode in enumerate(modes):
            ax, peak_ax = axes[:, col]
            for r in sorted(data, key=lambda r: r["metadata"]["args"]["ownership"]):
                meta = r["metadata"]
                owner = meta["args"]["ownership"]
                records = [x for x in r["records"] if x["phase"] in ("baseline", "after_forward", "periodic_gc", "oom")]
                x = [p["step"] for p in records]
                y = [p["allocated"] / GIB for p in records]
                label = "Before fix: strong ref" if owner == "strong" else "After fix: weak ref"
                ax.plot(x, y, color=COLORS[owner], lw=2, label=label)
                # The second row includes ephemeral allocations inside each forward.
                peak_ax.plot(x, [p["peak"] / GIB for p in records], color=COLORS[owner], lw=2, label=label)
                peak_ax.plot(
                    x,
                    [p["reserved"] / GIB for p in records],
                    color=COLORS[owner],
                    lw=1,
                    ls=":",
                    alpha=0.6,
                    label=("Allocator reserved" if owner == "strong" else None),
                )
                if meta["oom_step"]:
                    for target in (ax, peak_ax):
                        target.axvline(meta["oom_step"], color=COLORS[owner], alpha=0.4, ls="--")
                    peak_ax.annotate(
                        f"OOM at forward {meta['oom_step']}",
                        (x[-1], records[-1]["peak"] / GIB),
                        xytext=(8, -28),
                        textcoords="offset points",
                        color=COLORS[owner],
                        weight="bold",
                    )
                for event in r["gc_events"]:
                    if owner == "strong" and event["phase"] == "stop" and event["collected"]:
                        ax.axvline(event["step"], color="#777777", alpha=0.15, lw=0.8)
            cap = data[0]["metadata"].get("allocator_budget_bytes")
            same = [
                r["metadata"].get("allocator_budget_bytes") for r in data if r["metadata"]["args"]["gc_mode"] == mode
            ]
            if cap and len(set(same)) == 1:
                for target in (ax, peak_ax):
                    target.axhline(cap / GIB, color="#333333", ls="--", label="Process allocator limit")
            ax.set_title(
                {
                    "default": "Python GC enabled (default thresholds)",
                    "disabled": "Cyclic GC disabled",
                    "periodic": "Explicit full GC every 8 forwards",
                }[mode],
                weight="bold",
            )
            for target in (ax, peak_ax):
                target.set_xlabel("Completed / attempted forward")
                target.set_ylabel("GPU memory (GiB)")
                target.grid(alpha=0.2)
                target.xaxis.set_major_locator(MaxNLocator(integer=True))
                target.legend(loc="best", fontsize=9)
            ax.set_title(ax.get_title() + "\nActive tensors after each forward", fontsize=12)
            peak_ax.set_title("Peak active tensors inside each forward; dotted = reserved", fontsize=12)
        heading = (
            (
                "Controlled allocation: real InferState + Triton attention state\n"
                "4 GiB persistent + 256 MiB temporary/forward, 8 GiB process allocator limit"
            )
            if workload == "controlled"
            else (
                f"Real {model_name} LightLLM forward, m35 H200 GPU 2\n"
                "BF16, TP=1, prefill 4096 tokens, Triton attention, CUDA Graph disabled"
            )
        )
        if group.startswith("model_logprobs"):
            heading += "\nPrompt logprobs enabled; " + (
                f"{limit_label} process allocator budget" if group == "model_logprobs" else "no process allocator limit"
            )
        fig.suptitle(heading, fontsize=14, weight="bold")
        fig.savefig(args.directory / f"{group}_memory.png", dpi=160)
        fig.savefig(args.directory / f"{group}_memory.pdf")
        plt.close(fig)
    impact = [r for r in runs if r["metadata"]["args"].get("prompt_logprobs")]
    if impact:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
        for capped, ax in [(False, axes[0]), (True, axes[1])]:
            variants = [r for r in impact if bool(r["metadata"]["args"].get("model_limit_mib")) == capped]
            for r in sorted(variants, key=lambda r: r["metadata"]["args"]["ownership"]):
                m = r["metadata"]
                owner = m["args"]["ownership"]
                color = COLORS[owner]
                label = "Before fix: strong ref" if owner == "strong" else "After fix: weak ref"
                rows = [x for x in r["records"] if x["phase"] in ("baseline", "after_forward", "oom")]
                metric = "allocated" if capped else "peak"
                ax.plot([x["seconds"] for x in rows], [x[metric] / GIB for x in rows], color=color, lw=2.3, label=label)
                if m["oom_step"]:
                    t = rows[-1]["seconds"]
                    ax.scatter([t], [rows[-1][metric] / GIB], color=color, marker="x", s=65, zorder=4)
                    allocation = re.search(r"Tried to allocate ([0-9.]+ [GM]iB)", m.get("oom_message", ""))
                    allocation_label = allocation.group(1) if allocation else "CUDA memory"
                    ax.annotate(
                        f"OOM at step {m['oom_step']} ({t:.2f} s)\nNext +{allocation_label} allocation rejected",
                        (t, rows[-1][metric] / GIB),
                        xytext=(0.06, 0.32),
                        textcoords="axes fraction",
                        arrowprops={"arrowstyle": "->", "color": color},
                        color=color,
                        weight="bold",
                        ha="left",
                    )
                    if capped:
                        ax.set_xlim(-0.01, max(0.6, t * 2))
                    else:
                        ax.axhline(
                            m["gpu_total_bytes"] / GIB,
                            color="#333333",
                            ls="--",
                            lw=1,
                            label=f"GPU capacity: {m['gpu_total_bytes'] / GIB:.1f} GiB",
                        )
                if capped and owner == "strong":
                    ax.axhline(
                        m["allocator_budget_bytes"] / GIB,
                        color="#333333",
                        ls="--",
                        lw=1,
                        label=f"{limit_label} process allocator limit",
                    )
                    ax.text(
                        0.98,
                        0.05,
                        "Weak ref: 150 forwards completed",
                        transform=ax.transAxes,
                        ha="right",
                        color=COLORS["weak"],
                        weight="bold",
                    )
            ax.set(
                xlabel="Runtime after warmup (seconds)",
                ylabel=("Active GPU tensors (GiB)" if capped else "Peak active GPU tensors (GiB)"),
            )
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right", fontsize=9)
        axes[0].set_title("No allocator limit: natural GC memory peaks")
        axes[1].set_title(f"Same {limit_label} budget: allocated memory at runtime")
        fig.suptitle(
            "InferState reference cycle raises runtime memory peaks\n"
            f"Real {model_name} / 4096-token prompt logprobs / default Python GC / m35 H200 GPU 2",
            fontsize=14,
            weight="bold",
        )
        fig.savefig(args.directory / "memory_impact.png", dpi=180)
        fig.savefig(args.directory / "memory_impact.pdf")
        plt.close(fig)
    summary = []
    for r in runs:
        m = r["metadata"]
        summary.append(
            {
                k: m[k]
                for k in [
                    "args",
                    "startup_passed",
                    "completed",
                    "oom_step",
                    "baseline_allocated_bytes",
                    "peak_allocated_bytes",
                    "final_allocated_bytes",
                    "allocator_budget_bytes",
                    "output_tokens_unique",
                ]
            }
        )
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2))
    for r in runs:
        a = r["metadata"]["args"]
        stem = Path(a["output"]).stem
        with (args.directory / f"{stem}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=r["records"][0].keys())
            writer.writeheader()
            writer.writerows(r["records"])
    rows = []
    for r in runs:
        m, a = r["metadata"], r["metadata"]["args"]
        workload = (
            "受控分配"
            if a["workload"] == "controlled"
            else (f"{model_name} prompt logprobs" if a.get("prompt_logprobs") else f"{model_name} 普通 prefill")
        )
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in [
                    workload
                    + (
                        f" / {m['allocator_budget_bytes'] / GIB:g} GiB 预算"
                        if m.get("allocator_budget_bytes")
                        else " / 无限制"
                    ),
                    a["ownership"],
                    a["gc_mode"],
                    f"{m['peak_allocated_bytes'] / GIB:.3f}",
                    m["completed"],
                    m["oom_step"] or "无",
                    "通过" if m["startup_passed"] else "未通过",
                ]
            )
            + "</tr>"
        )
    charts = []
    for name in [
        "memory_impact",
        "model_logprobs_memory",
        "model_logprobs_uncapped_memory",
        "model_memory",
        "controlled_memory",
    ]:
        p = args.directory / f"{name}.png"
        if p.exists():
            charts.append(f'<img alt="{name}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">')
    report = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>InferState 循环引用显存实验</title>'
    report += (
        "<style>body{font-family:system-ui;max-width:1250px;margin:40px au"
        "to;line-height:1.7;color:#17242c}img{width:100%;margin:20px 0}tab"
        "le{border-collapse:collapse;width:100%}td,th{padding:9px;border-b"
        "ottom:1px solid #ccd}code{background:#eef;padding:2px}</style>"
    )
    report += "<h1>InferState 循环引用如何抬高运行时显存峰值</h1>"
    report += (
        f"<p>地点：m35 / H200 GPU 2。实验未使用 m34。真实模型为 {model_name}，BF16，TP=1，409"
        f"6-token prefill，Triton attention，CUDA Graph 关闭。运行的是 LightLLM mode"
        f"l.forward，无 HTTP 服务和并发请求。</p>"
    )
    if any(r["metadata"].get("process_model_reused_between_phases") for r in runs):
        report += (
            "<p>六阶段对照复用同一份已加载权重；每阶段开始前执行完整 GC 和分配器缓存清理，再单独预热检查。Qwen3.5 每轮调用 in"
            "it_hybrid_attention_state 初始化新请求的 conv/SSM 状态，避免前一请求的线性状态影响比较。预算取"
            "弱引用 prompt logprobs 实测峰值加 1 GiB 后向上取整；两组使用完全相同预算。</p>"
        )
    report += (
        "<p>对照只改变 attention → InferState 的所有权：strong 恢复修复前的强引用，weak 使用修复后的"
        "弱引用。权重、输入和算子不变；两组生成的最后 token 相同。所有运行由 exp 包装，环境、配置、源代码版本、逐轮数据及 GC"
        " 事件保存在原始 JSON 中。</p>"
    )
    report += (
        f"<p><b>OOM 边界说明：</b>受限 prompt logprobs 实验使用两组相同的 {limit_label} PyT"
        f"orch 进程分配器预算，模拟常驻权重/KV 占用后可用空间有限的场景。它是实际捕获的 torch.OutOfMemoryErro"
        f"r，不是画出的预测线；这组对照不要求用满整张 H200；另有无预算对照，用于观察自然 GC 峰值及实际物理显存 OOM。普通 pr"
        f"efill 不设置分配器预算。</p>"
    )
    report += (
        "<p>曲线统计 memory_allocated（活跃 CUDA 张量）；peak 来自每次 forward 的 max_memo"
        "ry_allocated，能覆盖采样点之间的临时峰值。memory_reserved 仅表示缓存分配器保留量，详图以虚线呈现。横轴"
        "时间仅含预热后的测量阶段。受控实验主动申请指定大小的张量，不代表具体模型的每轮临时张量大小。</p>"
    )
    report += (
        "<table><thead><tr>"
        + "".join(f"<th>{x}</th>" for x in ["负载", "引用", "GC", "峰值 GiB", "完成轮数", "OOM 轮次", "启动检查"])
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    report += "".join(charts)
    report += (
        "<p>结果范围：影响取决于 InferState 实际持有的张量大小、GC 时机以及显存余量；不能把受控分配的增长率直接外推为所有"
        "模型的增长率。final GC 后活跃显存回归基线，说明这里测到的是延迟释放，而不是不可回收的永久泄漏。</p></html>"
    )
    (args.directory / "report.html").write_text(report)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
