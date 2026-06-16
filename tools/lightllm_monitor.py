#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lightllm_monitor.py — LightLLM /metrics 实时终端仪表盘。

依赖:  pip install rich
用法:  python lightllm_monitor.py --url http://localhost:8000 --interval 2

指标口径 (来自 lightllm/server/metrics/metrics.py + httpserver/manager.py):
  TTFT = lightllm_request_first_token_duration        (秒, histogram)
  ITL  = lightllm_request_mean_time_per_token_duration(秒, histogram, 是平均间隔非逐token分布)
  gen/input token 计数 = *_generation_tokens_total / *_prompt_tokens_total (counter)
  TPM / tok/s 由相邻两次采样的 counter 差值除以时间得到 (瞬时外推, 非严格1分钟窗口)
"""

import argparse
import sys
import time
import urllib.request
from collections import defaultdict

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ───────────────────────── Prometheus 文本解析 ─────────────────────────


def parse_prometheus(text):
    """解析 /metrics 文本 -> [(name, labels_dict, value), ...]"""
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # name{labels} value   或   name value
        if "{" in line:
            brace_open = line.index("{")
            brace_close = line.rindex("}")
            name = line[:brace_open]
            labels_str = line[brace_open + 1 : brace_close]
            value_str = line[brace_close + 1 :].strip()
        else:
            parts = line.split(None, 1)
            name = parts[0]
            labels_str = ""
            value_str = parts[1].strip() if len(parts) > 1 else ""
        try:
            value = float(value_str.split()[0])
        except (ValueError, IndexError):
            continue
        labels = {}
        if labels_str:
            for kv in labels_str.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')
        samples.append((name, labels, value))
    return samples


def aggregate(text):
    """聚合成 (histogram_buckets, scalars)。

    histogram_buckets: {base_name: {le_float: cumulative_count}}  (跨 model_name label 累加)
    scalars:           {name: value}                              (counter+gauge 统一按值累加)
    """
    hist = defaultdict(dict)
    scalars = defaultdict(float)
    for name, labels, value in parse_prometheus(text):
        if name.endswith("_bucket"):
            base = name[: -len("_bucket")]
            le = labels.get("le", "+Inf")
            le_f = float("inf") if le == "+Inf" else float(le)
            hist[base][le_f] = hist[base].get(le_f, 0.0) + value
        elif name.endswith("_sum") or name.endswith("_count"):
            # 分位数用 bucket 算, 但 histogram 的 _sum/_count 用来算均值 (sum/count), 故保留为 scalar
            scalars[name] += value
        else:
            scalars[name] += value
    return hist, scalars


def quantile(buckets, q):
    """从 histogram bucket 算分位数 (相邻桶间线性插值), 返回原始单位值或 None"""
    if not buckets:
        return None
    total = buckets.get(float("inf")) or max(buckets.values())
    if total <= 0:
        return None
    target = q * total
    finite_les = sorted(le for le in buckets.keys() if le != float("inf"))
    if not finite_les:
        return None
    prev_le, prev_count = 0.0, 0.0
    for le in finite_les:
        count = buckets[le]
        if count >= target:
            if count == prev_count:
                return le
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return finite_les[-1]  # 落在最高有限桶与 +Inf 之间, 用最高有限桶上界


# ───────────────────────── 渲染辅助 ─────────────────────────

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values):
    """把数值序列渲染成 block 火花线"""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return SPARK[3] * len(values)
    return "".join(SPARK[min(7, int((v - lo) / (hi - lo) * 7))] for v in values)


def fmt_ms(seconds):
    """秒 -> 'xx.xms' (None -> '—')"""
    if seconds is None:
        return "[dim]—[/dim]"
    return f"{seconds * 1000:.1f}ms"


def fmt_int(x):
    if x is None:
        return "[dim]—[/dim]"
    return f"{int(x):,}"


def fmt_float(x, prec=1):
    if x is None:
        return "[dim]—[/dim]"
    return f"{x:.{prec}f}"


# 监控的 histogram 指标 (显示名 -> metric base name)
HIST_METRICS = [
    ("TTFT", "lightllm_request_first_token_duration"),
    ("ITL", "lightllm_request_mean_time_per_token_duration"),
]


def _kv_table():
    """6 列表: label val label val label val (label 左对齐 bold, value 右对齐绿)"""
    t = Table(box=None, padding=(0, 1), show_header=False, show_edge=False)
    for _ in range(3):
        t.add_column(style="bold")
        t.add_column(justify="right", style="green")
    return t


MTP_METRIC = "lightllm_request_mtp_avg_token_per_step"


def build_panel(hist, scalars, prev, now, ttft_history, gen_history, mtp_history, url, interval, status):
    # —— LATENCY ——
    lat = Table(box=None, padding=(0, 2), show_header=True, show_edge=False)
    lat.add_column("", style="bold")
    for q in ("p50", "p95", "p99"):
        lat.add_column(q, justify="right", style="yellow", header_style="dim")
    for label, base in HIST_METRICS:
        b = hist.get(base, {})
        lat.add_row(label, fmt_ms(quantile(b, 0.50)), fmt_ms(quantile(b, 0.95)), fmt_ms(quantile(b, 0.99)))

    # —— rate helper (相邻两次 counter 差值 / dt) ——
    def rate(name):
        if name in prev and name in scalars:
            pv, pt = prev[name]
            dt = now - pt
            if dt > 0:
                return (scalars[name] - pv) / dt
        return None

    gen_tps = rate("lightllm_generation_tokens_total")
    in_tps = rate("lightllm_prompt_tokens_total")
    tpm = gen_tps * 60 if gen_tps is not None else None

    # —— THROUGHPUT ——
    thr = _kv_table()
    thr.add_row(
        "gen tok/s",
        fmt_float(gen_tps),
        "input tok/s",
        fmt_float(in_tps),
        "gen tput g",
        fmt_float(scalars.get("lightllm_gen_throughput")),
    )
    thr.add_row(
        "TPM",
        f"{tpm:,.0f}" if tpm is not None else "[dim]—[/dim]",
        "req success",
        fmt_int(scalars.get("lightllm_request_success")),
        "req fail",
        fmt_int(scalars.get("lightllm_request_failure")),
    )

    # —— SERVER ——
    srv = _kv_table()
    srv.add_row(
        "running",
        fmt_int(scalars.get("lightllm_num_running_reqs")),
        "queued",
        fmt_int(scalars.get("lightllm_queue_size")),
        "batch",
        fmt_int(scalars.get("lightllm_batch_current_size")),
    )
    hit = scalars.get("lightllm_cache_hit_rate")
    hit_pct = f"{hit * 100:.0f}%" if hit is not None else "[dim]—[/dim]"
    srv.add_row(
        "cache hit",
        hit_pct,
        "req total",
        fmt_int(scalars.get("lightllm_request_count")),
        "infer steps",
        fmt_int(scalars.get("lightllm_batch_inference_count")),
    )

    # —— MTP 平均 accepted len (tokens/step, histogram sum/count) ——
    # 累计均值 = sum/count; 窗口均值 = 相邻两次采样的 (Δsum/Δcount), 反映近期 accept 水平
    sum_name, cnt_name = MTP_METRIC + "_sum", MTP_METRIC + "_count"
    total_sum, total_cnt = scalars.get(sum_name), scalars.get(cnt_name)
    mtp_cum = total_sum / total_cnt if total_sum is not None and total_cnt else None

    mtp_win = None
    if sum_name in prev and cnt_name in prev and total_sum is not None and total_cnt is not None:
        d_sum = total_sum - prev[sum_name][0]
        d_cnt = total_cnt - prev[cnt_name][0]
        if d_cnt > 0:
            mtp_win = d_sum / d_cnt

    mtp = _kv_table()
    mtp.add_row(
        "accept len (cum)",
        fmt_float(mtp_cum, 3),
        "accept len (recent)",
        fmt_float(mtp_win, 3),
        "p50",
        fmt_float(quantile(hist.get(MTP_METRIC, {}), 0.50), 3),
    )
    mtp.add_row(
        "done reqs",
        fmt_int(total_cnt),
        "p95",
        fmt_float(quantile(hist.get(MTP_METRIC, {}), 0.95), 3),
        "p99",
        fmt_float(quantile(hist.get(MTP_METRIC, {}), 0.99), 3),
    )

    # —— 趋势火花线 ——
    def trend_line(name, values, unit):
        return Text.assemble(
            (f"{name} trend  ", "bold"),
            (sparkline(values) or "[dim]▁[/dim]", "cyan"),
            (f"   (last {len(values)} ticks, {unit})", "dim"),
        )

    section = lambda t: Text(t, style="bold cyan")

    clock = time.strftime("%H:%M:%S")
    title = f"LightLLM Live Monitor   {clock}   ↻ {interval}s"

    body = Group(
        section("LATENCY"),
        lat,
        Text(""),
        section("THROUGHPUT"),
        thr,
        Text(""),
        section("SERVER"),
        srv,
        Text(""),
        section("MTP"),
        mtp,
        Text(""),
        trend_line("TTFT p50", ttft_history, "ms"),
        trend_line("gen tok/s", gen_history, "tok/s"),
        trend_line("MTP accept", mtp_history, "tok/step"),
        Text(""),
        Text(f"{url}/metrics  ·  {status}", style="dim"),
    )
    border = "cyan" if status == "ok" else "red"
    return Panel(body, title=title, border_style=border, padding=(1, 2))


# ───────────────────────── HTTP + 主循环 ─────────────────────────


def fetch(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description="LightLLM /metrics 实时终端仪表盘")
    ap.add_argument("--url", default="http://localhost:8000", help="LightLLM 服务地址 (默认 http://localhost:8000)")
    ap.add_argument("--interval", type=float, default=2.0, help="刷新间隔秒数 (默认 2)")
    ap.add_argument("--window", type=int, default=30, help="趋势火花线保留点数 (默认 30)")
    args = ap.parse_args()

    metrics_url = args.url.rstrip("/") + "/metrics"
    base_url = args.url.rstrip("/")
    prev = {}
    ttft_history, gen_history, mtp_history = [], [], []

    console = Console()
    try:
        with Live(console=console, refresh_per_second=8) as live:
            while True:
                t0 = time.time()
                try:
                    text = fetch(metrics_url)
                    hist, scalars = aggregate(text)
                    now = time.time()

                    ttft = quantile(hist.get("lightllm_request_first_token_duration", {}), 0.50)
                    if ttft is not None:
                        ttft_history.append(ttft)
                        if len(ttft_history) > args.window:
                            ttft_history.pop(0)

                    # 瞬时 gen tok/s 用于趋势
                    gen_tps = None
                    if "lightllm_generation_tokens_total" in prev:
                        pv, pt = prev["lightllm_generation_tokens_total"]
                        dt = now - pt
                        if dt > 0:
                            gen_tps = (scalars["lightllm_generation_tokens_total"] - pv) / dt
                    if gen_tps is not None:
                        gen_history.append(gen_tps)
                        if len(gen_history) > args.window:
                            gen_history.pop(0)

                    # 窗口 MTP accept len 用于趋势 (Δsum/Δcount)
                    sn, cn = MTP_METRIC + "_sum", MTP_METRIC + "_count"
                    if sn in prev and cn in prev and sn in scalars and cn in scalars:
                        d_cnt = scalars[cn] - prev[cn][0]
                        if d_cnt > 0:
                            mtp_history.append((scalars[sn] - prev[sn][0]) / d_cnt)
                            if len(mtp_history) > args.window:
                                mtp_history.pop(0)

                    panel = build_panel(
                        hist, scalars, prev, now, ttft_history, gen_history, mtp_history, base_url, args.interval, "ok"
                    )
                    prev = {k: (v, now) for k, v in scalars.items()}
                except Exception as e:
                    panel = build_panel(
                        defaultdict(dict),
                        {},
                        prev,
                        time.time(),
                        ttft_history,
                        gen_history,
                        mtp_history,
                        base_url,
                        args.interval,
                        f"error: {e}",
                    )

                live.update(panel)
                time.sleep(max(0.0, args.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        console.print("\n[dim]bye 👋[/dim]")


if __name__ == "__main__":
    main()
