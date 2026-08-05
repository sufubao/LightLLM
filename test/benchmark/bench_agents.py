"""Agent-style multi-turn open-loop benchmark + one-shot robustness/perf suite.

What it does
------------
N concurrent agent sessions, each with a distinct long system prompt (so a
cache_aware router spreads them across prefill nodes); turns within a session
re-send the growing history (KV reuse). Turn emission is OPEN-LOOP: one turn is
fired every 1/QPS seconds regardless of whether previous turns finished. If every
session is still busy when a tick fires, the tick is counted as "blocked" -- the
saturation signal.

Two modes
---------
1) Single bench (default). Sweep one operating point and print full stats.

   python test/benchmark/bench_agents.py \\
       --base-url http://127.0.0.1:8000/v1 --model qwen35 \\
       --qps 5 --num-sessions 32 --turns-per-session 6 \\
       --isl 2048 --osl 256 --duration-sec 60 --warmup-sec 10

2) Suite (--suite). Runs a scripted sequence: smoke -> QPS sweep (find the
   saturation knee) -> overload -> idle-then-burst, [+ long-run with --full].
   Prints a compact progress line per phase and a markdown summary table at the
   end; each phase's full output goes to <out>/<phase>.log and the table to
   <out>/report.md.

   python test/benchmark/bench_agents.py --suite                    # ~7-8 min
   python test/benchmark/bench_agents.py --suite --quick            # halve durations
   python test/benchmark/bench_agents.py --suite --full             # + 10-min long-run
   python test/benchmark/bench_agents.py --suite --isl 4096 --osl 512

Key flags
---------
  --qps             turn emission rate (open loop)
  --num-sessions    concurrent sessions; distinct prefix each (= concurrency cap)
  --turns-per-session  turns before a session is recycled into a fresh identity
  --isl / --osl     system-prompt length (tokens) / max output tokens per turn
  --duration-sec    emission window
  --warmup-sec      discard metrics before this (also used as the idle period in
                    the suite's idle_burst phase)
  --report-sec      live progress print interval (0 disables)
  --suite / --quick / --full / --out   suite mode controls

Reading the output
------------------
  - Saturation knee: the QPS step where achieved QPS falls below target, or TTFT
    p95 spikes.
  - blocked > 0: the session pool filled before the server did -- raise
    --num-sessions and rerun that step.
  - idle_burst fail>0: requests rejected right after an idle window (health
    misclassification / spurious 503).
  - error breakdown: top error strings tell you whether failures are timeouts,
    connection resets, or empty streams.
"""

import argparse
import asyncio
import contextlib
import io
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100.0) * (len(s) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return s[int(idx)]
    w = idx - lo
    return s[lo] * (1 - w) + s[hi] * w


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: 0.0 for k in ["mean", "p50", "p90", "p95", "p99", "max"]}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def filler(seed: int, target_chars: int) -> str:
    """Deterministic per-seed text of ~target_chars. Distinct seed -> distinct bytes."""
    out = []
    n = 0
    k = 0
    while n < target_chars:
        word = f"{seed:08d}-agentctx-{k} "
        out.append(word)
        n += len(word)
        k += 1
    return "".join(out)[:target_chars]


def make_system_prompt(seed: int, isl: int) -> str:
    # ~4 chars/token; distinct per session so cache_aware hashes the prefix to
    # different P nodes across sessions.
    return filler(seed, isl * 4)


def make_user_turn(session_seed: int, turn_idx: int) -> str:
    return f"[session {session_seed} turn {turn_idx}] " + filler(session_seed * 31 + turn_idx, 256)


class Session:
    __slots__ = ("sid", "system", "history", "pending", "turns_done")

    def __init__(self, sid: int, system: str):
        self.sid = sid
        self.system = system
        self.history: List[Dict[str, str]] = []
        self.pending = False
        self.turns_done = 0

    def reset(self, system: str):
        self.system = system
        self.history = []
        self.pending = False
        self.turns_done = 0


async def do_turn(
    client: AsyncOpenAI,
    model: str,
    sess: Session,
    osl: int,
    metrics: List[Dict[str, Any]],
    warmup_until: float,
):
    user_msg = make_user_turn(sess.sid, sess.turns_done)
    messages = [{"role": "system", "content": sess.system}] + sess.history + [{"role": "user", "content": user_msg}]
    t0 = time.perf_counter()
    first_t = last_t = prev_t = None
    itl_sum = 0.0
    itl_n = 0
    ntok = 0
    err = None
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=osl,
            temperature=0.7,
            stream=True,
        )
        async for chunk in resp:
            if not (chunk.choices and chunk.choices[0].delta):
                continue
            delta = chunk.choices[0].delta
            extra = getattr(delta, "model_extra", None) or {}
            text = (
                getattr(delta, "content", None)
                or extra.get("reasoning")
                or extra.get("reasoning_content")
                or extra.get("text")
                or getattr(delta, "reasoning", None)
            )
            if text:
                now = time.perf_counter()
                ntok += 1
                if first_t is None:
                    first_t = now
                else:
                    itl_sum += now - prev_t
                    itl_n += 1
                prev_t = now
                last_t = now
        end = time.perf_counter()
        if ntok < 1:
            err = "no tokens"
        else:
            # append assistant reply so the next turn's prefix includes it
            sess.history.append({"role": "assistant", "content": "(reply)"})
            if time.perf_counter() >= warmup_until:
                metrics.append(
                    {
                        "ttft": (first_t - t0) * 1000,
                        "itl": (itl_sum / itl_n * 1000) if itl_n else None,
                        "tpot": (((last_t - first_t) / (ntok - 1)) * 1000)
                        if (ntok > 1 and first_t and last_t)
                        else 0.0,
                        "lat": (end - t0) * 1000,
                        "turn_idx": sess.turns_done,
                        "sid": sess.sid,
                    }
                )
    except Exception as e:  # keep the run alive; record failure count only
        err = str(e)
    finally:
        sess.turns_done += 1
        sess.pending = False
        if err:
            metrics.append({"error": err, "turn_idx": sess.turns_done, "sid": sess.sid})


async def run(args):
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    interval = 1.0 / args.qps if args.qps > 0 else 0.0

    # fresh set of sessions; distinct system prompt per session
    sessions = [Session(sid=sid, system=make_system_prompt(sid, args.isl)) for sid in range(args.num_sessions)]
    rr_idx = [0]  # round-robin cursor
    metrics: List[Dict[str, Any]] = []
    fired = 0
    blocked = 0
    warmup_until = time.perf_counter() + args.warmup_sec

    print(
        f"agent bench: qps={args.qps} sessions={args.num_sessions} turns/session={args.turns_per_session} "
        f"isl={args.isl} osl={args.osl} duration={args.duration_sec}s warmup={args.warmup_sec}s"
    )

    start = time.perf_counter()
    end_at = start + args.duration_sec
    next_tick = start

    async def reporter():
        # periodic one-liner: progress + running TTFT/TPOT over metrics so far
        try:
            while True:
                await asyncio.sleep(args.report_sec)
                now = time.perf_counter()
                if now >= end_at:
                    return
                ok_now = [m for m in metrics if "error" not in m]
                errs = [m for m in metrics if "error" in m]
                elapsed = now - start
                done = len(ok_now)
                ttft_s = stats([m["ttft"] for m in ok_now])
                tpot_s = stats([m["tpot"] for m in ok_now if m["tpot"] > 0])
                in_flight = sum(1 for s in sessions if s.pending)
                last_err = f"  last_err={errs[-1]['error'][:120]!r}" if errs else ""
                print(
                    f"[{elapsed:5.1f}s] fired={fired:<5} blocked={blocked:<5} "
                    f"ok={done:<5} err={len(errs):<3} pend={in_flight:<3} "
                    f"achQPS={done / elapsed if elapsed > 0 else 0:.2f}  "
                    f"TTFT p50={ttft_s['p50']:.0f} p95={ttft_s['p95']:.0f} max={ttft_s['max']:.0f}  "
                    f"TPOT p50={tpot_s['p50']:.0f} p95={tpot_s['p95']:.0f}"
                    f"{last_err}"
                )
        except asyncio.CancelledError:
            pass

    reporter_task = asyncio.create_task(reporter()) if args.report_sec > 0 else None

    async def fire_one():
        nonlocal fired, blocked
        # pick next non-pending session (round-robin over ready ones)
        n = len(sessions)
        chosen: Optional[Session] = None
        for i in range(n):
            s = sessions[(rr_idx[0] + i) % n]
            if not s.pending:
                chosen = s
                rr_idx[0] = (rr_idx[0] + i + 1) % n
                break
        if chosen is None:
            blocked += 1  # saturation: no session has finished its previous turn
            return
        fired += 1
        # retire a finished session into a fresh identity so the pool stays populated
        if chosen.turns_done >= args.turns_per_session:
            chosen.reset(system=make_system_prompt(seed=10_000_000 + rr_idx[0] + fired, isl=args.isl))
        # claim before scheduling: fire_one doesn't await, so without this a
        # catch-up burst re-picks the same session before do_turn runs.
        chosen.pending = True
        asyncio.create_task(do_turn(client, args.model, chosen, args.osl, metrics, warmup_until))

    # open-loop arrival loop: fire at fixed rate regardless of completions
    while True:
        now = time.perf_counter()
        if now >= end_at:
            break
        if now >= next_tick:
            await fire_one()
            next_tick += interval
            if interval == 0:
                next_tick = now
        else:
            await asyncio.sleep(min(next_tick - now, 0.01))

    # drain in-flight
    deadline = time.perf_counter() + args.drain_sec
    while any(s.pending for s in sessions) and time.perf_counter() < deadline:
        await asyncio.sleep(0.05)
    reporter_task and reporter_task.cancel()
    if reporter_task:
        await reporter_task

    total_time = time.perf_counter() - start
    return summarize(metrics, fired, blocked, total_time, args)


def summarize(metrics, fired, blocked, total_time, args):
    ok = [m for m in metrics if "error" not in m]
    fail = [m for m in metrics if "error" in m]
    ttfts = [m["ttft"] for m in ok]
    itls = [m["itl"] for m in ok if m["itl"] is not None]
    tpots = [m["tpot"] for m in ok if m["tpot"] > 0]
    lats = [m["lat"] for m in ok]

    # break TTFT down by turn index to show prefix-cache benefit (turn 0 vs later)
    by_turn: Dict[int, List[float]] = {}
    for m in ok:
        by_turn.setdefault(m["turn_idx"], []).append(m["ttft"])

    def line(name, s):
        print(
            f"  {name:<30} mean={s['mean']:8.1f}  p50={s['p50']:8.1f}  "
            f"p90={s['p90']:8.1f}  p95={s['p95']:8.1f}  p99={s['p99']:8.1f}  max={s['max']:8.1f}  (ms)"
        )

    print("\n" + "=" * 96)
    print("AGENT OPEN-LOOP BENCHMARK RESULTS")
    print("=" * 96)
    print(f"  target QPS        : {args.qps}")
    print(f"  fired turns       : {fired}    blocked ticks (saturation): {blocked}")
    print(f"  completed (ok/fail): {len(ok)} / {len(fail)}")
    if fail:
        from collections import Counter

        c = Counter(m["error"][:160] for m in fail)
        print("  error breakdown:")
        for msg, n in c.most_common(5):
            print(f"    [{n:>3}] {msg}")
    print(f"  achieved turn QPS : {len(ok) / total_time:.2f} /s   (wall {total_time:.1f}s)")
    print("-" * 96)
    line("TTFT", stats(ttfts))
    line("ITL", stats(itls))
    line("TPOT", stats(tpots))
    line("Total turn latency", stats(lats))
    print("-" * 96)
    print("  TTFT by turn index (prefix grows each turn; later turns should be faster if cache hits):")
    for idx in sorted(by_turn):
        s = stats(by_turn[idx])
        print(
            f"    turn {idx:<3} n={len(by_turn[idx]):<5} mean={s['mean']:8.1f}  "
            f"p50={s['p50']:8.1f}  p90={s['p90']:8.1f} (ms)"
        )
    print("=" * 96 + "\n")

    from collections import Counter

    top_err = Counter(m["error"][:120] for m in fail).most_common(1)
    return {
        "target_qps": args.qps,
        "fired": fired,
        "blocked": blocked,
        "ok": len(ok),
        "fail": len(fail),
        "achieved_qps": len(ok) / total_time if total_time > 0 else 0.0,
        "wall": total_time,
        "ttft": stats(ttfts),
        "itl": stats(itls),
        "tpot": stats(tpots),
        "lat": stats(lats),
        "top_error": top_err[0][0] if top_err else None,
    }


def _run_phase(name: str, out_dir, base: dict, **kw) -> dict:
    args = make_args(**{**base, **kw, "report_sec": 5})
    print(f"\n>>> {name}  qps={args.qps} sessions={args.num_sessions} "
          f"isl={args.isl} osl={args.osl} dur={args.duration_sec}s warmup={args.warmup_sec}s", flush=True)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        res = asyncio.run(run(args))
    res["phase"] = name
    res["suite_wall"] = time.perf_counter() - t0
    Path(out_dir, f"{name}.log").write_text(buf.getvalue())
    ttft, tpot = res["ttft"], res["tpot"]
    err = f"  err={res['fail']}({res['top_error'][:40]})" if res["fail"] else ""
    print(f"    done in {res['suite_wall']:4.0f}s | ok={res['ok']:<4} fail={res['fail']:<3} "
          f"blocked={res['blocked']:<3} achQPS={res['achieved_qps']:.2f} "
          f"TTFT p50={ttft['p50']:.0f} p95={ttft['p95']:.0f}  "
          f"TPOT p50={tpot['p50']:.0f} p95={tpot['p95']:.0f}{err}", flush=True)
    return res


def _md_table(rows: list) -> str:
    head = ("| phase | targetQPS | fired | ok | fail | blocked | achQPS | "
            "TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | top error |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|")
    body = []
    for r in rows:
        t, p = r["ttft"], r["tpot"]
        body.append(
            f"| {r['phase']} | {r['target_qps']} | {r['fired']} | {r['ok']} | {r['fail']} | "
            f"{r['blocked']} | {r['achieved_qps']:.2f} | "
            f"{t['p50']:.0f} | {t['p95']:.0f} | {p['p50']:.0f} | {p['p95']:.0f} | "
            f"{(r['top_error'] or '')[:50]} |"
        )
    return head + "\n" + "\n".join(body)


def run_suite(args) -> None:
    import os

    d = 0.5 if args.quick else 1.0
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    base = {"base_url": args.base_url, "model": args.model, "isl": args.isl, "osl": args.osl}

    def dur(s):
        return max(10.0, s * d)

    phases = [
        ("smoke", dict(qps=1, num_sessions=2, turns_per_session=3, duration_sec=dur(20), warmup_sec=dur(5))),
        ("perf_q2", dict(qps=2, num_sessions=64, duration_sec=dur(60), warmup_sec=dur(15))),
        ("perf_q5", dict(qps=5, num_sessions=64, duration_sec=dur(60), warmup_sec=dur(15))),
        ("perf_q10", dict(qps=10, num_sessions=64, duration_sec=dur(60), warmup_sec=dur(15))),
        ("perf_q20", dict(qps=20, num_sessions=64, duration_sec=dur(60), warmup_sec=dur(15))),
        ("overload_q40", dict(qps=40, num_sessions=128, duration_sec=dur(60), warmup_sec=0)),
        ("idle_burst", dict(qps=20, num_sessions=64, duration_sec=dur(90), warmup_sec=dur(45))),
    ]
    if args.full:
        phases.append(
            ("longrun", dict(qps=8, num_sessions=64, duration_sec=dur(600), warmup_sec=dur(30), report_sec=30))
        )

    print(f"SUITE -> {args.base_url} model={args.model} isl={args.isl} osl={args.osl} "
          f"quick={args.quick} full={args.full} out={out_dir}", flush=True)
    t_start = time.perf_counter()
    results = [_run_phase(name, out_dir, base, **kw) for name, kw in phases]
    total = time.perf_counter() - t_start

    table = _md_table(results)
    print("\n" + "=" * 96)
    print(f"SUITE DONE in {total:.0f}s   ({len(results)} phases)")
    print("=" * 96)
    print(table)
    Path(out_dir, "report.md").write_text(
        f"# Bench suite report\n\nTarget: `{args.base_url}` model=`{args.model}` "
        f"isl={args.isl} osl={args.osl} quick={args.quick} full={args.full}\n\n"
        f"Total wall: {total:.0f}s\n\n{table}\n"
    )
    print(f"\nreport -> {Path(out_dir, 'report.md')}   (per-phase logs: {out_dir}/*.log)")


def build_parser():
    p = argparse.ArgumentParser(description="Agent multi-turn open-loop benchmark")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen35")
    p.add_argument("--qps", type=float, default=5.0, help="turn emission rate (open loop)")
    p.add_argument("--num-sessions", type=int, default=32, help="concurrent sessions; distinct prefix each")
    p.add_argument("--turns-per-session", type=int, default=6, help="turns before a session is recycled")
    p.add_argument("--isl", type=int, default=2048, help="system-prompt length (tokens) per session")
    p.add_argument("--osl", type=int, default=256, help="max output tokens per turn")
    p.add_argument("--duration-sec", type=float, default=60.0, help="emission window")
    p.add_argument("--warmup-sec", type=float, default=10.0, help="discard metrics before this")
    p.add_argument("--drain-sec", type=float, default=120.0, help="max wait for in-flight turns after window")
    p.add_argument("--report-sec", type=float, default=5.0, help="live progress print interval (0 disables)")
    p.add_argument("--suite", action="store_true", help="run the full robustness+perf suite instead of one bench")
    p.add_argument("--quick", action="store_true", help="(suite) halve all durations")
    p.add_argument("--full", action="store_true", help="(suite) add a 10-min long-run stability phase")
    p.add_argument("--out", default="bench_suite_out", help="(suite) dir for per-phase logs + report.md")
    return p


def make_args(**overrides):
    args = build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def main():
    args = build_parser().parse_args()
    if args.suite:
        run_suite(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
