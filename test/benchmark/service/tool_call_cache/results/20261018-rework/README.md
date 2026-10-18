# Exploratory rework validation, 2026-10-18

These are 60 complete exploratory paired comparisons: 44 on superseded sources and 16 on candidate `5bae2ef5ea481bbf53da26007169c94737d86062`. They include neutral and negative outcomes. This folder does not include independent-corpus performance or GPU crossover evidence, and cannot establish a production benefit.

All completed comparisons below use Qwen3-4B BF16, TP=1, isolated H200 GPUs on a shared host, KV capacity 32,768 tokens, request limit 16,384 tokens, prefill batch 8,192 tokens and 32 running requests. Both policies use the same pinned existing image, attention backends, debug logging and dependency overlay. Physical KV capacity is identical; static-headroom changes admission only. No images were built. Full per-group image/model/dependency fingerprints and configuration accompany the numerical exports. Weight/tokenizer provenance remains in [the preceding study](../20261017/model-checksums.json).

The source is pinned public [CacheWise metadata](https://github.com/cachewise-project/cachewise-coding-traces/tree/181c435a090d328d00bbbee4c8eeb27d32f3abd2). Text is synthetic; actual generated tool results close the agent loop. Lengths, waits and controlled mixtures are explicitly scaled/capped. These are closed-loop workloads, not verbatim production traffic or an open-loop overload test. These frozen clients reset compacted history to the same system/reference scope; identical synthetic filler can therefore be reused across compactions. New clients provide an explicit fresh-context sensitivity and must be reported separately. Source snapshots differ between studies; replay namespaces differ too. Comparisons are paired within each phase, and studies are never pooled.

v11 has two validation seeds per case; v13/v14/v15r1 have one. Intervals, when present, are descriptive seed-cluster bootstrap intervals. Requests are not independent performance repetitions; there is no GPU crossover or independent-corpus performance proof here. v12 stopped during warmup on unexpected end states and has zero measured pairs. Raw requests, responses and logs remain private in the experiment store.

## Complete results

Ratios above 1 favor the candidate. A positive result against deliberately reduced static-headroom admission is not a win against ordinary disabled caching. Control columns explicitly distinguish them.

### budget-v11-validation.json.gz

| Candidate | Control | Case | Pairs / seeds | Wall ratio | Uncached input reduction (%) |
|---|---|---|---:|---:|---:|
| budget-only | historical eviction policy | mixed25-c16 | 2 / 2 | 1.0011x | 2.23 |
| budget-only | historical eviction policy | ordinary-c8 | 2 / 2 | 0.9979x | 0.00 |
| budget-only | historical eviction policy | pressure-c24 | 2 / 2 | 0.9999x | -0.11 |
| budget-only | historical eviction policy | tool-wait-c8 | 2 / 2 | 1.0072x | 2.18 |
| budget-only | historical eviction policy | trace-c16 | 2 / 2 | 1.0044x | -0.58 |
| budget-only | historical eviction policy | trace-c8 | 2 / 2 | 1.0003x | 4.03 |

[All aggregate and per-trial numeric metrics](budget-v11-validation.json.gz).

### stable-prefix-v13-validation.json.gz

| Candidate | Control | Case | Pairs / seeds | Wall ratio | Uncached input reduction (%) |
|---|---|---|---:|---:|---:|
| stable-prefix | budget-only | agent25-c16-fast | 1 / 1 | 1.0246x | 1.62 |
| stable-prefix | budget-only | agent25-c16-slow | 1 / 1 | 1.0002x | -1.48 |
| stable-prefix | budget-only | agent50-c16-fast | 1 / 1 | 1.0018x | 0.00 |
| stable-prefix | budget-only | trace-c16 | 1 / 1 | 1.0031x | -7.14 |
| stable-prefix | same source, disabled | agent25-c16-fast | 1 / 1 | 1.0193x | 1.62 |
| stable-prefix | same source, disabled | agent25-c16-slow | 1 / 1 | 1.0042x | -1.56 |
| stable-prefix | same source, disabled | agent50-c16-fast | 1 / 1 | 1.0035x | 0.00 |
| stable-prefix | same source, disabled | ordinary-c8 | 1 / 1 | 1.0005x | 0.00 |
| stable-prefix | same source, disabled | pressure-c24 | 1 / 1 | 0.9972x | 0.20 |
| stable-prefix | same source, disabled | trace-c16 | 1 / 1 | 1.0000x | -0.06 |
| stable-prefix | historical eviction policy | agent25-c16-fast | 1 / 1 | 1.0076x | 0.22 |
| stable-prefix | historical eviction policy | agent25-c16-slow | 1 / 1 | 1.0097x | 0.00 |
| stable-prefix | historical eviction policy | agent50-c16-fast | 1 / 1 | 1.0249x | -1.17 |
| stable-prefix | historical eviction policy | ordinary-c8 | 1 / 1 | 0.9978x | 0.00 |
| stable-prefix | historical eviction policy | pressure-c24 | 1 / 1 | 0.9739x | -0.00 |
| stable-prefix | historical eviction policy | trace-c16 | 1 / 1 | 0.9992x | 2.42 |

[All aggregate and per-trial numeric metrics](stable-prefix-v13-validation.json.gz).

### reserved-kv-v14-validation.json.gz

| Candidate | Control | Case | Pairs / seeds | Wall ratio | Uncached input reduction (%) |
|---|---|---|---:|---:|---:|
| reserved-KV | static-headroom control | agent25-c16-fast | 1 / 1 | 1.1141x | 11.40 |
| reserved-KV | static-headroom control | agent25-c16-slow | 1 / 1 | 1.0549x | 5.03 |
| reserved-KV | static-headroom control | agent50-c16-fast | 1 / 1 | 1.0323x | 1.93 |
| reserved-KV | static-headroom control | ordinary-c8 | 1 / 1 | 1.0102x | -0.02 |
| reserved-KV | static-headroom control | pressure-c24 | 1 / 1 | 1.0035x | -0.00 |
| reserved-KV | static-headroom control | trace-c16 | 1 / 1 | 0.9944x | 5.18 |
| reserved-KV | stable-prefix | agent25-c16-fast | 1 / 1 | 0.9672x | 9.08 |
| reserved-KV | stable-prefix | agent25-c16-slow | 1 / 1 | 0.9326x | -0.63 |
| reserved-KV | stable-prefix | agent50-c16-fast | 1 / 1 | 0.8453x | 1.16 |
| reserved-KV | stable-prefix | trace-c16 | 1 / 1 | 0.9974x | -8.67 |
| reserved-KV | same source, disabled | agent25-c16-fast | 1 / 1 | 0.9529x | 9.41 |
| reserved-KV | same source, disabled | agent25-c16-slow | 1 / 1 | 0.9379x | 3.08 |
| reserved-KV | same source, disabled | agent50-c16-fast | 1 / 1 | 0.8542x | 2.00 |
| reserved-KV | same source, disabled | ordinary-c8 | 1 / 1 | 1.0001x | -0.01 |
| reserved-KV | same source, disabled | pressure-c24 | 1 / 1 | 0.9036x | -0.53 |
| reserved-KV | same source, disabled | trace-c16 | 1 / 1 | 0.9930x | -9.36 |

[All aggregate and per-trial numeric metrics](reserved-kv-v14-validation.json.gz).

## Causal accounting

Final archived counters cover 2,072 candidate requests and 1,062 actual tool calls in each of v13 and v14, including one 24-request warmup. All 1,062 tool calls were detected in each. v13 lost 3,693,774 protected token leases through forced eviction. v14 reduced that count to zero, but budget demotions increased from 1,668,794 to 5,296,127 and consumed waiting leases decreased from 839,020 to 750,275 tokens. These are cumulative accounting events, not unique compute savings; workloads have equal prescribed cases but cross-study namespace tokenization differs. The result demonstrates that preventing forced eviction alone does not establish an end-to-end gain.

In the v14 agent25-fast comparison against disabled caching, agent uncached input fell 35.91%, yet mean agent request ratio was 0.9843x and independent ordinary request ratio was 0.8801x. Thus cache metrics alone cannot support a latency claim. The next candidate addresses returning continuations being queued behind the reservation. Its results must be reported separately.

[Complete final numeric counters](cache-counters-v13-v14.json). Earlier live diagnostic totals were partial and must not be presented as whole-study totals.

The benchmark-only static control is `0e68cc17feecbcf85e43d9a69c7ebe9471e10bab`, with retention disabled and constant 50% admission headroom. It retains the same physical cache, request and batch limits. This source is isolated on the `cache-headroom-control` branch and must not be merged as production behavior.

## Candidate with continuation priority: v15r1

These are one-seed validation results under the earlier shared-compaction identity assumption. They are not pooled with earlier sources. The service adds prefix-aware borrowing and bounded successful bypasses; all physical cache/batch/request limits remain identical.

| Control | Case | Pairs / seeds | Wall ratio | Uncached input reduction (%) |
|---|---|---:|---:|---:|
| static headroom (retention off) | agent25-c16-fast | 1 / 1 | 1.3766x | 20.76 |
| static headroom (retention off) | agent25-c16-slow | 1 / 1 | 1.2873x | 13.79 |
| static headroom (retention off) | agent50-c16-fast | 1 / 1 | 1.6116x | 34.55 |
| static headroom (retention off) | ordinary-c8 | 1 / 1 | 1.0104x | -0.02 |
| static headroom (retention off) | pressure-c24 | 1 / 1 | 1.3420x | 44.07 |
| static headroom (retention off) | trace-c16 | 1 / 1 | 1.0149x | 25.12 |
| same source, disabled | agent25-c16-fast | 1 / 1 | 1.1304x | 13.26 |
| same source, disabled | agent25-c16-slow | 1 / 1 | 1.1107x | 8.93 |
| same source, disabled | agent50-c16-fast | 1 / 1 | 1.2913x | 32.56 |
| same source, disabled | ordinary-c8 | 1 / 1 | 0.9994x | 0.00 |
| same source, disabled | pressure-c24 | 1 / 1 | 1.1274x | 49.60 |
| same source, disabled | trace-c16 | 1 / 1 | 0.9780x | 12.96 |
| reservation without continuation priority | agent25-c16-fast | 1 / 1 | 1.2344x | 8.79 |
| reservation without continuation priority | agent25-c16-slow | 1 / 1 | 1.2088x | 7.92 |
| reservation without continuation priority | agent50-c16-fast | 1 / 1 | 1.4852x | 28.68 |
| reservation without continuation priority | trace-c16 | 1 / 1 | 1.0070x | 23.96 |

[All candidate aggregate/per-trial metrics](continuation-v15r1-validation.json.gz). Ratios against the deliberately smaller static admission headroom are separate from disabled-cache-policy comparisons.

Ordinary cohort tail latency is a real tradeoff. In agent50-fast versus disabled, mean agent request ratio is 2.3037x and mean ordinary request ratio is 1.1314x, but ordinary p95 request ratio is 0.8494x and ordinary p95 TTFT ratio is 0.8024x. In agent25-slow, mean ordinary request ratio is 0.9424x. Do not claim that every latency percentile improves or that priority is free. Cohort metrics and all neutral/negative cases remain in the JSON.

Final counters now cover all 2,072 candidate requests / 1,062 actual tool calls for each of three versions. Candidate v15r1 consumed 3,388,092 token leases versus 750,275 in reservation-only v14. Budget demotions fell to 2,480,859; forced protected evictions rose to 327,736. Allowing useful continuations to run is more valuable here than reducing eviction events to zero. These remain cumulative accounting events, not unique compute savings. [Full final counters](cache-counters-v13-v15r1.json).

The completed pressure pair has 163 actual append continuations: cached tokens were 612,137 versus 9,201. Five context-reset requests had only 25 versus 26 cached tokens, so that observed gain did not come from repeated filler after compaction. [Numerical append/reset audit](pressure-append-diagnostic-v15r1.json). Future fresh-identity sensitivity still needs independent validation; the old frozen clients and results remain intact.
