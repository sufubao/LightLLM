# Tool-call-aware KV retention experiments

This directory distinguishes three kinds of evidence:

* `trace_data.py`: observations from pinned, public, sanitized CacheWise sessions.
  Multiple assistant chunks are grouped by request ID. Missing original finish
  reasons remain missing. Tool-content labels are reported separately. Final
  observations are censored; return probabilities are conditional on observing
  another request. Input readiness and the next response's first chunk are
  different timing measurements, neither an exact HTTP arrival timestamp.
* `smoke.py`: actual OpenAI tool generation and a live tool-result continuation.
  Tool definitions remain identical across turns to preserve the prompt prefix.
* `pressure_probe.py`: controlled residency diagnostic with real tool generation
  and independent, one-token completion requests that fill the cache. Repeating
  the original prompt isolates eviction from chat-history reserialization. This
  is not a production throughput result.
* `replay.py`: closed-loop, trace-shaped load against a real inference server.
  The released text is redacted, so contexts are synthetic. Actual model outputs
  feed forward into subsequent requests. Original context sizes and waits are
  scaled explicitly; wait caps and compactions are counted. Prompt-token hashes
  permit checking that paired runs received identical token sequences.

All benchmark/evaluation commands must be wrapped in `exp -m "purpose" ...`.
Save the exact service source revision/checksum, image digest, command line,
hardware, driver, model configuration, dataset revision and dataset checksums.
Raw downloaded traces and generated results belong outside the source tree.

## Comparison protocol

1. Compare the same source with retention disabled and enabled. Separately check
   the unchanged base revision for implementation overhead and regressions.
2. Keep model, precision, attention backend, cache-token capacity, batch limits,
   CUDA graph settings, load, client code and warmup identical. Exclude compilation
   and warmup from measured results. Do not run timed trials while another server
   is starting or warming up on the host.
3. Select sessions/windows with a fixed seed. Tune settings on the validation
   partition only; report held-out test partitions. Alternate trial order and
   exchange policy/GPU assignments to control hardware and temporal differences.
4. Include low pressure, cache pressure, overload, ordinary-only traffic,
   tool-heavy traffic, expired/abandoned continuations and oversized prefixes.
   Check TP=1 and TP>1, ordinary and hybrid attention. Record errors and unexpected
   finish states; failed runs remain visible and are not silently filtered.
5. Report cache-hit and newly prefilled tokens, client request latency, session
   completion time and throughput. Correlate server TTFT logs by `X-Request-Id`.
   Check prompt hashes, end states and output signatures between policies before
   interpreting performance. Report all repetitions and uncertainty, including
   neutral or negative outcomes. Do not extrapolate controlled residency gains
   into production throughput gains.

## Feature configuration

`--tool_call_cache_retention_seconds 30 --tool_call_cache_max_fraction 0.5`
enables temporary eviction preferences after normally stopped generations that
contain the configured parser's tool-start marker. Only tool-enabled chat
requests supply that marker. `length`, abort/error and paused requests receive
no new preference. The default duration of zero preserves ordinary eviction.
An ordinary stop or length-limited continuation also revokes stale preferences
along its completed branch, while preserving ancestors shared by other protected
branches. Abort/error and pauses do not revoke other requests' preferences.

The token detector is advisory, independent of API parsing: quoted or malformed
tool markers may receive a preference without changing the response. Prefixes
larger than the configured budget fall back to ordinary eviction. Preferences
include shared ancestors, count shared KV only once, expire without refreshing
recency, and never acquire references or prevent pressure-driven reclamation.
Hybrid recurrent-state eviction follows the same preference but still reclaims
unusable leaves first. TP ranks consume a common clock through the existing
scheduling broadcast, so expiry cannot diverge due to process-local clocks.

Example trace collection and analysis (replace `TRACE_SHA` with a full SHA):

```bash
python test/benchmark/service/tool_call_cache/trace_data.py download \
  --revision TRACE_SHA --directory /tmp/tool-call-cache/data
exp -m "Analyze pinned public coding trace continuation timing" \
  python test/benchmark/service/tool_call_cache/trace_data.py analyze \
  --directory /tmp/tool-call-cache/data --output /tmp/tool-call-cache/analysis.json
```

Example replay; save the service manifest alongside the result:

```bash
exp -m "Held-out trace-shaped replay; attach exact service/model/hardware configuration" \
  python test/benchmark/service/tool_call_cache/replay.py \
  --url http://127.0.0.1:8000 --model-dir /models/Qwen3-4B \
  --rounds /tmp/tool-call-cache/analysis.rounds.jsonl \
  --split test --sessions 8 --turns 8 --seed 42 \
  --output /tmp/tool-call-cache/replay.json
```

Dataset: [CacheWise coding traces](https://github.com/cachewise-project/cachewise-coding-traces),
[paper](https://arxiv.org/abs/2606.16824). Preserve the authors' attribution and
the source revision when publishing derived results.
