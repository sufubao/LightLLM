# On-demand profiling (`/start_profile`)

Launch the server with `--enable_profiling`. Traces are written under
`LIGHTLLM_TORCH_PROFILER_DIR` (default `/tmp/lightllm_profile`), one gzipped
chrome trace per worker rank:
`{prefix}-{profile_id}-TP-{t}-DP-{d}.trace.json.gz`.
View them at https://ui.perfetto.dev.

## Capture a fixed window (recommended)

    curl -X POST http://HOST:PORT/start_profile -H "Content-Type: application/json" \
      -d '{"num_steps": 16, "activities": ["CPU", "GPU"], "with_stack": false}'

`num_steps` counts real model forward steps — idle scheduler iterations do not
count, and in MTP mode the draft-model forwards of a step count as part of that
step. The capture starts at the next forward after the command reaches the
workers and auto-stops after `num_steps` forwards (±1 under the dual-thread
overlap pipeline; see Caveats). `start_step` arms the capture to begin at an
absolute forward count (skip warmup).

Body fields: `output_dir` (must be under the `LIGHTLLM_TORCH_PROFILER_DIR`
root), `num_steps`, `start_step`, `activities` (`["CPU","GPU"]`), `with_stack`
(default true — large traces; turn off under load), `record_shapes`
(default false), `profile_prefix` (single path component).

## Manual stop

Omit `num_steps`, then `POST /stop_profile`.

## Status

A 202 from start/stop only means the command was queued — poll
`GET /profile_status`, which returns per-rank
`{state, profile_id, forward_ct, target_ct, error_code}` plus a router slot.
States: `idle / armed / running / flushing / error`. Error codes:
1 = profiler start failed, 2 = trace export failed, 3 = router could not
deliver the command (shm buffer busy). The capture is finished when all worker
slots return to `idle` with `error_code` 0.

## Caveats

- **CUDA graphs**: decode runs inside CUDA graphs — kernels appear in traces
  but lose Python-op correlation. For kernel→source attribution relaunch with
  `--disable_cudagraph`.
- **Single-thread CPU ops** (verified on GPU): kineto records CPU-op rows only
  for the infer thread that started the capture; forwards launched by the
  other overlap thread appear as GPU kernels without CPU correlation. GPU-side
  analysis is complete either way.
- **Serving stalls during flush**: stop/export runs synchronously in the infer
  thread; the node serves nothing while a slot shows `flushing`. Keep
  `num_steps` ≤ ~20 and `with_stack` off for low-stall captures.
- **±1 step window**: kineto start/stop must happen on the same thread, so a
  stop condition hit on the other infer thread is deferred one boundary.
- **Multi-node**: each node's HTTP port profiles that node's ranks only.
- Covered infer loops: chunked_prefill (default; incl. constraint/reward/PD
  subclasses) and dp_backend.
