# On-demand profiling endpoints for LightLLM (`/start_profile`, `/stop_profile`)

Date: 2026-06-12. Survey of vLLM (commit `480fadab1`) and SGLang (commit `9fe8b7291`)
implementations, followed by a design for LightLLM's multi-process tree.

---

## Part 1 — How vLLM does it

### Plumbing

```
POST /start_profile                       entrypoints/serve/profile/api_router.py:21
  → AsyncLLM.start_profile()              v1/engine/async_llm.py:905
      ├─ frontend CPU torch.profiler      (profiles the HTTP/frontend process itself)
      └─ engine_core.profile_async()      v1/engine/core_client.py:1110  (zmq utility RPC)
          → EngineCore.profile()          v1/engine/core.py:624
            → executor.collective_rpc("profile")   v1/executor/abstract.py:260
              (shm ring buffer + zmq XPUB/SUB)     multiproc_executor.py:340
                → GPU worker .profile()   v1/worker/gpu_worker.py:877
```

- **Gating**: endpoints are only registered when `--profiler-config
  '{"profiler": "torch", "torch_profiler_dir": "..."}'` is passed. Returns empty 200.
- **Worker profiler** (`vllm/profiler/wrapper.py:160`): `torch.profiler.profile`
  with activities CPU+CUDA; knobs in `ProfilerConfig` (`vllm/config/profiler.py`):
  `with_stack` (default **True**), `record_shapes`/`memory`/`flops` (default False),
  `use_gzip` (True). Supports a torch profiler **schedule**
  (`delay/wait/warmup/active_iterations`) driven by `profiler.step()` called once
  per model iteration (`gpu_worker.py:758`).
- **Trace naming**: one file per rank via `tensorboard_trace_handler`,
  `{prefix}_dp{..}_pp{..}_tp{..}_rank{n}.pt.trace.json.gz`.
- **Second mode** `"profiler": "cuda"`: just `cudaProfilerStart/Stop`
  (`wrapper.py:310`) — used as a capture window for **Nsight Systems**.
- **CUDA graphs**: no special handling; docs recommend
  `nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node`.
- **Operational caveat**: stop/flush of large traces is slow → docs say raise
  `VLLM_RPC_TIMEOUT` to 30 min when profiling.

### Worth stealing from vLLM
1. Profiling the **frontend (HTTP) process** too, not just GPU workers.
2. The iteration **schedule** (warmup/active) concept.
3. The `"cuda"` profiler mode as an nsys capture window.

---

## Part 2 — How SGLang does it

### Plumbing

```
POST /start_profile  (rich JSON body)     entrypoints/http_server.py:956
  → TokenizerManager.start_profile()      managers/tokenizer_control_mixin.py:318
      → ProfileReq via zmq FanOutCommunicator to every scheduler (all DP ranks)
        → Scheduler dispatch              managers/scheduler.py:1340
          → SchedulerProfilerManager      scheduler_components/profiler_manager.py:45
```

Key structural fact: SGLang's scheduler is **colocated in the same process as the
TP worker**, so the profiler manager runs where the GPU work happens — only one
IPC hop is needed.

- **Request schema** (`ProfileReqInput`): `output_dir`, `start_step`,
  **`num_steps`** (auto-stop — no manual `/stop_profile` needed), `activities`
  (`CPU` / `GPU` / `MEM` / `RPD` / `CUDA_PROFILER`), `with_stack`,
  `record_shapes`, `profile_by_stage`, `merge_profiles`, `profile_prefix`.
- **Auto-stop**: target = `forward_ct + num_steps`; `_profile_batch_predicate()`
  (profiler_manager.py:366) is called once per forward batch and stops + flushes
  when the counter passes the target. `start_step` arms profiling to begin at a
  future step (skip warmup).
- **`profile_by_stage`**: separate per-stage counters and separate
  `-EXTEND` / `-DECODE` trace files for prefill vs decode.
- **`MEM` activity**: `torch.cuda.memory._record_memory_history(max_entries=1e5)`
  + `_dump_snapshot()` on stop.
- **Trace naming**: `{prefix}-{id}-TP-{t}[-DP-{d}][-PP-{p}][-EP-{e}].trace.json.gz`,
  with `torch.distributed.barrier` sync around export; optional **ProfileMerger**
  combines per-rank traces into one perfetto-friendly file.
- **Bench integration**: `sglang.bench_serving --profile` hits the endpoints
  automatically; PD-disaggregated profiling targets each node's own URL
  (`--profile-prefill-url` / `--profile-decode-url`).
- **CUDA graphs**: docs say to map kernels back to Python source you must run
  with `--disable-cuda-graph`; for nsys they recommend
  `--capture-range=cudaProfilerApi` + the `CUDA_PROFILER` activity, optionally
  with `--enable-layerwise-nvtx-marker` hooks.
- Env defaults: `SGLANG_TORCH_PROFILER_DIR`, `SGLANG_PROFILE_WITH_STACK`, etc.

### Worth stealing from SGLang
1. **`num_steps` auto-stop + `start_step`** — deterministic capture window, no
   manual-stop race, bounded trace size. The single best UX decision here.
2. `profile_by_stage` (prefill/decode separation).
3. `CUDA_PROFILER` activity as nsys capture window + memory-snapshot activity.
4. Per-rank naming convention + optional merge.

---

## Part 3 — Design for LightLLM

### The structural difference

vLLM and SGLang need 1–2 IPC hops to reach GPU code. LightLLM's chain is longer
and the scheduler (router) is **not** colocated with the GPU workers:

```
httpserver ──zmq PUSH──▶ router ──ShmObjsIOBuffer──▶ infer workers (N ranks)
                            │                            (infer_loop threads)
                            └──zmq──▶ detokenization
```

- httpserver → router currently carries only `GroupReqIndexes`
  (`router/manager.py:498` asserts the type).
- router → workers: rpyc is **init-only** (`model_rpc.py` exposes `init_model`);
  steady-state work and commands go through one shared `ShmObjsIOBuffer`
  (`router/manager.py:97`) with a ready flag that all TP ranks consume in
  lockstep at a step boundary (`base_backend.py:361-379`).
- There is an exact precedent for non-request commands riding this buffer:
  `AbortedReqCmd` / `StopStrMatchedReqCmd`
  (`core/objs/io_objs/group_req.py:31-38`, written by
  `_aborted_reqs()` at `router/manager.py:306-312`, dispatched worker-side in
  `_read_reqs_buffer_and_init_reqs()` at `base_backend.py:419-435`).

This is good news: the command channel we need already exists end to end, and —
unlike vLLM's out-of-band RPC — the shm-buffer broadcast delivers the command
to all ranks **rank-coordinated at the next request-read boundary**. (Not
"perfectly step-aligned": the actual `profiler.start()` calls can still differ
by thread/rank timing under the dual-thread overlap pipeline. For exact capture
boundaries, the cmd only *arms* a target forward counter, and every rank starts
at the same counted forward boundary — see "step semantics" below.)

Two channel caveats: the buffer is single-slot, so a profile cmd queues behind
in-flight batch payloads — semantics are "accepted, eventually applied", and the
router must use a bounded wait (surface a timeout into the status object rather
than spinning forever if a rank stops consuming).

A second consequence: in LightLLM the *host side* (router/httpserver/detoken,
each a separate pure-Python process) is a first-class bottleneck suspect (cf.
the eagle-overlap ~2.4 ms/step bubble). So unlike the baselines, host-process
profiling is not an afterthought here — it's half the point.

### API surface

Gated behind a new `--enable_profiling` launch flag (vLLM-style): when unset,
the endpoints return 501 and no profiler state is even constructed. The
HTTP-supplied `output_dir` must be validated against an allowlisted root
(default `$LIGHTLLM_TORCH_PROFILER_DIR`) — it's a filesystem path coming over
HTTP.

```
POST /start_profile        # api_http.py
{
  "output_dir":    str,          # default: $LIGHTLLM_TORCH_PROFILER_DIR or /tmp
  "num_steps":     int | null,   # auto-stop after N forward steps (recommended)
  "start_step":    int | null,   # arm: begin at a future forward step
  "activities":    ["CPU","GPU","MEM","CUDA_PROFILER"],   # default CPU+GPU
  "with_stack":    bool,         # default true
  "record_shapes": bool,         # default false
  "profile_prefix": str | null,
  "targets":       ["worker","router","httpserver"]       # default ["worker"]
}
POST /stop_profile         # manual stop (when num_steps not given)
GET  /profile_status       # LightLLM addition, see "Status" below
```

Both endpoints return **202 Accepted with the `profile_id`** immediately — the
response only confirms the command was queued, not that workers started
successfully (there is no reply channel). Clients poll `/profile_status` for
actual state, including per-rank start failures; flushing happens in the
background.

### Message plumbing (new pieces marked ★)

```
api_http.py  /start_profile
  → HttpServerManager.start_profile()                ★ httpserver/manager.py
      ├─ targets∋httpserver: start CPU profiler in-process            ★
      └─ send_to_router.send_pyobj(ProfileControlReq)                 ★
          → router _recv_new_reqs_and_schedule(): type dispatch       ★ (manager.py:498)
              ├─ targets∋router: RouterProfiler start/stop            ★
              └─ targets∋worker: write StartProfileCmd/StopProfileCmd ★
                   into shm_reqs_io_buffer (same pattern as _aborted_reqs)
                  → all ranks: _read_reqs_buffer_and_init_reqs() dispatch ★
                      → ModeBackend.profiler (WorkerProfilerManager)  ★
```

New objects:
- `ProfileControlReq` dataclass (httpserver→router zmq message) in
  `core/objs/io_objs/`.
- `StartProfileCmd` / `StopProfileCmd` dataclasses next to `AbortedReqCmd`.
- The router's `_recv_new_reqs_and_schedule` loses its
  `isinstance(GroupReqIndexes)` assert in favor of a small dispatch.

### Worker-side profiler (`WorkerProfilerManager`, the core)

Owned by `ModeBackend`; mirrors SGLang's `SchedulerProfilerManager`:

- **Start** (on cmd receipt, i.e. at a step boundary):
  `torch.profiler.profile(activities, with_stack, record_shapes).start()`.
  `MEM` → `torch.cuda.memory._record_memory_history`; `CUDA_PROFILER` →
  `cudaProfilerStart()` (nsys capture window).
- **Step semantics (critical, per Codex review)**: a "step" is an **actual
  model forward batch**, NOT an `infer_loop` iteration. The loop spins through
  idle `pass` iterations when there's no work, and DP overlap mode can launch
  two microbatches per loop call — counting iterations would make
  `num_steps`/`start_step` drift arbitrarily from real forward work. So the
  hooks are: `profiler.maybe_start(forward_ct)` immediately before a forward
  is launched, `profiler.step(forward_ct)` immediately after it (matching
  vLLM's per-iteration `profiler.step()` and SGLang's forward-batch
  predicate). The cmd received in `_read_reqs_buffer_and_init_reqs()` only
  *arms* the state machine; transitions happen at forward boundaries.
- **Overlap-mode hazard (LightLLM-specific)**: overlap mode runs **two**
  `infer_loop` threads per process (`base_backend.py:243-248`), with the
  second thread potentially mid-launch on a separate overlap stream when a
  start/stop fires. `torch.profiler` is process-global, so: (a) one shared
  state machine (IDLE → ARMED → RUNNING → FLUSHING) + shared forward counter
  guarded by a lock, transitions idempotent; (b) before `start()` and before
  `export`, **synchronize/drain the overlap pipeline** (wait for the other
  thread's in-flight forward events) so captures don't begin or end mid-step.
  MVP fallback if draining proves fiddly: refuse `/start_profile` in overlap
  mode unless a `force` flag is set, and document profiling with overlap
  disabled.
- **Export**: `export_chrome_trace` to
  `{output_dir}/{prefix}-{profile_id}-TP-{t}[-DP-{d}].trace.json.gz`, one file
  per rank, with a `dist.barrier()` before/after export like SGLang
  (all ranks see the cmd at the same step, so they stop together).
- **profile_id**: timestamp issued once in the HTTP process and carried in the
  cmd, so all ranks + router + httpserver traces of one capture share an id.

### Host-process profiling (router / httpserver)

- **Router** (`targets: ["router"]`): start a CPU-only `torch.profiler` (or
  `viztracer`, decided at implementation time) in the router process around its
  asyncio loop. Stop is **wall-clock based** (or on `/stop_profile`): the
  router's existing step counter (`counter_count` at `manager.py:220`) is a
  local variable inside `loop_for_fwd`, not accessible state, and router steps
  don't correspond 1:1 to worker forwards anyway. Exports
  `{prefix}-{id}-ROUTER.trace.json.gz`. This is the tool for "is the bubble in
  scheduling / shm-buffer waits / zmq?" questions.
- **httpserver**: same, in-process, trivially (it receives the HTTP call).
- **detokenization**: phase 2 — router forwards the cmd over its existing zmq
  socket to detoken, same dispatch pattern.
- Cheap alternative documented but not built: `py-spy dump/record -p <pid>` on
  any of these pure-Python processes.

### NVTX step-phase ranges (always-on, near-zero cost)

Independent of the endpoints, add `torch.cuda.nvtx.range` annotations:

- Worker `infer_loop`: one range per step (`step{ct} bs={n} mode={prefill|decode}`)
  with sub-ranges `prepare_inputs` / `forward` / `sample` / `post_handle`.
- Router `_step`: one range per scheduler iteration.

Rule: never emit NVTX inside a CUDA-graph-**captured** region (capture fails);
annotate around graph replay only. These make `nsys` timelines readable and are
what answers "GPU-bound or bubble-bound" at a glance.

### CUDA graphs caveat (document prominently)

Decode runs inside CUDA graphs: kernels still appear in torch traces (CUPTI
sees graph-launched kernels) but lose Python-op correlation. For kernel→source
attribution, rerun with `--disable_cudagraph` (`api_cli.py:554`) — same guidance
as SGLang. For graph-aware kernel timelines use
`nsys profile --cuda-graph-trace=node --capture-range=cudaProfilerApi` with
`activities: ["CUDA_PROFILER"]`.

### Status endpoint

There is no router→httpserver reply channel for control messages, and building
one just for acks is overkill. Instead: a shared-memory status **table indexed
by rank** (same shm pattern as `SharedTokenLoad`, cf. `/token_load` at
`api_http.py:209`) — one slot per worker rank plus slots for router/httpserver:
`{profile_id, rank, state, forward_ct, target_ct, error_code}`. A single scalar
status object cannot represent partial failure (rank 2's `profiler.start()`
threw while ranks 0-1 are recording) and would be racily overwritten by
multiple writers; per-rank slots give each writer exclusive ownership.
`GET /profile_status` aggregates the table and reports partial failures.
Lets benchmark scripts poll for "flushing finished" instead of guessing.

### Phasing

1. **Phase 1 (MVP, covers ~80%)**: `--enable_profiling` gate;
   `ProfileControlReq` + cmd plumbing; worker `torch.profiler` with
   forward-boundary `num_steps`/`start_step` auto-stop **including the
   overlap-mode drain (or the refuse-in-overlap fallback)**; per-rank gzip
   chrome traces; `LIGHTLLM_TORCH_PROFILER_DIR`; per-rank `/profile_status`
   shm table. (Per review: overlap-step semantics, per-rank status, and the
   safety gate are not deferrable.)
2. **Phase 2**: NVTX step-phase ranges; `CUDA_PROFILER` activity for nsys;
   router + httpserver CPU profiling; `MEM` snapshots.
3. **Phase 3**: `profile_by_stage`; trace merger; detoken target; `--profile`
   flag in the benchmark client (aiperf wrapper / test/benchmark scripts), like
   `sglang.bench_serving --profile`.

### Scope limits / risks

- **Single-node first.** `ShmObjsIOBuffer` is node-local shared memory; the
  typical profiling session is single-node anyway. PD-disaggregated setups are
  handled naturally: each prefill/decode node runs its own httpserver — hit each
  node's own port (SGLang does exactly this).
- **Trace size / flush time**: with `with_stack=True` traces grow fast; default
  docs should say "use `num_steps` ≤ 20 under load". Stop is async so no RPC
  timeout issue (unlike vLLM).
- **DP backend**: cmds ride the same buffer all DP ranks read; the cmd dispatch
  must run on every rank *before* any dp-index request filtering. Note
  `is_multinode_tp` (`base_backend.py:106`) only covers `nnodes>1, dp==1` —
  **multi-node DP** gets no automatic cross-node fanout; profile each node via
  its own router/HTTP port (documented limitation).
- **MTP/eagle**: draft + main model forwards happen in the same worker process
  and step, so they're captured together; NVTX sub-ranges (phase 2) are what
  separates them visually.
