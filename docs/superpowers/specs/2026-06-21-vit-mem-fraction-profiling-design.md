# Fold ViT worst-case activation peak into LLM `mem_fraction` profiling

**Date:** 2026-06-21
**Status:** Design approved, pending implementation plan
**Scope:** All ViT families served by LightLLM's visual server

## Problem

When a ViT (vision encoder) is co-located on the same GPU as LLM ranks, LightLLM
OOMs frequently at runtime. The root cause is **not** "missing custom operators."
It is that the LLM's KV-pool sizing does not reliably account for the ViT's
*worst-case activation peak*:

- `MemoryManager.profile_size` (`lightllm/common/kv_cache_mem_manager/mem_manager.py:61`)
  sizes the KV pool from `torch.cuda.mem_get_info()` free memory
  (`lightllm/utils/profile_max_tokens.py:18`), minus a `(1 - mem_fraction)`
  headroom.
- The visual server is a **separate process** (`start_visual_process`,
  `lightllm/server/visualserver/manager.py:193`) and starts **before** the router
  via a blocking `"init ok"` barrier (`lightllm/utils/start_utils.py:29`,
  `manager.py:208`). So at profiling time the router's `mem_get_info` reading
  already excludes ViT **weights**.
- But ViT **activations** are dynamic. Only InternVL's `VisionTransformer`
  (`lightllm/models/vit/model.py:61`, `_check_max_len_infer`) runs a dummy
  worst-case pass at init and — because it does `del` but never `empty_cache()` —
  incidentally *holds* that high-water mark as reserved allocator memory, which
  the router then sees. The Qwen2/2.5/3-VL, Gemma3/4, etc. visual classes
  (`lightllm/server/visualserver/model_infer/model_rpc.py:78-111`) have **no**
  such pass → their activation peak is invisible to the router → co-located OOM.
- Even InternVL's pass is fragile: `MAX_PATH_NUM=13` (`vit/model.py:97`) is a
  fixed assumption that a real `image_patch_max_num` request can exceed, and the
  "no `empty_cache`" behavior is incidental, not guaranteed.

Runtime forwards are batch-bounded to `infer_max_batch_size`
(`model_rpc.py:247`), but per-image patch count is variable, so the reserved peak
must be derived from the configured caps, not assumed.

## Goal & non-goals

**Goal:** Guarantee the LLM KV-pool sizing accounts for the ViT worst-case
activation peak, for **all** ViT families, so co-located deployments stop OOMing.

**Non-goals:**
- Reducing the ViT's actual peak (kernel-level work).
- Changing separate-GPU / proxy / `visual_only` paths (they already degrade
  correctly — see Edge cases).
- The audio server (same class of bug; noted, out of scope).

## Approach: C (Hybrid)

Each visual rank, at init (before the `"init ok"` barrier), pushes a worst-case
dummy batch through its ViT and **holds** the resulting allocator high-water mark.
Because the router profiles *after* the barrier, `mem_get_info` already excludes
the ViT peak, so the KV pool is sized correctly **with no LLM-side arithmetic**
(self-enforcing, automatic for any TP/DP device mapping). A per-device
`SharedInt` carries the measured number across the process boundary **purely for
a startup log line and a fail-fast "does it fit" check** — never on the KV-sizing
hot path.

This was chosen over:
- **A (hold reservation only):** same physical mechanism, but no observability or
  early-failure check.
- **B (measure & declare budget):** router explicitly subtracts a per-device
  number; rejected for the hot path because per-device double-counting is easy to
  ship and hard to notice. Its observability benefit is retained in C as the
  diagnostic SharedInt.

## Components

### Component 1 — per-model worst-case dummy-input builder *(bulk of the work)*

New method on the visual-model contract:

```python
def build_worst_case_input(self) -> dict:  # kwargs for self.forward / self.encode
    ...
```

- **InternVL (`VisionTransformer`)**: refactor the existing `_check_max_len_infer`
  body into this method — `MAX_PATH_NUM × max_batch_size` images of
  `(3, IMAGE_H, IMAGE_W)`.
- **Qwen2/2.5/3-VL (variable resolution)**: derive worst case from existing caps:
  `infer_max_batch_size` images, each sized to
  `min(--max_image_pixels, tokens→pixels(--max_image_token_count))`. Compute
  `grid_thw` from that pixel budget (t=1; h·w from max pixels / patch²) and
  synthesize the matching `pixel_values`. **Derive strictly from the same caps
  that gate real requests**, so the dummy is a true upper bound by construction.
- **Gemma3/4 and fixed-resolution models**: `infer_max_batch_size` images at the
  model's native resolution.
- Models without a builder fall back to the override knob (Component 5) or current
  behavior **plus a warning**.

### Component 2 — generalized reserve-and-hold at ViT init

Base-class `reserve_worst_case_activation()`, called by every visual model after
weights load (generalizing today's InternVL-only `_check_max_len_infer`):

```python
torch.cuda.reset_peak_memory_stats(device)
out = self.forward(**self.build_worst_case_input())
del out
peak = torch.cuda.max_memory_reserved(device)   # high-water mark, HELD
# deliberately NO torch.cuda.empty_cache()  ← holding the reservation IS the mechanism
publish_vit_reserved(device, rank, peak)         # → SharedInt (Component 3)
```

The **"never `empty_cache` in the visual process"** invariant is load-bearing;
it gets an explicit comment and a code-review note. On failure, raise the same
actionable error as today (lower `--visual_infer_batch_size` / `--max_image_pixels`).

### Component 3 — cross-process budget channel

Reuse the existing `SharedInt` primitive
(`lightllm/server/router/dynamic_prompt/shared_arr.py`). One slot per
`(device_id, visual_rank)`, named off `get_unique_server_name()`. Visual ranks
**write** before `"init ok"`; the router **reads** after — the barrier guarantees
ordering, so no races. **Diagnostic only.**

### Component 4 — router-side validation + logging

After `profile_size`, the router reads the ViT reservations for its own device and
logs an attributed breakdown, e.g.:

```
[mem] device 0: total 80.0G | weights 28.4G | ViT worst-case 12.1G |
      headroom(1-mem_fraction) 8.0G | KV pool 31.5G (→ 215040 tokens)
```

**Fail-fast policy (Decision A — RESOLVED: hard-fail):** if the resulting
`max_total_token_num < max_req_total_len` (pool cannot hold even one max-length
request), raise a startup error that names the ViT reservation and the knobs to
turn (`--visual_infer_batch_size`, `--max_image_pixels`, `--mem_fraction`, or
`--visual_gpu_ids` to relocate the ViT). Emit a **warning** if the pool is below a
small multiple of that floor but still serviceable.

### Component 5 — manual override knob

**Decision B — RESOLVED: include it.** Add `--visual_reserved_mem_gb`. If set, the
visual rank reserves exactly that amount (a held guard tensor) and skips the dummy
pass. Backstop for models without a builder yet, or to override a bad estimate.

## Data flow (startup)

```
visual proc:  load ViT weights ─► reserve_worst_case_activation() ─► [HOLD peak]
                                  ─► publish SharedInt ─► "init ok"
                                                              │ (barrier)
router proc:        load LLM weights ─► profile_size() reads mem_get_info ◄┘
                    (free already excludes held ViT peak) ─► validate + log
```

## Edge cases / graceful degradation

- **ViT on a different GPU** (`--visual_gpu_ids`): reservation sits on that GPU;
  the LLM rank on its own GPU sees full memory → correct automatically.
- **Proxy / `visual_only` mode:** no local ViT → no reservation → no change.
- **Multiple visual DP/TP ranks on one GPU:** each holds its own physical
  reservation (additive for free); validation sums their SharedInt slots.
- **`DISABLE_CHECK_MAX_LEN_INFER`:** kept as an escape hatch; now also emits a
  "co-location OOM risk" warning.
- **Text-only models:** ViT path inert; pool sizing unchanged.

## Testing

- **Unit:** each `build_worst_case_input` produces shapes within the configured
  caps (CPU-constructible; GPU forward gated on CUDA availability).
- **Integration:** launch InternVL **and** a Qwen-VL on a single GPU with an
  aggressive `--visual_infer_batch_size`; assert the logged KV pool shrinks vs.
  text-only and that a max-size-image load test no longer OOMs (uses the
  `running-lightllm-with-docker` skill).
- **Regression:** text-only model → ViT path inert, pool size unchanged.

## Risks

- **Variable-resolution worst case (Qwen) is genuinely subtle.** If
  `build_worst_case_input` under-shoots the real peak, the held reservation is too
  small and OOM returns. Mitigation: derive strictly from the same caps that gate
  real requests; `--visual_reserved_mem_gb` as backstop.
- **The "no `empty_cache`" invariant is implicit** and could be violated by a
  future edit. Mitigation: comment + the SharedInt fail-fast check surfaces the
  regression loudly at startup rather than as a silent runtime OOM.

## Key file references

| Location | Role |
|---|---|
| `lightllm/common/kv_cache_mem_manager/mem_manager.py:61` | `profile_size` — KV pool sizing (router side) |
| `lightllm/utils/profile_max_tokens.py:18` | `mem_get_info` free-memory read |
| `lightllm/models/vit/model.py:61` | InternVL `_check_max_len_infer` (to generalize) |
| `lightllm/server/visualserver/model_infer/model_rpc.py:78-111` | per-family visual model construction |
| `lightllm/server/visualserver/manager.py:208` | `"init ok"` barrier send |
| `lightllm/utils/start_utils.py:29` | barrier receive (ordering guarantee) |
| `lightllm/server/router/dynamic_prompt/shared_arr.py` | `SharedInt` primitive |
| `lightllm/server/api_cli.py` | `--visual_infer_batch_size`, `--max_image_pixels`, new `--visual_reserved_mem_gb` |
