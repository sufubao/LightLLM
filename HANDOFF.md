# Feature Handoff: Co-located ViT worst-case memory reservation

**Branch:** `impl_vit` · **Commits:** `318368ed` → `f7e74dd4` (11 commits) · **Net:** ~302 insertions, 11 files

## 1. Problem & fix

When a multimodal model's **ViT (vision encoder) and LLM share the same GPU**, LightLLM
frequently OOMs at runtime. Root cause: the LLM sizes its KV-cache pool from free GPU
memory, but the ViT's *dynamic activation peak* was never accounted for — so the LLM
over-claimed memory the ViT later needed.

This feature makes each visual rank run a **worst-case dummy forward at startup and hold
the allocator high-water mark** (deliberately no `torch.cuda.empty_cache()`). Because the
router profiles memory *after* the visual server's startup barrier, the LLM's
`torch.cuda.mem_get_info()` already excludes the held ViT peak and sizes the KV pool
correctly — with **no LLM-side arithmetic**. A per-device `SharedInt` carries the measured
number to the router for an attributed log line and a fail-fast check.

Note: the fix is memory **pooling/profiling**, not "custom operators."

## 2. New config surface

- **`--visual_reserved_mem_gb <float>`** (default `None`): override knob. When set, the
  visual rank holds exactly that many GB and **skips** the auto-probe. Use as a backstop
  for unsupported model families or to override a bad estimate.
- **`DISABLE_CHECK_MAX_LEN_INFER=1`** (env, pre-existing): now a documented escape hatch —
  skips the reservation and logs a co-location-OOM-risk warning.

## 3. What changed (by area)

| Area | Change |
|---|---|
| `lightllm/server/visualserver/model_infer/mem_reserve.py` (new) | Per-`(device,rank)` `SharedInt` publish/read channel + guard-tensor helper + pure Qwen worst-case shape math |
| `lightllm/server/visualserver/model_infer/worst_case_reserve.py` (new) | `WorstCaseReserveMixin` (reserve-and-hold) + `QwenVLWorstCaseMixin` |
| `lightllm/models/{qwen2_vl,qwen2_5_vl,qwen3_vl}/*_visual.py`, `lightllm/models/vit/model.py` | Visual classes inherit the mixin; each exposes a `build_worst_case_input` |
| `lightllm/server/visualserver/model_infer/model_rpc.py` | At ViT init: reserve+publish, precedence = override knob → disable-env → auto-probe → warn-fallback |
| `lightllm/common/basemodel/basemodel.py` | Router-side attributed `[mem]` log + enriched fail-fast assert |
| `lightllm/server/visualserver/model_infer/__init__.py` | Lazy imports + PEP 562 `__getattr__` to break an import cycle the feature exposed |
| `lightllm/server/api_cli.py` | `--visual_reserved_mem_gb` |
| `unit_tests/server/visualserver/test_mem_reserve.py` (new) | 4 unit tests (shape math + SharedInt IO) |

**Auto-coverage:** InternVL (`internvl_chat`) + Qwen2/2.5/3-VL.
**Fallback (warn + `--visual_reserved_mem_gb`):** gemma3/4, qwen(1), llava, tarsier,
qwen3_omni — by design, not a silent gap.

## 4. How to verify (oracles = startup logs)

Launch co-located (ViT + LLM on one GPU, the default). Expect two log lines:

```
[model_rpc.py] ViT rank 0 on device 0 reserved X.XX GB worst-case activation memory.
[basemodel.py] [mem] device 0: co-located ViT worst-case reserved X.XX GB; KV pool max_total_token_num=N
```

`X.XX GB` is the **activation headroom above the ViT weights** (measured as
`max_memory_reserved − reserved_before_probe`), not the ViT's total footprint. E.g. Qwen3-VL-8B
reports ~1.83 GB and InternVL2_5-26B ~1.80 GB (the InternViT-6B *weights* are excluded). The
physical hold still covers the full peak — only the reported/attributed number is the tunable
activation component.

Verification matrix already run on H200 / Qwen3-VL-8B + InternVL2_5-26B
(please re-run independently):

| Scenario | Config | Expected |
|---|---|---|
| Feature shrinks pool | co-located, ON vs `DISABLE_CHECK_MAX_LEN_INFER=1` | ON pool **smaller** by the held ViT activation (~1.9 GB / ~13.8k tokens in our run) |
| Override knob | `--visual_reserved_mem_gb 6` | log shows `reserved 6.00 GB`; probe skipped; pool shrinks accordingly |
| Fail-fast | `--max_req_total_len 3000000` (forces floor > pool) | **startup aborts** with assert naming the ViT GB + knobs (`--visual_infer_batch_size / --max_image_pixels / --max_image_token_count`, `--mem_fraction`, `--visual_gpu_ids`) |
| Separate-GPU | `--visual_gpu_ids 1` (ViT off the LLM's GPU) | LLM device reads **0**, no `[mem]` line, full pool |
| Text-only regression | any non-multimodal model | no reservation, no `[mem]` line, pool unchanged |
| Unit tests | `pytest unit_tests/server/visualserver/test_mem_reserve.py` | 4 passed |

**Test-env note:** the stock docker images don't contain this branch — **mount the worktree**
into the container so the branch code runs: `-v <worktree>:/lightllm` (we used
`ghcr.io/modeltc/lightllm:main`).

Reference: a co-located Qwen3-VL-8B run measured three internally-consistent pool sizes —
`690,746` (text-only, no ViT) → `672,615` (ViT weights present, feature off) →
`658,777` (ViT weights + activation reservation, feature on). The feature's effect is the
`672,615 → 658,777` carve-out.

## 5. Known limitations to probe

1. **Not exhaustively tested:** a sustained **max-size-image load test** to confirm zero
   runtime OOM. The mechanism (held reservation excluded from LLM profiling) is verified;
   the Qwen worst-case dummy is built from the same caps that bound real requests, so by
   construction runtime ≤ reserved — but a real image stress test under concurrency is the
   highest-value thing for the test team to add.
2. **InternVL worst case rests on `MAX_PATH_NUM` (=13 default).** If a deployment's actual
   `image_patch_max_num` exceeds it, the reservation can undershoot → OOM returns;
   `--visual_reserved_mem_gb` is the backstop. Worth a targeted test with high-tile
   InternVL inputs.
3. **Fallback families** (gemma/llava/tarsier/qwen3_omni) get only the warning + manual
   knob — confirm the warning fires and the knob reserves correctly for at least one of them.

## 6. Design docs

- `docs/superpowers/specs/2026-06-21-vit-mem-fraction-profiling-design.md`
- `docs/superpowers/plans/2026-06-21-vit-mem-fraction-profiling.md`
