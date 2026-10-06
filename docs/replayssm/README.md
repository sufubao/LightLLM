# ReplaySSM for GDN

ReplaySSM reduces speculative SSM storage by keeping one checkpoint per request
and replaying accepted updates. It is opt-in; `--ssm_state_mode native` remains
the default.

## Configuration

```bash
python -m lightllm.server.api_server \
    --model_dir /path/to/qwen3.5 \
    --ssm_state_mode replay \
    --linear_att_ssm_data_type bfloat16 \
    --replayssm_cache_len 8 \
    --replayssm_projection_mode precompute
```

The existing MTP flags can be added to this command. History capacity must cover
the maximum verify width (`mtp_step + 1`), including dynamic MTP. Supported
capacities are 4, 8, 16, 32 and 64.

| State mode | Storage and update policy |
| --- | --- |
| `native` | Existing per-token speculative state snapshots. |
| `compact` | One state plus this round's raw inputs; merge the accepted prefix every round. Requires MTP. |
| `replay` | One state plus bounded accepted history; fold before the next verify window would overflow. Supports ordinary decode and MTP. |

Non-native modes currently support Qwen3-Next and Qwen3.5 GDN with FP32 or BF16
state. Other models retain their native path. `--replayssm_projection_mode`
selects `inline` (default) or `precompute`; the latter is only valid with replay.
Choose the projection mode and capacity using the target workload.

## State lifecycle and precision

Verify overwrites the uncommitted history suffix. Acceptance advances only the
accepted prefix, so rejected proposals never enter a checkpoint. Fold alternates
history buffers to avoid overwriting records still read by another CTA. Padding
and the HOLD request do not advance history.

Prefill folds accepted updates before using the existing chunk kernel. CPU
prefix checkpoints and PD pages contain canonical SSM state and the accepted
conv window, without replay scratch. Exporting a checkpoint does not mutate
the active state. Restoring a checkpoint or reusing a request slot invalidates
history and resets the MTP offset.

Replay accumulates in FP32 and rounds when folding or materializing a checkpoint.
BF16 native and compact modes round after each token. Replay therefore does not
promise bitwise equivalence or identical generated text to native BF16. The
`precompute` path also uses activation-precision projections. Correctness tests
use an independent recurrence with each mode's rounding boundary.

The SSM autotuner measures prepare, forward, acceptance and fold on disposable
state. Its selected layout is fixed before CUDA Graph capture and reused for
verification and checkpoint reconstruction; serving state is never a tuning
input. Cached configurations distinguish precision, stride, heads, verify width,
history capacity and projection mode.

## Memory and performance

For a TP4 model with 48 GDN layers, 12 local value heads and 128x128 BF16 state,
one full-model checkpoint occupies 18 MiB per GPU. Native MTP3 uses 72 MiB per
request. Replay L8/precompute uses about 24.07 MiB including history and cursors.
Conv state, full-attention KV, request tables and sampling buffers are additional
allocations.

More KV space improves throughput only when the workload and request capacity
can use it. Increasing request capacity can also increase per-token latency.
Keep algorithm comparisons at equal capacity separate from capacity comparisons.

## Tests

```bash
PYTHONPATH=. python -m pytest -q \
    unit_tests/server/test_ssm_state_mode.py \
    unit_tests/common/basemodel/attention/linear/test_gdn.py \
    unit_tests/common/basemodel/triton_kernel/linear_att/test_replayssm*.py \
    unit_tests/common/basemodel/triton_kernel/linear_att/test_ssm_autotune.py \
    unit_tests/common/basemodel/triton_kernel/linear_att/test_mtp_state_params.py \
    unit_tests/common/basemodel/triton_kernel/linear_att/test_acceptance_fusion.py \
    unit_tests/common/test_paged_kv_transfer.py
```

CUDA tests cover partial acceptance, history folds, BF16 rounding, grouped keys,
dynamic verify lengths, CUDA Graph replay, HOLD padding, checkpoint/PD transfer,
slot reuse and fused acceptance. CPU tests cover startup validation and tuner
isolation. CUDA tests skip when a GPU is unavailable.
