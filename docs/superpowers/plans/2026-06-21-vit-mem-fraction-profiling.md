# ViT Worst-Case Activation Reservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the LLM's KV-pool sizing (`MemoryManager.profile_size`) reliably account for a co-located ViT's worst-case activation peak across all major ViT families, so same-GPU ViT+LLM deployments stop OOMing at runtime.

**Architecture:** Approach C (Hybrid). Each visual rank, at init (before the `"init ok"` barrier), pushes a worst-case dummy batch through its ViT and **holds** the allocator high-water mark (no `empty_cache`). Because the router profiles *after* the barrier, `torch.cuda.mem_get_info()` already excludes the ViT peak — the KV pool is sized correctly with zero LLM-side arithmetic. A per-device `SharedInt` carries the measured number across the process boundary, used only for a startup log line and an attributed fail-fast check (an augmentation of the existing `_check_mem_size` assert).

**Tech Stack:** Python, PyTorch CUDA allocator stats, LightLLM `SharedInt` (POSIX shared memory), Triton-backed ViT models, pytest.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `lightllm/server/api_cli.py` | Add `--visual_reserved_mem_gb` override knob | Modify |
| `lightllm/server/visualserver/model_infer/mem_reserve.py` | SharedInt naming, publish/read, guard-tensor allocation, pure worst-case shape math | Create |
| `lightllm/server/visualserver/model_infer/worst_case_reserve.py` | `WorstCaseReserveMixin` + `QwenVLWorstCaseMixin` (reserve-and-hold + per-family dummy builders) | Create |
| `lightllm/models/qwen2_vl/qwen2_visual.py` | Inherit `QwenVLWorstCaseMixin` | Modify |
| `lightllm/models/qwen2_5_vl/qwen2_5_visual.py` | Inherit `QwenVLWorstCaseMixin` | Modify |
| `lightllm/models/qwen3_vl/qwen3_visual.py` | Inherit `QwenVLWorstCaseMixin` | Modify |
| `lightllm/models/vit/model.py` | InternVL `VisionTransformer`: inherit `WorstCaseReserveMixin`, add builder, drop `__init__` dummy call | Modify |
| `lightllm/server/visualserver/model_infer/model_rpc.py` | Call reserve+publish after `.cuda()`, with override-knob precedence and fallback warning | Modify |
| `lightllm/common/basemodel/basemodel.py` | Router-side attributed log + fail-fast augmentation in `_check_mem_size` | Modify |
| `unit_tests/server/visualserver/test_mem_reserve.py` | Tests for SharedInt IO + worst-case shape math | Create |

---

### Task 1: Add `--visual_reserved_mem_gb` CLI argument

**Files:**
- Modify: `lightllm/server/api_cli.py` (near the visual args, after line 497)

- [ ] **Step 1: Add the argument**

In `lightllm/server/api_cli.py`, immediately after the `--visual_gpu_ids` argument block (ends at line 497), add:

```python
    parser.add_argument(
        "--visual_reserved_mem_gb",
        type=float,
        default=None,
        help="""Override the automatic ViT worst-case activation reservation. When set, each visual rank
        reserves exactly this many GB of GPU memory (held, not freed) and skips the dummy-image probe.
        Use as a backstop for models without an automatic worst-case builder, or to override a bad estimate.""",
    )
```

- [ ] **Step 2: Verify it parses**

Run: `python -c "from lightllm.server.api_cli import make_argument_parser; p = make_argument_parser(); a = p.parse_args(['--model_dir','x','--visual_reserved_mem_gb','3.5']); print(a.visual_reserved_mem_gb)"`
Expected: prints `3.5` (if the parser entry function has a different name, use `grep -n "def make_argument_parser\|ArgumentParser(" lightllm/server/api_cli.py` to find it; the assertion is only that the new flag parses to `3.5`).

- [ ] **Step 3: Commit**

```bash
git add lightllm/server/api_cli.py
git commit -m "feat(visual): add --visual_reserved_mem_gb override knob"
```

---

### Task 2: Create `mem_reserve.py` — SharedInt channel + guard tensor + shape math

**Files:**
- Create: `lightllm/server/visualserver/model_infer/mem_reserve.py`
- Test: `unit_tests/server/visualserver/test_mem_reserve.py`

- [ ] **Step 1: Write the failing test**

Create `unit_tests/server/visualserver/test_mem_reserve.py`:

```python
import math
import types
from lightllm.server.visualserver.model_infer.mem_reserve import (
    compute_qwen_worst_case_grid,
    publish_vit_reserved_mem,
    read_vit_reserved_mem_for_device,
)


def test_qwen_worst_case_grid_is_bounded_and_square():
    (total_patches, row_width), grid_thw = compute_qwen_worst_case_grid(
        batch_size=2,
        max_image_pixels=8294400,
        max_image_token_count=8192,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
    )
    # row width = in_channels * temporal_patch_size * patch_size**2
    assert row_width == 3 * 2 * 14 * 14
    # one [t, h, w] triple per image, sides are multiples of spatial_merge_size
    assert len(grid_thw) == 2
    for t, h, w in grid_thw:
        assert t == 1
        assert h % 2 == 0 and w % 2 == 0
    # token budget (8192 merged tokens * 4) is tighter than pixel budget here
    side = grid_thw[0][1]
    assert side == 180  # isqrt(32768)=181 -> floor to even -> 180
    assert total_patches == side * side * 2


def test_qwen_worst_case_respects_pixel_cap_when_tighter():
    (_, _), grid_thw = compute_qwen_worst_case_grid(
        batch_size=1,
        max_image_pixels=200704,  # 448*448 -> 1024 patches at patch_size 14
        max_image_token_count=8192,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
    )
    side = grid_thw[0][1]
    assert side * side <= 200704 // (14 * 14)


def test_shared_int_publish_and_read_sums_per_device():
    # device 0 has ranks 0 and 1; device 1 has rank 2
    publish_vit_reserved_mem(device_id=0, global_rank=0, reserved_bytes=1 * 1024 ** 3)
    publish_vit_reserved_mem(device_id=0, global_rank=1, reserved_bytes=2 * 1024 ** 3)
    publish_vit_reserved_mem(device_id=1, global_rank=2, reserved_bytes=5 * 1024 ** 3)

    args = types.SimpleNamespace(
        disable_vision=False,
        enable_multimodal=True,
        visual_gpu_ids=[0, 0, 1],
    )
    assert read_vit_reserved_mem_for_device(args, device_id=0) == 3 * 1024 ** 3
    assert read_vit_reserved_mem_for_device(args, device_id=1) == 5 * 1024 ** 3


def test_read_returns_zero_when_vision_disabled():
    args = types.SimpleNamespace(disable_vision=True, enable_multimodal=False, visual_gpu_ids=[0])
    assert read_vit_reserved_mem_for_device(args, device_id=0) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest unit_tests/server/visualserver/test_mem_reserve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lightllm.server.visualserver.model_infer.mem_reserve'`

- [ ] **Step 3: Write the implementation**

Create `lightllm/server/visualserver/model_infer/mem_reserve.py`:

```python
import math
from typing import List, Tuple
import torch
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_vit_reserved_shm_name(device_id: int, global_rank: int) -> str:
    return f"{get_unique_server_name()}_vit_reserved_mem_d{device_id}_r{global_rank}"


def publish_vit_reserved_mem(device_id: int, global_rank: int, reserved_bytes: int) -> None:
    """Visual rank writes its held worst-case reservation (bytes) for cross-process discovery."""
    shm = SharedInt(get_vit_reserved_shm_name(device_id, global_rank))
    shm.set_value(int(reserved_bytes))


def read_vit_reserved_mem_for_device(args, device_id: int) -> int:
    """Router side: sum reservations of all visual ranks placed on `device_id`. Diagnostic only."""
    if getattr(args, "disable_vision", False) or not getattr(args, "enable_multimodal", False):
        return 0
    gpu_ids = getattr(args, "visual_gpu_ids", None) or []
    total = 0
    for global_rank, dev in enumerate(gpu_ids):
        if dev == device_id:
            total += int(SharedInt(get_vit_reserved_shm_name(dev, global_rank)).get_value())
    return total


def reserve_guard_tensor(device_id: int, reserved_gb: float) -> Tuple[torch.Tensor, int]:
    """Allocate and HOLD a guard tensor of `reserved_gb` GB so the allocator high-water mark persists.
    Returns (tensor, nbytes). The caller MUST keep a reference to the tensor."""
    nbytes = int(reserved_gb * 1024 ** 3)
    guard = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{device_id}")
    return guard, nbytes


def compute_qwen_worst_case_grid(
    batch_size: int,
    max_image_pixels: int,
    max_image_token_count: int,
    patch_size: int,
    temporal_patch_size: int,
    in_channels: int,
    spatial_merge_size: int,
) -> Tuple[Tuple[int, int], List[List[int]]]:
    """Pure shape math for the Qwen-VL worst case.

    Returns ((total_patches, row_width), grid_thw) where pixel_values has shape
    (total_patches, row_width) and grid_thw is one [t, h, w] triple per dummy image.
    Bounds each image by BOTH the per-image token cap and pixel cap (whichever is tighter),
    using a near-square grid whose sides are multiples of spatial_merge_size.
    """
    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    patches_by_tokens = max_image_token_count * spatial_merge_unit
    patches_by_pixels = max_image_pixels // (patch_size * patch_size)
    max_patches = max(1, min(patches_by_tokens, patches_by_pixels))

    side = int(math.isqrt(max_patches))
    side -= side % spatial_merge_size
    side = max(side, spatial_merge_size)

    grid_h = grid_w = side
    row_width = in_channels * temporal_patch_size * patch_size * patch_size
    total_patches = grid_h * grid_w * batch_size
    grid_thw = [[1, grid_h, grid_w] for _ in range(batch_size)]
    return (total_patches, row_width), grid_thw
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest unit_tests/server/visualserver/test_mem_reserve.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add lightllm/server/visualserver/model_infer/mem_reserve.py unit_tests/server/visualserver/test_mem_reserve.py
git commit -m "feat(visual): SharedInt ViT-reservation channel + worst-case shape math"
```

---

### Task 3: Create `worst_case_reserve.py` — reserve-and-hold mixins

**Files:**
- Create: `lightllm/server/visualserver/model_infer/worst_case_reserve.py`

This task defines the mixins. The GPU-dependent `reserve_worst_case_activation` is exercised by the integration test (Task 8); the dummy-input *shape* logic delegates to the already-tested `compute_qwen_worst_case_grid`.

- [ ] **Step 1: Write the implementation**

Create `lightllm/server/visualserver/model_infer/worst_case_reserve.py`:

```python
import torch
from lightllm.server.visualserver.model_infer.mem_reserve import compute_qwen_worst_case_grid
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_RESERVE_OOM_HINT = (
    "ViT worst-case activation reservation hit OOM. Lower --visual_infer_batch_size, "
    "--max_image_pixels, or --max_image_token_count, or place the ViT on a separate GPU "
    "with --visual_gpu_ids."
)


class WorstCaseReserveMixin:
    """Adds a reserve-and-HOLD worst-case activation probe to a visual model.

    Subclasses MUST implement build_worst_case_input(...). The reservation is held by
    deliberately NOT calling torch.cuda.empty_cache() — the retained allocator high-water
    mark is what the LLM router observes via mem_get_info during KV-pool profiling.
    """

    def build_worst_case_input(self, batch_size, max_image_pixels, max_image_token_count) -> dict:
        raise NotImplementedError

    def run_worst_case_forward(self, dummy: dict):
        return self.forward(**dummy)

    @torch.no_grad()
    def reserve_worst_case_activation(
        self, device_id: int, batch_size: int, max_image_pixels: int, max_image_token_count: int
    ) -> int:
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats(device_id)
        try:
            dummy = self.build_worst_case_input(batch_size, max_image_pixels, max_image_token_count)
            out = self.run_worst_case_forward(dummy)
            del out, dummy
        except (RuntimeError, torch.OutOfMemoryError) as e:
            logger.exception(str(e))
            raise Exception(_RESERVE_OOM_HINT)
        # NB: intentionally NO torch.cuda.empty_cache() here — holding the high-water mark IS the mechanism.
        return int(torch.cuda.max_memory_reserved(device_id))


class QwenVLWorstCaseMixin(WorstCaseReserveMixin):
    """Worst-case builder for Qwen2/2.5/3-VL visual towers (shared forward(hidden_states, grid_thw))."""

    def build_worst_case_input(self, batch_size, max_image_pixels, max_image_token_count) -> dict:
        (total_patches, row_width), grid_thw = compute_qwen_worst_case_grid(
            batch_size=batch_size,
            max_image_pixels=max_image_pixels,
            max_image_token_count=max_image_token_count,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=self.in_channels,
            spatial_merge_size=self.spatial_merge_size,
        )
        # Derive dtype from the loaded weights rather than self.data_type — the latter is not
        # guaranteed to be a torch.dtype on every Qwen visual class; parameters() always is.
        dtype = next(self.parameters()).dtype
        hidden_states = torch.randn((total_patches, row_width), dtype=dtype, device="cuda")
        grid_thw_t = torch.tensor(grid_thw, dtype=torch.long, device="cuda")
        return {"hidden_states": hidden_states, "grid_thw": grid_thw_t}
```

- [ ] **Step 2: Verify the module imports**

Run: `python -c "from lightllm.server.visualserver.model_infer.worst_case_reserve import WorstCaseReserveMixin, QwenVLWorstCaseMixin; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add lightllm/server/visualserver/model_infer/worst_case_reserve.py
git commit -m "feat(visual): WorstCaseReserveMixin reserve-and-hold + Qwen-VL builder"
```

---

### Task 4: Wire Qwen2/2.5/3-VL classes to the mixin

**Files:**
- Modify: `lightllm/models/qwen2_5_vl/qwen2_5_visual.py:138`
- Modify: `lightllm/models/qwen2_vl/qwen2_visual.py` (class `Qwen2VisionTransformerPretrainedModel`)
- Modify: `lightllm/models/qwen3_vl/qwen3_visual.py` (class `Qwen3VisionTransformerPretrainedModel`)

All three classes already expose `self.patch_size`, `self.temporal_patch_size`, `self.in_channels`, `self.spatial_merge_size`, and `self.data_type`, and share a `forward(hidden_states, grid_thw, ...)` signature.

- [ ] **Step 1: Add the import + base to Qwen2.5-VL**

In `lightllm/models/qwen2_5_vl/qwen2_5_visual.py`, add to the imports (after line 18):

```python
from lightllm.server.visualserver.model_infer.worst_case_reserve import QwenVLWorstCaseMixin
```

Change the class declaration at line 138 from:

```python
class Qwen2_5_VisionTransformerPretrainedModel(nn.Module):
```

to:

```python
class Qwen2_5_VisionTransformerPretrainedModel(QwenVLWorstCaseMixin, nn.Module):
```

- [ ] **Step 2: Add the import + base to Qwen2-VL**

In `lightllm/models/qwen2_vl/qwen2_visual.py`, add the same import near the top, and change `class Qwen2VisionTransformerPretrainedModel(nn.Module):` to `class Qwen2VisionTransformerPretrainedModel(QwenVLWorstCaseMixin, nn.Module):`.
(Find the line with: `grep -n "class Qwen2VisionTransformerPretrainedModel" lightllm/models/qwen2_vl/qwen2_visual.py`.)

- [ ] **Step 3: Add the import + base to Qwen3-VL**

In `lightllm/models/qwen3_vl/qwen3_visual.py`, add the same import near the top, and change `class Qwen3VisionTransformerPretrainedModel(nn.Module):` to `class Qwen3VisionTransformerPretrainedModel(QwenVLWorstCaseMixin, nn.Module):`.
(Find the line with: `grep -n "class Qwen3VisionTransformerPretrainedModel" lightllm/models/qwen3_vl/qwen3_visual.py`.)

- [ ] **Step 4: Verify all three import and resolve the mixin**

Run:
```bash
python -c "from lightllm.models.qwen2_5_vl.qwen2_5_visual import Qwen2_5_VisionTransformerPretrainedModel as M; from lightllm.server.visualserver.model_infer.worst_case_reserve import QwenVLWorstCaseMixin; assert issubclass(M, QwenVLWorstCaseMixin); print('ok')"
python -c "from lightllm.models.qwen2_vl.qwen2_visual import Qwen2VisionTransformerPretrainedModel as M; from lightllm.server.visualserver.model_infer.worst_case_reserve import QwenVLWorstCaseMixin; assert issubclass(M, QwenVLWorstCaseMixin); print('ok')"
python -c "from lightllm.models.qwen3_vl.qwen3_visual import Qwen3VisionTransformerPretrainedModel as M; from lightllm.server.visualserver.model_infer.worst_case_reserve import QwenVLWorstCaseMixin; assert issubclass(M, QwenVLWorstCaseMixin); print('ok')"
```
Expected: prints `ok` three times. No circular import should occur: `worst_case_reserve` and `mem_reserve` import only `shared_arr`, `envs_utils`, and `log_utils` — never the model modules. If a cycle ever appears, that is the signal one of those helper modules wrongly imported a model module; fix the helper, do not work around it in the model.

- [ ] **Step 5: Commit**

```bash
git add lightllm/models/qwen2_vl/qwen2_visual.py lightllm/models/qwen2_5_vl/qwen2_5_visual.py lightllm/models/qwen3_vl/qwen3_visual.py
git commit -m "feat(visual): Qwen2/2.5/3-VL inherit QwenVLWorstCaseMixin"
```

---

### Task 5: InternVL `VisionTransformer` — mixin + builder, drop the in-`__init__` probe

**Files:**
- Modify: `lightllm/models/vit/model.py` (class `VisionTransformer`, lines 28-80)

The existing `_check_max_len_infer` (lines 59-80) runs the dummy pass inside `__init__` (called at line 56). We move that responsibility to `model_rpc` (Task 6) so every family is handled uniformly, and express InternVL's worst case as `build_worst_case_input`.

- [ ] **Step 1: Add the mixin import**

In `lightllm/models/vit/model.py`, after the existing imports (after line 22), add:

```python
from lightllm.server.visualserver.model_infer.worst_case_reserve import WorstCaseReserveMixin
```

- [ ] **Step 2: Add the mixin to the class and the builder**

Change `class VisionTransformer:` (line 28) to:

```python
class VisionTransformer(WorstCaseReserveMixin):
```

Remove the `self._check_max_len_infer()` call at line 56 (delete that single line inside `__init__`).

Replace the entire `_check_max_len_infer` method (lines 59-80) with the builder:

```python
    def build_worst_case_input(self, batch_size, max_image_pixels, max_image_token_count) -> dict:
        # InternVL uses fixed-size tiles: worst case is batch_size * MAX_PATH_NUM tiles of (3, IMAGE_H, IMAGE_W).
        num_tiles = int(self.MAX_PATH_NUM) * int(batch_size)
        dummy_images = torch.randn(
            (num_tiles, 3, self.IMAGE_H, self.IMAGE_W), dtype=self.data_type, device="cuda"
        )
        return {"pixel_values": dummy_images}
```

Note: `VisionTransformer.forward(self, pixel_values)` (line 161) already matches the `{"pixel_values": ...}` kwargs, so the mixin's default `run_worst_case_forward` works unchanged.

- [ ] **Step 3: Verify import + subclass + no leftover reference**

Run:
```bash
python -c "from lightllm.models.vit.model import VisionTransformer as V; from lightllm.server.visualserver.model_infer.worst_case_reserve import WorstCaseReserveMixin; assert issubclass(V, WorstCaseReserveMixin); assert not hasattr(V, '_check_max_len_infer'); print('ok')"
grep -n "_check_max_len_infer" lightllm/models/vit/model.py || echo "no leftover references"
```
Expected: prints `ok`, then `no leftover references`.

- [ ] **Step 4: Commit**

```bash
git add lightllm/models/vit/model.py
git commit -m "refactor(visual): InternVL uses WorstCaseReserveMixin; drop in-__init__ probe"
```

---

### Task 6: Wire `model_rpc` to reserve + publish (override precedence + fallback warning)

**Files:**
- Modify: `lightllm/server/visualserver/model_infer/model_rpc.py:116-134`

- [ ] **Step 1: Add the helper imports**

In `lightllm/server/visualserver/model_infer/model_rpc.py`, add to the imports (after line 28). Note `os` may already be imported — if `grep -n "^import os" lightllm/server/visualserver/model_infer/model_rpc.py` finds nothing, add it too:

```python
import os
from lightllm.server.visualserver.model_infer.mem_reserve import publish_vit_reserved_mem, reserve_guard_tensor
from lightllm.server.visualserver.model_infer.worst_case_reserve import WorstCaseReserveMixin
```

- [ ] **Step 2: Call the reservation after `.cuda()`**

In `exposed_init_model`, the model is moved to CUDA at line 116 (`self.model = self.model.cuda()`). Immediately after that line (and before the `if not self.is_visual_only_mode:` block at line 117), insert:

```python
            self._reserve_vit_worst_case_mem()
```

- [ ] **Step 3: Add the `_reserve_vit_worst_case_mem` method**

Add this method to `VisualModelRpcServer` (e.g. directly after `exposed_init_model`, before `exposed_run_task`):

```python
    def _reserve_vit_worst_case_mem(self):
        args = get_env_start_args()
        global_rank = self.dp_rank_id * self.vit_tp + self.tp_rank_id
        reserved_bytes = 0
        if getattr(args, "visual_reserved_mem_gb", None) is not None:
            # Manual override: hold an explicit guard tensor, skip the dummy probe.
            self._mem_reserve_guard, reserved_bytes = reserve_guard_tensor(
                self.device_id, args.visual_reserved_mem_gb
            )
        elif os.getenv("DISABLE_CHECK_MAX_LEN_INFER", None) is not None:
            # Preserved escape hatch: probe disabled. Reservation is skipped → co-location OOM risk.
            logger.warning(
                "DISABLE_CHECK_MAX_LEN_INFER is set: skipping ViT worst-case reservation. "
                "A co-located LLM may OOM at runtime. Unset it, or set --visual_reserved_mem_gb instead."
            )
        elif isinstance(self.model, WorstCaseReserveMixin):
            reserved_bytes = self.model.reserve_worst_case_activation(
                self.device_id,
                self.infer_max_batch_size,
                args.max_image_pixels,
                args.max_image_token_count,
            )
        else:
            logger.warning(
                f"co-location OOM risk: model_type={self.model_type} has no ViT worst-case reservation. "
                f"Set --visual_reserved_mem_gb to reserve headroom, or place the ViT on a separate GPU "
                f"with --visual_gpu_ids."
            )
        publish_vit_reserved_mem(self.device_id, global_rank, reserved_bytes)
        logger.info(
            f"ViT rank {global_rank} on device {self.device_id} reserved "
            f"{reserved_bytes / 1024 ** 3:.2f} GB worst-case activation memory."
        )
```

- [ ] **Step 4: Verify the module imports and the method exists**

Run: `python -c "from lightllm.server.visualserver.model_infer.model_rpc import VisualModelRpcServer as S; assert hasattr(S, '_reserve_vit_worst_case_mem'); print('ok')"`
Expected: prints `ok`

- [ ] **Step 5: Commit**

```bash
git add lightllm/server/visualserver/model_infer/model_rpc.py
git commit -m "feat(visual): reserve+publish ViT worst-case mem at init (override precedence, fallback warn)"
```

---

### Task 7: Router-side attributed log + fail-fast augmentation

**Files:**
- Modify: `lightllm/common/basemodel/basemodel.py:199-218` (`_check_mem_size`)

The existing assert at lines 211-216 already enforces the floor (`max_seq_length <= max_total_token_num`). We add a ViT-attributed log line and enrich the assert message so a co-location shortfall names the ViT reservation and the knobs to turn.

- [ ] **Step 1: Add the breakdown read + log at the top of `_check_mem_size`**

In `_check_mem_size`, after `self.max_total_token_num = self.mem_manager.size` (line 200), insert:

```python
        from lightllm.server.visualserver.model_infer.mem_reserve import read_vit_reserved_mem_for_device
        from lightllm.utils.dist_utils import get_current_device_id

        device_id = get_current_device_id()
        vit_reserved_bytes = read_vit_reserved_mem_for_device(self.args, device_id)
        if vit_reserved_bytes > 0:
            logger.info(
                f"[mem] device {device_id}: co-located ViT worst-case reserved "
                f"{vit_reserved_bytes / 1024 ** 3:.2f} GB; KV pool max_total_token_num="
                f"{self.max_total_token_num}"
            )
```

- [ ] **Step 2: Enrich the floor assert message**

Replace the existing assert block (lines 210-216) with one that attributes a co-location shortfall:

```python
        if self.args.performance_mode != "personal":
            vit_hint = ""
            if vit_reserved_bytes > 0:
                vit_hint = (
                    f" A co-located ViT reserved {vit_reserved_bytes / 1024 ** 3:.2f} GB on this device; "
                    f"lower --visual_infer_batch_size / --max_image_pixels / --max_image_token_count, "
                    f"reduce --mem_fraction, or move the ViT to another GPU with --visual_gpu_ids."
                )
            assert self.max_seq_length <= self.max_total_token_num, (
                f"max_total_token_num must be >= max_seq_length, "
                f"got max_total_token_num={self.max_total_token_num}, "
                f"max_seq_length={self.max_seq_length}. "
                f"Try set --max_req_total_len a smaller value < {self.max_total_token_num}.{vit_hint}"
            )
```

- [ ] **Step 3: Verify the module still imports**

Run: `python -c "import lightllm.common.basemodel.basemodel; print('ok')"`
Expected: prints `ok` (the new imports inside `_check_mem_size` are function-local to avoid any import cycle at module load).

- [ ] **Step 4: Commit**

```bash
git add lightllm/common/basemodel/basemodel.py
git commit -m "feat(mem): attribute KV-pool shortfall to co-located ViT reservation"
```

---

### Task 8: Integration verification (GPU)

**Files:** none (manual / scripted verification using the `running-lightllm-with-docker` skill).

This task requires a GPU and a multimodal model checkpoint. It validates the end-to-end behavior the unit tests cannot.

- [ ] **Step 1: Baseline — launch a Qwen-VL co-located, capture the KV pool size**

Launch the server with the ViT on the same GPU as the LLM (default `--visual_gpu_ids`), with a deliberately large probe (`--visual_infer_batch_size 4`). In the logs, find:
- the new `ViT rank ... reserved X.XX GB worst-case activation memory.` line (visual process), and
- the `... is the profiled max_total_token_num ...` line (router) and the new `[mem] device 0: co-located ViT worst-case reserved ...` line.

Expected: `max_total_token_num` is **smaller** than the same model launched with `--disable_vision` (or with the ViT on a separate `--visual_gpu_ids`), confirming the ViT peak is now subtracted.

- [ ] **Step 2: OOM-regression check**

Send a request with the largest images the caps allow (near `--max_image_pixels`), at the configured concurrency. 
Expected: no CUDA OOM during ViT encode or LLM decode (previously this co-located config OOMed).

- [ ] **Step 3: Fail-fast check**

Relaunch with an aggressive probe that cannot fit (e.g. `--visual_infer_batch_size 64 --mem_fraction 0.95`).
Expected: startup aborts with the enriched assert from Task 7 naming the ViT reservation and the knobs — **not** a silent runtime OOM later.

- [ ] **Step 4: Override-knob check**

Relaunch with `--visual_reserved_mem_gb 6`.
Expected: the visual log shows `reserved 6.00 GB`, the dummy probe is skipped, and `max_total_token_num` reflects a 6 GB carve-out.

- [ ] **Step 5: Regression — text-only model unaffected**

Launch a text-only model (no `--enable_multimodal`).
Expected: `read_vit_reserved_mem_for_device` returns 0, no `[mem] ... ViT` line, `max_total_token_num` unchanged vs. before this feature.

- [ ] **Step 6: Run the kernel/unit suite for the touched area**

Run: `pytest unit_tests/server/visualserver/test_mem_reserve.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Lint**

Run: `black --line-length=120 lightllm/server/visualserver/model_infer/mem_reserve.py lightllm/server/visualserver/model_infer/worst_case_reserve.py lightllm/server/visualserver/model_infer/model_rpc.py lightllm/models/vit/model.py lightllm/common/basemodel/basemodel.py lightllm/server/api_cli.py`
Then: `pre-commit run --files <those files>`
Expected: black + flake8 pass.

---

## Notes for the implementer

- **The "no `empty_cache`" invariant is load-bearing.** The mixin's `reserve_worst_case_activation` must never call `torch.cuda.empty_cache()`, and no code in the visual process should release the allocator cache after init — that is precisely what keeps the high-water mark reserved for the router to observe. The Task 7 fail-fast surfaces a regression loudly if this is ever broken.
- **Coverage boundary (explicit, not a placeholder):** InternVL and Qwen2/2.5/3-VL get automatic worst-case reservation. Other families currently served by `model_rpc.py` (gemma3, gemma4, qwen, llava, tarsier, qwen3_omni) fall through to the fallback in Task 6 Step 3: they emit a co-location warning and rely on `--visual_reserved_mem_gb`. Extending automatic coverage to one of those is a self-contained follow-up — add the appropriate `*WorstCaseMixin` base + `build_worst_case_input` for that family's `forward` signature, mirroring Tasks 3-5.
- **Qwen3-VL `forward(hidden_states, grid_thw, **kwargs)`** accepts extra keyword args (deepstack). The dummy builder passes only `hidden_states` and `grid_thw`; if a future Qwen3 variant makes another input mandatory, the Task 8 Step 1 launch will surface it immediately.
```
