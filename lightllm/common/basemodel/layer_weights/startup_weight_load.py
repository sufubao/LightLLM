import concurrent.futures
import dataclasses
import os
import threading
import time
from typing import Callable, List, Optional, Tuple

import torch

from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeight
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

CAPTURE_SAFE_WEIGHT_SENTINEL = 1e-3
_PREFETCH_BLOCK_SIZE = 16 * 1024 * 1024
_PREFETCH_STOP_TIMEOUT_SECONDS = 30


def _iter_weight_tensors(pre_post_layer, transformer_layer_list):
    """Yield every tensor owned by LightLLM's graph-visible weight objects."""
    seen_objects = set()
    seen_tensors = set()

    def visit(value, name):
        if isinstance(value, torch.Tensor):
            tensor_id = id(value)
            if tensor_id not in seen_tensors:
                seen_tensors.add(tensor_id)
                yield name, value
            return

        if isinstance(value, (BaseWeight, WeightPack)):
            object_id = id(value)
            if object_id in seen_objects:
                return
            seen_objects.add(object_id)
            for attr_name, attr in sorted(vars(value).items()):
                yield from visit(attr, f"{name}.{attr_name}")
            return

        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from visit(item, f"{name}[{index}]")
            return

        if isinstance(value, dict):
            for key in sorted(value, key=str):
                yield from visit(value[key], f"{name}[{key!r}]")

    roots = []
    if pre_post_layer is not None:
        roots.append(("pre_post", pre_post_layer))
    if transformer_layer_list is not None:
        roots.extend((f"layers[{index}]", layer) for index, layer in enumerate(transformer_layer_list))

    for root_name, root in roots:
        for attr_name, attr in sorted(vars(root).items()):
            if isinstance(attr, (BaseWeight, WeightPack, torch.Tensor, list, tuple, dict)):
                yield from visit(attr, f"{root_name}.{attr_name}")


@dataclasses.dataclass(frozen=True)
class TensorStorageMetadata:
    name: str
    tensor: torch.Tensor = dataclasses.field(repr=False, compare=False)
    data_ptr: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    storage_offset: int

    @classmethod
    def from_tensor(cls, name: str, tensor: torch.Tensor):
        return cls(
            name=name,
            tensor=tensor,
            data_ptr=tensor.data_ptr(),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
            storage_offset=tensor.storage_offset(),
        )

    def matches(self, other) -> bool:
        return self.tensor is other.tensor and (
            self.data_ptr,
            self.shape,
            self.stride,
            self.dtype,
            self.device,
            self.storage_offset,
        ) == (
            other.data_ptr,
            other.shape,
            other.stride,
            other.dtype,
            other.device,
            other.storage_offset,
        )


@dataclasses.dataclass(frozen=True)
class WeightStorageManifest:
    tensors: Tuple[TensorStorageMetadata, ...]
    device_type: Optional[str]

    @classmethod
    def capture(cls, pre_post_layer, transformer_layer_list, device_type: Optional[str] = "cuda"):
        entries = []
        for name, tensor in _iter_weight_tensors(pre_post_layer, transformer_layer_list):
            if device_type is None or tensor.device.type == device_type:
                entries.append(TensorStorageMetadata.from_tensor(name, tensor))
        return cls(tensors=tuple(entries), device_type=device_type)

    def changed_names(self, pre_post_layer, transformer_layer_list) -> Tuple[str, ...]:
        after = WeightStorageManifest.capture(
            pre_post_layer,
            transformer_layer_list,
            device_type=self.device_type,
        )
        before_by_id = {id(metadata.tensor): metadata for metadata in self.tensors}
        after_by_id = {id(metadata.tensor): metadata for metadata in after.tensors}
        changed = set()
        for tensor_id in sorted(before_by_id.keys() | after_by_id.keys()):
            before = before_by_id.get(tensor_id)
            current = after_by_id.get(tensor_id)
            if before is None:
                changed.add(current.name)
            elif current is None or not before.matches(current):
                changed.add(before.name)
        return tuple(sorted(changed))

    def unchanged_floating_names(self, value: float) -> Tuple[str, ...]:
        names = []
        checks = []
        for metadata in self.tensors:
            tensor = metadata.tensor
            if torch.is_floating_point(tensor):
                names.append(metadata.name)
                checks.append(torch.all(tensor == value))
        if not checks:
            return ()
        unchanged = torch.stack(checks).cpu().tolist()
        return tuple(name for name, is_unchanged in zip(names, unchanged) if is_unchanged)


@torch.no_grad()
def initialize_capture_safe_weights(
    pre_post_layer,
    transformer_layer_list,
    value: float = CAPTURE_SAFE_WEIGHT_SENTINEL,
    device_type: Optional[str] = "cuda",
) -> None:
    for _, tensor in _iter_weight_tensors(pre_post_layer, transformer_layer_list):
        if (device_type is None or tensor.device.type == device_type) and torch.is_floating_point(tensor):
            tensor.fill_(value)


class CheckpointPrefetchHandle:
    def __init__(self, paths: List[str], num_threads: int):
        self._paths = paths
        self._num_threads = num_threads
        self._cancel_event = threading.Event()
        self._errors = []
        self._thread = threading.Thread(target=self._run, name="checkpoint-page-cache-prefetch", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _prefetch_file(self, path: str) -> None:
        with open(path, "rb") as checkpoint_file:
            while not self._cancel_event.is_set():
                if not checkpoint_file.read(_PREFETCH_BLOCK_SIZE):
                    break

    def _run(self) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            pending = {executor.submit(self._prefetch_file, path): path for path in self._paths}
            for future in concurrent.futures.as_completed(pending):
                path = pending[future]
                try:
                    future.result()
                except Exception as error:
                    self._errors.append((path, error))

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    @property
    def errors(self):
        return tuple(self._errors)

    def wait(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("Timed out waiting for checkpoint page-cache prefetch")

    def stop(self, timeout: float = _PREFETCH_STOP_TIMEOUT_SECONDS) -> None:
        self._cancel_event.set()
        self.wait(timeout)


def start_checkpoint_prefetch(
    weight_dir: str,
    num_threads: int,
    rank_in_node: int,
    node_world_size: int,
) -> CheckpointPrefetchHandle:
    if num_threads < 1:
        raise ValueError("weight_loader_prefetch_num_threads must be at least 1")
    if not os.path.isdir(weight_dir):
        raise ValueError("startup weight overlap supports local checkpoint directories only")

    paths = sorted(os.path.join(weight_dir, name) for name in os.listdir(weight_dir) if name.endswith(".safetensors"))
    if not paths:
        raise ValueError("startup weight overlap requires a safetensors checkpoint")
    local_paths = paths[rank_in_node::node_world_size]
    logger.info(
        "Start checkpoint page-cache prefetch for %d/%d shards with %d threads on local rank %d.",
        len(local_paths),
        len(paths),
        num_threads,
        rank_in_node,
    )
    handle = CheckpointPrefetchHandle(local_paths, num_threads)
    handle.start()
    return handle


class StartupWeightLoadManager:
    """Overlap checkpoint page-cache staging with capture, then commit in place."""

    def __init__(
        self,
        pre_post_layer,
        transformer_layer_list,
        weight_dir: str,
        num_threads: int,
        rank_in_node: int,
        node_world_size: int,
    ):
        self.pre_post_layer = pre_post_layer
        self.transformer_layer_list = transformer_layer_list
        self.weight_dir = weight_dir
        self.num_threads = num_threads
        self.rank_in_node = rank_in_node
        self.node_world_size = node_world_size
        self.prefetch_handle = None
        self.started_at = None

    def prepare(self) -> None:
        initialize_capture_safe_weights(self.pre_post_layer, self.transformer_layer_list)
        torch.cuda.synchronize()
        self.started_at = time.perf_counter()
        self.prefetch_handle = start_checkpoint_prefetch(
            weight_dir=self.weight_dir,
            num_threads=self.num_threads,
            rank_in_node=self.rank_in_node,
            node_world_size=self.node_world_size,
        )

    def commit(self, load_weights: Callable[[], None]) -> None:
        if self.prefetch_handle is None:
            raise RuntimeError("startup weight overlap was not prepared")
        manifest = WeightStorageManifest.capture(self.pre_post_layer, self.transformer_layer_list)
        commit_started_at = time.perf_counter()
        try:
            load_weights()
            torch.cuda.synchronize()
        finally:
            self.stop_prefetch()

        changed_names = manifest.changed_names(self.pre_post_layer, self.transformer_layer_list)
        if changed_names:
            raise RuntimeError(
                "Startup weight commit changed graph-visible tensor storage: " + ", ".join(changed_names[:8])
            )
        unchanged_names = manifest.unchanged_floating_names(CAPTURE_SAFE_WEIGHT_SENTINEL)
        if unchanged_names:
            raise RuntimeError(
                "Startup weight commit did not replace capture-safe values: " + ", ".join(unchanged_names[:8])
            )
        logger.info(
            "Committed real weights after CUDA Graph capture in %.2fs (overlap window %.2fs).",
            time.perf_counter() - commit_started_at,
            commit_started_at - self.started_at,
        )

    def stop_prefetch(self) -> None:
        handle = self.prefetch_handle
        if handle is None:
            return
        self.prefetch_handle = None
        try:
            handle.stop()
        except TimeoutError as error:
            logger.warning("%s; the daemon prefetch thread will exit asynchronously.", error)
        if handle.errors:
            path, error = handle.errors[0]
            logger.warning(
                "Checkpoint prefetch had %d failure(s), first at %s: %s. Normal weight loading still ran.",
                len(handle.errors),
                path,
                error,
            )
