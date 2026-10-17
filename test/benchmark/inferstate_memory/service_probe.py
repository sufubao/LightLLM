"""Passive instrumentation for an isolated HTTP experiment source snapshot.

Import/install only from that snapshot's basemodel.py. Never installed globally.
No manual GC and no CUDA synchronization in the request path.
"""
import functools
import gc
import json
import os
from pathlib import Path
import threading
import time
import weakref


class StrongRef:
    def __get__(self, instance, owner=None):
        return None if instance is None else instance.__dict__.get("_strong_infer_state")

    def __set__(self, instance, value):
        instance.__dict__["_strong_infer_state"] = value


def install(model_class):
    directory = os.environ.get("INFERSTATE_HTTP_PROBE_DIR")
    if not directory or getattr(model_class, "_memory_probe_installed", False):
        return
    model_class._memory_probe_installed = True
    from lightllm.common.basemodel.attention.base_att import BasePrefillAttState, BaseDecodeAttState
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.utils.envs_utils import get_model_init_status
    import torch

    if os.getenv("INFERSTATE_HTTP_OWNERSHIP") == "strong":
        BasePrefillAttState.infer_state = StrongRef()
        BaseDecodeAttState.infer_state = StrongRef()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    refs = {}
    model_ref = None
    descriptor = None
    started = False
    forward_index = 0

    def phase():
        try:
            return (directory / "phase").read_text().strip()
        except OSError:
            return "startup"

    def emit(event, **extra):
        nonlocal descriptor
        if descriptor is None:
            descriptor = os.open(
                str(directory / f"worker_{os.getpid()}.jsonl"), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600
            )
        payload = dict(
            time=time.time(),
            event=event,
            phase=phase(),
            pid=os.getpid(),
            forward=forward_index,
            allocated=torch.cuda.memory_allocated(0),
            reserved=torch.cuda.memory_reserved(0),
            **extra,
        )
        os.write(descriptor, (json.dumps(payload) + "\n").encode())

    def live_counts():
        cached = set()
        model = None if model_ref is None else model_ref()
        if model is not None:
            for attribute in ("graph", "prefill_graph"):
                manager = getattr(model, attribute, None)
                for entry in getattr(manager, "graph", {}).values():
                    for value in entry:
                        if isinstance(value, InferStateInfo):
                            cached.add(id(value))
        result = dict(prefill_uncached=0, decode_uncached=0, graph_cached=len(cached))
        for identifier, (ref, stage) in refs.items():
            if identifier not in cached and ref() is not None:
                result[stage + "_uncached"] += 1
        return result

    def gc_event(event, info):
        if event == "stop" and torch.cuda.is_initialized():
            emit("gc", generation=info["generation"], collected=info["collected"], **live_counts())

    def monitor():
        while True:
            emit("sample")
            time.sleep(0.1)

    original_create = model_class._create_inferstate

    @functools.wraps(original_create)
    def create(self, model_input, *args, **kwargs):
        state = original_create(self, model_input, *args, **kwargs)
        refs[id(state)] = (weakref.ref(state), "prefill" if model_input.is_prefill else "decode")
        return state

    model_class._create_inferstate = create

    original_forward = model_class.forward

    @functools.wraps(original_forward)
    def forward(self, model_input):
        nonlocal started, model_ref, forward_index
        model_ref = weakref.ref(self)
        if not started:
            started = True
            emit(
                "worker_metadata",
                ownership=os.getenv("INFERSTATE_HTTP_OWNERSHIP"),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),
                gc_threshold=gc.get_threshold(),
                disable_cudagraph=self.disable_cudagraph,
                model_dir=self.weight_dir_,
                dtype=str(self.data_type),
                visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
            )
            gc.callbacks.append(gc_event)
            threading.Thread(target=monitor, daemon=True).start()
        forward_index += 1
        torch.cuda.reset_peak_memory_stats(0)
        stage = "prefill" if model_input.is_prefill else "decode"
        try:
            output = original_forward(self, model_input)
        except torch.OutOfMemoryError as exc:
            emit(
                "oom",
                stage=stage,
                error=str(exc),
                peak=torch.cuda.max_memory_allocated(0),
                initialized=get_model_init_status(),
                **live_counts(),
            )
            raise
        emit(
            "forward",
            stage=stage,
            batch_size=model_input.batch_size,
            input_tokens=0 if model_input.input_ids is None else model_input.input_ids.numel(),
            peak=torch.cuda.max_memory_allocated(0),
            initialized=get_model_init_status(),
            **live_counts(),
        )
        if forward_index % 128 == 0:
            for identifier in [key for key, (ref, _) in refs.items() if ref() is None]:
                del refs[identifier]
        return output

    model_class.forward = forward
