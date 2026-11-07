from dataclasses import dataclass, field, fields

import torch

from lightllm.core.contracts.provider import AttentionOffer, ProbeResult
from lightllm.core.compute.attention.batch import AttentionBatch
from .inputs import query_tensor
from .spec import DenseAttentionSpec


@dataclass
class AscendMetadata(AttentionBatch):
    tasks: list = field(default_factory=list)


class AscendGraphBinding:
    def __init__(self, metadata, runtime):
        self.metadata, self.runtime = metadata, runtime
        self.update_stream = runtime.create_stream()

    def wait_for_update(self):
        self.update_stream.synchronize()

    def before_replay(self):
        pass

    def after_replay(self):
        import torch_npu

        api = self.runtime.api
        with self.runtime.use_stream(self.update_stream):
            for handle, event, kwargs, workspace, output in self.metadata.tasks:
                api.graph_task_update_begin(self.update_stream, handle)
                torch_npu.npu_fused_infer_attention_score.out(**kwargs, workspace=workspace, out=output)
                api.graph_task_update_end(self.update_stream)
                event.record(self.update_stream)

    def close(self):
        self.wait_for_update()
        self.runtime.synchronize()
        self.metadata.tasks.clear()


class AscendAttentionBackend:
    supports_auxiliary_states = False
    implementation = "ascend"

    def __init__(self, phase, *, workspace_bytes=128 * 1024 * 1024):
        self.phase, self.name = phase, f"ascend.{phase}"
        if workspace_bytes <= 0:
            raise ValueError("Ascend FIA workspace reservation must be positive")
        self.workspace_bytes = workspace_bytes

    def probe(self, spec, fmt, caps, limits, *, auxiliary_formats=None):
        if not isinstance(spec, DenseAttentionSpec) or caps.platform != "ascend" or caps.device_type != "npu":
            return ProbeResult(reason="requires Dense attention on Ascend NPU")
        if self.phase not in ("prefill", "decode") or limits.dynamic_verify or limits.max_verify_tokens != 1:
            return ProbeResult(reason="Ascend Dense does not yet support speculative verification")
        if limits.microbatch_count != 1 or limits.expert_parallel or limits.transfer_paths:
            return ProbeResult(reason="Ascend Dense requires local state without EP or microbatch overlap")
        if any(phase != "decode" for phase in limits.graph_phases):
            return ProbeResult(reason="Ascend supports decode Graph only")
        if (
            fmt.encoding != "plain"
            or fmt.layout != "token_head_dim"
            or fmt.fields != ("k", "v")
            or fmt.version != 1
            or fmt.dtype not in (torch.float16, torch.bfloat16)
            or spec.dtype not in (None, fmt.dtype)
        ):
            return ProbeResult(reason="Ascend Dense requires plain FP16/BF16 K/V")
        if fmt.page_size % 32 or not 32 <= fmt.page_size <= 512 or spec.head_dim not in (64, 128, 256):
            return ProbeResult(reason="Ascend requires page size divisible by 32 in [32,512] and head dim 64/128/256")
        if not spec.causal or spec.window != (-1, -1) or spec.softcap or spec.attention_sink:
            return ProbeResult(reason="Ascend Dense requires causal full-history attention without softcap or sinks")
        # FIA exposes its exact workspace size only after concrete tensors are available.
        # Reserve a bound before allocation; warmup checks the vendor requirement against it.
        size = self.workspace_bytes if self.phase == "decode" and "decode" in limits.graph_phases else 0
        return ProbeResult(
            offers=(
                AttentionOffer(
                    self.name,
                    self.phase,
                    fmt,
                    size,
                    workspace_group="ascend_fia",
                    execution_limits=limits,
                ),
            )
        )

    def bind(self, offer, state_view, static_params, workspace, *, auxiliary_views=None):
        if offer.backend != self.name or offer.format != state_view.format:
            raise ValueError("Ascend binding does not match the selected offer")
        k, v = state_view.fields
        spec = static_params.spec
        if (
            k.shape != v.shape
            or k.shape[1:] != (spec.kv_heads, spec.head_dim)
            or k.dtype != offer.format.dtype
            or v.dtype != k.dtype
            or k.device != v.device
            or k.device.type != "npu"
            or not k.is_contiguous()
            or not v.is_contiguous()
            or k.shape[0] % offer.format.page_size
        ):
            raise ValueError("Ascend attention requires contiguous native K/V with complete pages")
        if offer.workspace_bytes and (
            workspace is None
            or workspace.numel() < offer.workspace_bytes
            or workspace.dtype != torch.uint8
            or workspace.device != k.device
        ):
            raise ValueError("Ascend attention workspace is smaller than the selected offer")
        return BoundAscendAttention(offer, state_view, static_params, workspace)


class BoundAscendAttention:
    transfer_requires_compute_sync = False

    def __init__(self, offer, view, params, workspace):
        self.name, self.phase = offer.backend, offer.phase
        self.spec, self.limits = params.spec, offer.execution_limits
        self.page_size, self.workspace = view.format.page_size, workspace
        self.k, self.v = [t.view(-1, self.page_size, self.spec.kv_heads * self.spec.head_dim) for t in view.fields]
        self.workspace_sizes = {}
        self.mask = None
        if self.phase == "prefill":
            self.mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8, device=self.k.device), diagonal=1)

    @property
    def metadata_key(self):
        return self.name

    def create_metadata(self, batch):
        if batch.batch_size > self.limits.max_batch_size or batch.max_kv_len > self.limits.max_seq_len:
            raise ValueError("batch exceeds the planned Ascend attention capacity")
        if batch.host_ready is not None:
            batch.host_ready.synchronize()
        values = {f.name: getattr(batch, f.name) for f in fields(AttentionBatch)}
        values["host_cu_q"] = (batch.host_cu_q if batch.host_cu_q is not None else batch.cu_q.cpu()).clone()
        values["host_seq_lengths"] = (
            batch.host_seq_lengths if batch.host_seq_lengths is not None else batch.seq_lengths.cpu()
        ).clone()
        # Keep the full table width fixed across replay. Each entry names a physical page.
        tokens = batch.req_to_tokens.index_select(0, batch.req_indices.long())
        values["page_table"] = (
            (tokens[:, : self.limits.max_seq_len : self.page_size] // self.page_size).int().contiguous()
        )
        return AscendMetadata(**values)

    def bind_graph(self, metadata, runtime):
        return AscendGraphBinding(metadata, runtime)

    def run(self, inputs, metadata, out=None):
        import torch_npu

        q = query_tensor(inputs)
        native_q = q if self.phase == "prefill" else q.unsqueeze(2).contiguous()
        kwargs = dict(
            query=native_q,
            key=self.k,
            value=self.v,
            input_layout="TND" if self.phase == "prefill" else "BNSD",
            sparse_mode=3 if self.phase == "prefill" else 0,
            atten_mask=self.mask,
            scale=self.spec.scale,
            next_tokens=0,
            num_heads=self.spec.q_heads,
            num_key_value_heads=self.spec.kv_heads,
            actual_seq_lengths=metadata.host_cu_q[1:]
            if self.phase == "prefill"
            else torch.ones_like(metadata.host_seq_lengths),
            actual_seq_lengths_kv=metadata.host_seq_lengths,
            block_table=metadata.page_table,
            block_size=self.page_size,
        )
        capturing = torch.npu.is_current_stream_capturing()
        if self.phase == "prefill" or self.workspace is None:
            if capturing:
                raise ValueError("Ascend Graph capture was not included in the execution plan")
            result = torch_npu.npu_fused_infer_attention_score(**kwargs)[0]
        else:
            if metadata.batch_size not in self.workspace_sizes:
                if capturing:
                    raise RuntimeError("Ascend FIA workspace must be prepared before capture")
                reserve = dict(
                    kwargs, actual_seq_lengths_kv=torch.full_like(metadata.host_seq_lengths, self.limits.max_seq_len)
                )
                native_workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**reserve)
                size = native_workspace.numel() * native_workspace.element_size()
                if size > self.workspace.numel():
                    raise ValueError(
                        f"Ascend FIA requires {size} bytes, exceeding the planned {self.workspace.numel()}"
                    )
                self.workspace_sizes[metadata.batch_size] = size
            result = torch.empty_like(native_q)
            lse = torch.empty(1, dtype=torch.float16, device=q.device)
            workspace = self.workspace[: self.workspace_sizes[metadata.batch_size]]
            if capturing:
                stream, event = torch.npu.current_stream(), torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(**kwargs, workspace=workspace, out=[result, lse])
            if capturing:
                handle = torch.npu.graph_task_group_end(stream)
                weak = torch_npu._C._weak_ref_tensor
                borrowed = {
                    key: weak(value) if isinstance(value, torch.Tensor) and value.device.type == "npu" else value
                    for key, value in kwargs.items()
                }
                metadata.tasks.append((handle, event, borrowed, workspace, [weak(result), weak(lse)]))
        result = result if self.phase == "prefill" else result.squeeze(2)
        if out is not None:
            out.copy_(result)
            return out
        return result
