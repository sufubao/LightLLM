"""Best-effort D -> P checkpoint export, outside the inference critical path.

The existing NIXL/NCCL mover is a unidirectional GPU P -> D pipeline. This
transport moves already-frozen CPU cache pages directly between the two nodes;
PD Master carries only the owner address. Missing base pages are negotiated
before transfer, and receiving threads only queue imports. The inference
coordinator must prepare/commit each import with TP consensus.
"""

import atexit
import hashlib
import io
import json
import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import torch

from lightllm.server.router.dynamic_prompt.checkpoint_cache import CpuCheckpointCache
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_hostname_ip

logger = init_logger(__name__)
_PROTOCOL_VERSION = 2


def checkpoint_registry_token():
    # The service ID is public in get_server_info's IPC address. Keep discovery
    # credentials independent, while sharing one secret across HTTP workers.
    directory = _registry_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / ".registry_token"
    if not path.exists():
        temporary = directory / f".registry_token.{os.getpid()}.{secrets.token_hex(8)}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w") as output:
                output.write(secrets.token_urlsafe(32))
            try:
                # Publish a complete file without replacing another worker's
                # secret; O_EXCL creation above and link are both atomic.
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    token = path.read_text()
    if len(token) < 32:
        raise ValueError("invalid internal checkpoint registry credential")
    return token


def _registry_dir():
    name = hashlib.sha256(get_unique_server_name().encode()).hexdigest()[:24]
    return Path("/dev/shm") / f"lightllm-checkpoint-{name}"


def read_checkpoint_registry():
    """Called by the local HTTP process; entries contain only CPU transport metadata."""
    entries = []
    for path in _registry_dir().glob("*.json"):
        try:
            entry = json.loads(path.read_text())
            os.kill(entry["pid"], 0)
            entries.append(entry)
        except (OSError, ValueError, KeyError):
            continue
    return sorted(entries, key=lambda x: (x["dp_index"], x["tp_rank"]))


def _save_payload(payload):
    output = io.BytesIO()
    torch.save(payload, output)
    return output.getvalue()


def _load_payload(data):
    # No remote Python class or callable is deserialized.
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def _validate_origins(payload):
    origins = payload.get("origins")
    if not isinstance(origins, (list, tuple)) or len(origins) != payload["length"]:
        raise ValueError("checkpoint KV provenance must cover the token prefix")
    if any(type(origin) is not int or not 0 < origin < (1 << 63) for origin in origins):
        raise ValueError("checkpoint KV provenance must use positive integer IDs")


def _head_slice(rank, world_size, global_heads):
    if global_heads >= world_size:
        if global_heads % world_size:
            raise ValueError("checkpoint heads do not divide TP size")
        width = global_heads // world_size
        return slice(rank * width, (rank + 1) * width)
    if world_size % global_heads:
        raise ValueError("checkpoint replicated heads do not divide TP size")
    head = rank // (world_size // global_heads)
    return slice(head, head + 1)


def _merge_heads(shards, dim, global_heads):
    world_size = len(shards)
    if global_heads >= world_size:
        if global_heads % world_size:
            raise ValueError("checkpoint heads do not divide source TP size")
        return torch.cat(shards, dim=dim)
    if world_size % global_heads:
        raise ValueError("checkpoint replicated heads do not divide source TP size")
    copies = world_size // global_heads
    # Replicated KV/key heads carry the same values. Keep a single copy per head.
    return torch.cat([shards[i * copies] for i in range(global_heads)], dim=dim)


def _check_layouts(source, destination):
    ignored = {"kv_layers", "draft_layer_num"}
    if {k: v for k, v in source.items() if k not in ignored} != {
        k: v for k, v in destination.items() if k not in ignored
    }:
        raise ValueError("incompatible checkpoint model/state layout")
    source_draft, destination_draft = source.get("draft_layer_num", 0), destination.get("draft_layer_num", 0)
    if destination_draft and destination_draft != source_draft:
        raise ValueError("destination requires unavailable draft KV")
    if source["kv_layers"] - source_draft != destination["kv_layers"] - destination_draft:
        raise ValueError("incompatible target KV layer layout")
    return bool(source_draft and not destination_draft)


def reshard_checkpoint_payloads(
    payloads, source_layout, destination_layout, destination_rank, destination_world, destination_namespace=None
):
    """Convert local TP fragments through the existing Q/K/V global-head layout."""
    strip_draft = _check_layouts(source_layout, destination_layout)
    first = payloads[0]
    _validate_origins(first)
    for part in payloads[1:]:
        for key in ("version", "namespace", "page_size", "length", "tokens", "origins", "page_keys"):
            if part[key] != first[key]:
                raise ValueError(f"TP checkpoint manifest mismatch: {key}")
        if set(part["pages"]) != set(first["pages"]):
            raise ValueError("TP checkpoint page coverage mismatch")
        if part.get("draft_tail_dependency", False) != first.get("draft_tail_dependency", False):
            raise ValueError("TP checkpoint draft dependencies mismatch")

    kv_heads = source_layout["kv_heads"]
    k_heads = source_layout["linear_k_heads"]
    v_heads = source_layout["linear_v_heads"]
    k_dim = source_layout["linear_k_dim"]
    v_dim = source_layout["linear_v_dim"]
    source_world = len(payloads)
    result = {
        key: first[key] for key in ("version", "namespace", "page_size", "length", "tokens", "origins", "page_keys")
    }
    result["draft_tail_dependency"] = False if strip_draft else first.get("draft_tail_dependency", False)
    if destination_namespace is not None:
        result["namespace"] = destination_namespace
    if result["namespace"] != first["namespace"] or strip_draft:
        result["page_keys"] = CpuCheckpointCache.derive_page_keys(
            first["tokens"],
            namespace=result["namespace"],
            page_size=first["page_size"],
            draft_tail_dependency=result["draft_tail_dependency"],
            origins=first["origins"],
        )
    renamed = dict(zip(first["page_keys"], result["page_keys"]))
    result["pages"] = {}
    for key in first["pages"]:
        keys, values = [], []
        for part in payloads:
            page = part["pages"][key]
            if strip_draft:
                page = page[: destination_layout["kv_layers"]]
            local_heads = page.shape[2] // 2
            keys.append(page[:, :, :local_heads])
            values.append(page[:, :, local_heads:])
        head_range = _head_slice(destination_rank, destination_world, kv_heads)
        result["pages"][renamed[key]] = torch.cat(
            (_merge_heads(keys, 2, kv_heads)[:, :, head_range], _merge_heads(values, 2, kv_heads)[:, :, head_range]),
            dim=2,
        ).contiguous()

    q_parts, k_parts, v_parts = [], [], []
    for part in payloads:
        conv = part["conv_state"]
        local_k = max(1, k_heads // source_world)
        local_v = max(1, v_heads // source_world)
        q, k, v = conv.split((local_k * k_dim, local_k * k_dim, local_v * v_dim), dim=1)
        q_parts.append(q.reshape(q.shape[0], local_k, k_dim, q.shape[-1]))
        k_parts.append(k.reshape(k.shape[0], local_k, k_dim, k.shape[-1]))
        v_parts.append(v.reshape(v.shape[0], local_v, v_dim, v.shape[-1]))
    k_range = _head_slice(destination_rank, destination_world, k_heads)
    v_range = _head_slice(destination_rank, destination_world, v_heads)
    result["conv_state"] = torch.cat(
        (
            _merge_heads(q_parts, 1, k_heads)[:, k_range].flatten(1, 2),
            _merge_heads(k_parts, 1, k_heads)[:, k_range].flatten(1, 2),
            _merge_heads(v_parts, 1, v_heads)[:, v_range].flatten(1, 2),
        ),
        dim=1,
    ).contiguous()
    result["ssm_state"] = _merge_heads([p["ssm_state"] for p in payloads], 1, v_heads)[:, v_range].clone(
        memory_format=torch.contiguous_format
    )
    # The LM-head input is replicated after target TP reduction. A transport
    # must not interpret a vocabulary-sharded logits tensor as such a seed.
    result["output_seed"] = first["output_seed"]
    return result


@dataclass
class CheckpointImport:
    import_id: str
    payload: dict
    created_at: float


class PDCheckpointTransport:
    """One CPU-only endpoint per inference rank, with bounded background work."""

    def __init__(self, backend, cache, namespace="default", target_namespace=None):
        self.backend = backend
        self.cache = cache
        self.args = backend.args
        self.tp_rank = backend.rank_in_dp
        self.tp_world = backend.dp_world_size
        self.dp_index = backend.dp_rank_in_node
        self.namespace = namespace
        self.target_namespace = namespace if target_namespace is None else target_namespace
        self.timeout = 60.0
        self._lock = threading.Lock()
        self._imports = {}
        self._import_status = {}
        self._exports = {}
        self._jobs = queue.Queue(maxsize=4)
        self._closed = False
        self._generation = 0
        self._auth = secrets.token_urlsafe(32)
        cfg = backend.model.mem_manager.linear_config
        self.layout = {
            "kind": "qwen-linear-v1",
            "kv_heads": cfg.full_att_all_num_kv_heads,
            "kv_head_dim": cfg.full_att_head_dim,
            "kv_layers": cfg.get_full_att_kv_layer_num_with_draft_model(),
            "target_layer_num": cfg.get_main_model_full_att_layer_num(),
            "draft_layer_num": cfg.draft_full_att_kv_layer_num,
            "linear_k_heads": cfg.global_linear_k_heads,
            "linear_v_heads": cfg.global_linear_v_heads,
            "linear_k_dim": cfg.head_linear_k_dim,
            "linear_v_dim": cfg.head_linear_v_dim,
            "linear_layers": cfg.linear_layer_num,
            "conv_width": cfg.conv_kernel_size - 1,
            "kv_dtype": str(cfg.full_att_dtype),
            "conv_dtype": str(cfg.conv_state_dtype),
            "ssm_dtype": str(cfg.ssm_state_dtype),
        }
        host = self.args.host
        if host in ("0.0.0.0", "127.0.0.1", "localhost"):
            host = get_hostname_ip()
        transport = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                with transport._lock:
                    authorized = self.headers.get("Authorization") == f"Bearer {transport._auth}"
                    generation = transport._generation
                if not authorized:
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    # A valid payload contains no more than one cache-sized
                    # checkpoint plus serialization metadata.
                    limit = max(16 << 20, int(getattr(cache, "max_bytes", 1 << 30)) * 2)
                    if not 0 < length <= limit:
                        self.send_error(413)
                        return
                    self.connection.settimeout(transport.timeout)
                    data = self.rfile.read(length)
                    if len(data) != length:
                        raise ValueError("incomplete checkpoint request")
                    value = _load_payload(data)
                    response = transport._handle(self.path, value, generation=generation)
                    body = response if isinstance(response, bytes) else _save_payload(response)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as exc:
                    logger.warning(f"checkpoint transport request rejected: {exc}")
                    self.send_error(409, "checkpoint unavailable")

        self._server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self._server.daemon_threads = True
        self.descriptor = {
            "protocol_version": _PROTOCOL_VERSION,
            "pid": os.getpid(),
            "dp_index": self.dp_index,
            "tp_rank": self.tp_rank,
            "tp_world": self.tp_world,
            "url": f"http://{host}:{self._server.server_port}",
            "auth": self._auth,
            "layout": self.layout,
            "namespace": self.namespace,
            "target_namespace": self.target_namespace,
            "draft_tail_dependency": bool(cache.draft_tail_dependency),
            "page_size": cache.page_size,
        }
        directory = _registry_dir()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._registry_path = directory / f"{backend.rank_in_node}.json"
        self._write_registry()
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        if self.tp_rank == 0:
            threading.Thread(target=self._export_loop, daemon=True).start()
        atexit.register(self.close)

    def _write_registry(self):
        temporary = self._registry_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.descriptor))
        os.replace(temporary, self._registry_path)

    def clear(self):
        """Invalidate queued/in-flight imports without a device or network wait."""
        with self._lock:
            self._generation += 1
            self._auth = secrets.token_urlsafe(32)
            self.descriptor = dict(self.descriptor, auth=self._auth)
            self._imports.clear()
            self._import_status.clear()
            leases = list(self._exports.values())
            self._exports.clear()
        for _, lease in leases:
            lease.close()
        while True:
            try:
                self._jobs.get_nowait()
                self._jobs.task_done()
            except queue.Empty:
                break
        self._write_registry()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._registry_path.unlink(missing_ok=True)
        self._server.shutdown()
        self._server.server_close()
        with self._lock:
            leases = list(self._exports.values())
            self._exports.clear()
        for _, lease in leases:
            lease.close()

    def drain_imports(self):
        """Peek pending imports. Call finish_import after a TP-consistent decision."""
        now = time.monotonic()
        with self._lock:
            expired = [key for key, item in self._imports.items() if now - item.created_at > self.timeout]
            for key in expired:
                self._imports.pop(key)
                self._import_status[key] = (now, "expired")
            return list(self._imports.values())

    def finish_import(self, import_id, success):
        with self._lock:
            self._imports.pop(import_id, None)
            self._import_status[import_id] = (time.monotonic(), "ready" if success else "rejected")

    def publish_checkpoint(self, tokens, namespace, owner_url, export_id, owner_dp_index=0, owner_auth=None):
        """Enqueue only on the DP leader, after local TP checkpoint commit."""
        if self.tp_rank != 0 or not owner_url or self._closed:
            return False
        if isinstance(owner_url, bytes):
            owner_url = owner_url.decode("utf-8")
        if isinstance(owner_auth, bytes):
            owner_auth = owner_auth.decode("utf-8")
        if not owner_auth:
            # A P node without exact caching did not register this capability.
            return False
        parsed = urlsplit(owner_url)
        if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password or parsed.path:
            return False
        try:
            # Request token views contain NumPy scalars; the wire format deliberately
            # permits only built-in values and tensors for weights_only decoding.
            self._jobs.put_nowait(
                (
                    [int(token) for token in tokens],
                    namespace,
                    owner_url,
                    str(export_id),
                    int(owner_dp_index),
                    owner_auth,
                )
            )
            return True
        except queue.Full:
            logger.info("checkpoint export admission skipped: queue full")
            return False

    def _handle(self, path, obj, generation=None):
        if not isinstance(obj, dict):
            raise ValueError("expected checkpoint request object")
        with self._lock:
            if generation is not None and generation != self._generation:
                raise ValueError("checkpoint cache was cleared during transfer")
        self._expire_exports()
        if path == "/pin":
            if obj["namespace"] != self.namespace:
                raise ValueError("checkpoint namespace incompatible")
            lease = self.cache.match(obj["tokens"], max_length=len(obj["tokens"]), namespace=obj["namespace"])
            if lease is None or lease.length != len(obj["tokens"]):
                if lease is not None:
                    lease.close()
                raise ValueError("checkpoint no longer present")
            export_id = secrets.token_hex(16)
            with self._lock:
                if (generation is not None and generation != self._generation) or len(self._exports) >= 8:
                    lease.close()
                    raise ValueError("checkpoint export leases full")
                self._exports[export_id] = (time.monotonic(), lease)
            return {
                "lease_id": export_id,
                "page_keys": lease.page_keys,
                "origins": lease.origins.tolist(),
                "layout": self.layout,
            }
        if path == "/export":
            with self._lock:
                _, lease = self._exports[obj["lease_id"]]
                # Serialize while this lease is protected from release/expiry.
                # No additional full-prefix tensor copy or GPU read is needed.
                payload = self.cache.export(lease, known_page_keys=obj.get("known_page_keys", ()))
                return _save_payload(payload)
        if path == "/release":
            with self._lock:
                entry = self._exports.pop(obj["lease_id"], None)
            if entry is not None:
                entry[1].close()
            return {"ok": True}
        if path == "/missing":
            return {"missing": self.cache.missing_page_keys(obj["page_keys"])}
        if path == "/import":
            import_id = obj["import_id"]
            _validate_origins(obj["payload"])
            if obj["layout"] != self.layout:
                raise ValueError("checkpoint layout mismatch")
            if obj["payload"]["namespace"] != self.namespace:
                raise ValueError("checkpoint namespace mismatch")
            with self._lock:
                if generation is not None and generation != self._generation:
                    raise ValueError("checkpoint cache was cleared during transfer")
                prior = self._import_status.get(import_id)
                if prior is not None:
                    return {"status": prior[1]}
                if import_id not in self._imports:
                    if len(self._imports) >= 4:
                        raise ValueError("checkpoint import queue full")
                    self._imports[import_id] = CheckpointImport(import_id, obj["payload"], time.monotonic())
            return {"status": "pending"}
        if path == "/status":
            with self._lock:
                status = self._import_status.get(obj["import_id"])
                return {"status": status[1] if status is not None else "pending"}
        if path == "/cancel":
            self.finish_import(obj["import_id"], False)
            return {"ok": True}
        raise ValueError("unknown checkpoint operation")

    def _expire_exports(self):
        now = time.monotonic()
        with self._lock:
            expired = [key for key, (created, _) in self._exports.items() if now - created > self.timeout * 2]
            leases = [self._exports.pop(key)[1] for key in expired]
            self._import_status = {key: value for key, value in self._import_status.items() if now - value[0] < 300}
        for lease in leases:
            lease.close()

    @staticmethod
    def _call(client, endpoint, path, payload):
        response = client.post(
            endpoint["url"] + path,
            content=_save_payload(payload),
            headers={"Authorization": f"Bearer {endpoint['auth']}", "Content-Type": "application/octet-stream"},
        )
        response.raise_for_status()
        return _load_payload(response.content)

    def _export_loop(self):
        while not self._closed:
            try:
                job = self._jobs.get(timeout=1)
            except queue.Empty:
                self._expire_exports()
                continue
            try:
                self._export_one(*job)
            except Exception as exc:
                logger.warning(f"checkpoint D -> P export skipped: {exc}")
            finally:
                self._jobs.task_done()

    def _export_one(self, tokens, namespace, owner_url, export_id, owner_dp_index, owner_auth):
        sources = [entry for entry in read_checkpoint_registry() if entry["dp_index"] == self.dp_index]
        if len(sources) != self.tp_world:
            raise ValueError("source TP checkpoint endpoints incomplete")
        pins, destinations, import_ids = [], [], []
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            try:
                response = client.get(
                    owner_url + "/pd_checkpoint/registry",
                    headers={"Authorization": f"Bearer {owner_auth}"} if owner_auth else {},
                )
                response.raise_for_status()
                destinations = [e for e in response.json()["ranks"] if e["dp_index"] == owner_dp_index]
                if not destinations or len(destinations) != destinations[0]["tp_world"]:
                    raise ValueError("destination TP checkpoint endpoints incomplete")
                origins, page_keys = None, None
                for source in sources:
                    if source.get("protocol_version") != _PROTOCOL_VERSION:
                        raise ValueError("source checkpoint protocol incompatible")
                    pin = self._call(client, source, "/pin", {"tokens": tokens, "namespace": namespace})
                    pins.append((source, pin["lease_id"]))
                    if pin["layout"] != self.layout:
                        raise ValueError("source checkpoint layouts differ")
                    if origins is not None and (pin["origins"] != origins or pin["page_keys"] != page_keys):
                        raise ValueError("source TP checkpoint provenance differs")
                    origins = pin["origins"]
                    page_keys = pin["page_keys"]
                target = destinations[0]
                if target["page_size"] != self.cache.page_size:
                    raise ValueError("checkpoint KV page sizes differ")
                strip_draft = _check_layouts(self.layout, target["layout"])
                expected_namespace = self.target_namespace if strip_draft else namespace
                if target["namespace"] != expected_namespace:
                    raise ValueError("destination checkpoint execution identity incompatible")
                destination_page_keys = CpuCheckpointCache.derive_page_keys(
                    tokens,
                    namespace=expected_namespace,
                    page_size=self.cache.page_size,
                    draft_tail_dependency=target["draft_tail_dependency"],
                    origins=origins,
                )
                missing = set()
                for destination in destinations:
                    if destination.get("protocol_version") != _PROTOCOL_VERSION:
                        raise ValueError("destination checkpoint protocol incompatible")
                    if destination["layout"] != target["layout"] or destination["namespace"] != expected_namespace:
                        raise ValueError("destination checkpoint layout incompatible")
                    missing.update(
                        self._call(client, destination, "/missing", {"page_keys": destination_page_keys})["missing"]
                    )
                known = [
                    source_key
                    for source_key, dest_key in zip(page_keys, destination_page_keys)
                    if dest_key not in missing
                ]
                payloads = [
                    self._call(client, source, "/export", {"lease_id": lease_id, "known_page_keys": known})
                    for source, lease_id in pins
                ]
                for destination in destinations:
                    payload = reshard_checkpoint_payloads(
                        payloads,
                        self.layout,
                        destination["layout"],
                        destination["tp_rank"],
                        len(destinations),
                        destination_namespace=expected_namespace,
                    )
                    self._call(
                        client,
                        destination,
                        "/import",
                        {
                            "import_id": export_id,
                            "layout": destination["layout"],
                            "payload": payload,
                        },
                    )
                    import_ids.append((destination, export_id))
                deadline = time.monotonic() + self.timeout
                while time.monotonic() < deadline:
                    states = [
                        self._call(client, dst, "/status", {"import_id": key})["status"] for dst, key in import_ids
                    ]
                    if all(state == "ready" for state in states):
                        logger.info(
                            f"checkpoint D -> P ready length={len(tokens)} pages_sent={len(missing)} "
                            f"pages_reused={len(known)} source_tp={len(sources)} destination_tp={len(destinations)}"
                        )
                        import_ids.clear()
                        return
                    if any(state not in ("pending", "ready") for state in states):
                        raise ValueError(f"checkpoint import rejected: {states}")
                    time.sleep(0.02)
                raise TimeoutError("checkpoint import commit timed out")
            finally:
                for destination, key in import_ids:
                    try:
                        self._call(client, destination, "/cancel", {"import_id": key})
                    except Exception:
                        pass
                for source, lease_id in pins:
                    try:
                        self._call(client, source, "/release", {"lease_id": lease_id})
                    except Exception:
                        pass
