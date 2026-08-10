# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Working constitution (LightLLM only)

These rules apply only to work in the LightLLM repository. Do not generalize them into global instructions or apply them to other repositories.

### The user must fully understand the work

Treat the user's understanding as a delivery requirement, not an optional explanation. Before implementation, establish and explain the observed behavior, relevant execution path, evidence, root cause or current uncertainty, and the proposed mechanism of change. Before handoff, explain what changed, why it fixes the problem, how it was verified, and any remaining uncertainty or tradeoff. Do not cross a material decision gate while the user indicates that they do not yet understand what is happening.

### Implementation must follow Ponytail

Before writing implementation code, apply the installed `ponytail` skill in its default full mode when it is available. If the skill is unavailable, follow its core policy directly: understand and trace the real flow first, then choose the smallest solution that actually works. Question unnecessary work (YAGNI), reuse existing project code before adding helpers, prefer the standard library or existing dependencies, fix the root cause at the shared path, minimize files and diff size, and avoid speculative abstractions, scaffolding, boilerplate, and cleverness. Do not add test code to a PR unless the user explicitly requests it; verify with existing checks instead. Never simplify away explicit requirements, trust-boundary validation, security, or error handling that prevents data loss.

## Development commands

### Install

The documented development environment uses Python 3.10 (the package requires Python >=3.9.16):

```bash
conda create -n lightllm python=3.10 -y
conda activate lightllm
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python setup.py install
```

For an editable checkout, the Docker build uses:

```bash
pip install -e . --no-cache-dir
```

Build the project image with:

```bash
docker build -t lightllm-dev -f docker/Dockerfile .
```

### Run the server

```bash
python -m lightllm.server.api_server --model_dir /path/to/model
```

The default HTTP endpoint is port 8000. A minimal request is:

```bash
curl http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is AI?","parameters":{"max_new_tokens":17}}'
```

Most runtime behavior is selected through CLI flags defined in `lightllm/server/api_cli.py` and normalized into `StartArgs` in `lightllm/server/core/objs/start_args_type.py`.

### Lint and format

The pre-commit configuration is authoritative: Black 21.12b0 and flake8 6.1.0, both with a 120-column limit.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

To check only files changed from a base revision:

```bash
pre-commit run --files $(git diff --name-only <base> HEAD)
```

Do not substitute a different system Black version; formatting can differ from the pinned hook. `format.py` is a legacy all-files autopep8 script, not the CI formatter.

### Tests

Pytest is not declared in `requirements.txt`; install it separately if needed.

```bash
python -m pip install pytest
python -m pytest unit_tests -q
python -m pytest unit_tests/server/test_hypercorn_config.py -q
python -m pytest unit_tests/server/test_hypercorn_config.py::test_hypercorn_config_is_parsed -q
```

There is no repository-wide pytest configuration or canonical full-suite CI command. Many tests exercise CUDA kernels or distributed/model-specific paths and require suitable GPUs, libraries, and model artifacts; prefer the narrowest relevant test first.

Every benchmark or performance experiment, including commands under `test/benchmark/` or `test/performance/`, must be recorded through the experiment ledger:

```bash
exp -m "short purpose" <benchmark command...>
```

### Documentation

```bash
python -m pip install -r docs/EN/requirements-docs.txt
make -C docs/EN html
```

The Chinese documentation has the analogous `docs/CN/Makefile`.

## Architecture

### Process topology and serving modes

`lightllm/server/api_server.py` is the main entrypoint. It parses CLI arguments and dispatches by `run_mode`: normal serving, prefill/decode workers, PD master, config server, or visual-only. `lightllm/server/api_start.py` validates and derives runtime settings, then uses multiprocessing `spawn` to assemble the HTTP API, router, detokenizer, metrics/cache, and optional vision/audio processes. Shared-memory objects, ZeroMQ channels, and NCCL groups connect these components.

The router is the boundary between request scheduling and GPU execution. `lightllm/server/router/manager.py` starts one model-inference subprocess per local global rank and communicates with each through Unix-socket RPyC. All ranks receive the same initialization arguments and must agree on profiled KV-cache capacity.

### Backend and inference loop

`lightllm/server/router/model_infer/model_rpc.py` selects a backend according to serving mode and features. Normal execution defaults to the chunked-prefill backend; PD, DP, reward, diverse, token-healing, and constrained-output modes select specialized backends under `lightllm/server/router/model_infer/mode_backend/`.

`ModeBackend` initializes distributed groups, model configuration, the model instance, request/KV managers, and the prompt cache, then runs the scheduling/inference loops. The chunked-prefill implementation overlaps GPU forward/sampling, CPU state updates, post-processing, and scheduling for the next batch. Requests enter through shared memory on the node master and are broadcast to participating ranks.

`ModelInput` and `ModelOutput` in `lightllm/server/router/model_infer/mode_backend/batch_objs.py` form the CPU/GPU batch boundary. Preprocessing in `generic_pre_process.py` performs prefix-cache lookup, request/KV allocation, and construction of prefill or decode inputs.

### Model composition and execution

Model implementations register themselves by Hugging Face `model_type` through decorators in `lightllm/models/registry.py`. Importing `lightllm.models` populates this registry. Model-specific classes are generally composition roots that select weight, layer-inference, and inference-state implementations before delegating lifecycle work to `TpPartBaseModel` in `lightllm/common/basemodel/basemodel.py`.

`TpPartBaseModel` owns the ordered initialization pipeline: config repair and validation, quantization setup, weight objects, request/KV managers, inference layers, Hugging Face weight loading, attention backend setup, autotuning, and CUDA-graph capture. Forward execution dispatches prefill versus decode; eligible decode batches use captured CUDA graphs. Per-call sequence, cache, distributed, graph, and microbatch metadata lives in `InferStateInfo`.

Weights are loaded from Hugging Face checkpoints, preferring safetensors, and tensor-parallel slices are applied during loading. Model-family code under `lightllm/models/` supplies architecture-specific weight mappings and kernels while common lifecycle and scheduling code remains shared.

### Token-level KV and request management

The core memory model is token-paged KV storage rather than per-request contiguous buffers. `MemoryManager` in `lightllm/common/kv_cache_mem_manager/` profiles rank-consistent capacity, owns the GPU KV tensor, and delegates free-slot bookkeeping to a pinned-CPU allocator. Specialized managers support normal, quantized, MLA/DSA, and other model-specific KV layouts.

`ReqManager` separately allocates compact request IDs and maintains the GPU request-to-token-index table. Batch preprocessing allocates physical KV slots and writes those mappings before model forward. Finishing or pausing a request releases unshared slots; reusable prefixes may first be inserted into the dynamic prompt cache.

`lightllm/server/router/dynamic_prompt/radix_cache.py` implements the ref-counted radix tree used for prefix reuse. It maps token segments to physical KV-slot segments and only evicts unreferenced leaves, returning their slots to `MemoryManager`.

### Parallelism and communication

The global world is divided into `dp` replicas; ranks within each replica form the tensor-parallel group. Rank topology and device setup are centralized in `lightllm/utils/dist_utils.py`. TP weight slicing and layer collectives live under `lightllm/common/basemodel/layer_weights/` and `lightllm/common/basemodel/layer_infer/`.

`lightllm/distributed/communication_op.py` centralizes communication-group creation and collective dispatch. All-reduce prefers optimized implementations when enabled and falls back to NCCL. Optional groups support TP+SP overlap, cross-DP prefill balancing, and DeepEP expert parallelism for MoE models.

### Where to make changes

- API flags, launch modes, and process wiring: `lightllm/server/api_cli.py`, `api_server.py`, `api_start.py`.
- Scheduling, request lifecycle, and batch construction: `lightllm/server/router/` and `mode_backend/`.
- Shared model lifecycle, memory, and distributed execution: `lightllm/common/` and `lightllm/distributed/`.
- Architecture-specific weights and kernels: `lightllm/models/<model_family>/`.
- API behavior and protocol compatibility: `lightllm/server/httpserver/`.
- Focused unit tests: `unit_tests/`; benchmark and performance harnesses: `test/benchmark/` and `test/performance/`.
