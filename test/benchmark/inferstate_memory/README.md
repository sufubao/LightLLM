# InferState ownership experiments

The production fix makes attention states borrow `InferStateInfo` through a weak
reference. These experiments restore just that edge to a strong reference for
an A/B comparison. Model weights, inputs and kernels stay the same.

Run every experiment through `exp -m 'purpose' COMMAND`. Choose idle GPUs and
unused ports; the commands below were run on m35, without using m34.
Dependencies are the normal LightLLM runtime plus `aiohttp`, `psutil` and
`matplotlib`. No image build is required.

## Actual HTTP requests

Prepare an isolated source snapshot first. This adds the opt-in worker probe
import; running the driver in an unprepared checkout will not install it.

```bash
python test/benchmark/inferstate_memory/prepare_http.py "$PWD" /tmp/http-repro
cd /tmp/http-repro
export CUDA_HOME=/usr/local/cuda-12.8
CUDA_VISIBLE_DEVICES=2 exp -m 'Qwen3.5 HTTP ownership, Graph OFF' \
    python test/benchmark/inferstate_memory/http_stress.py \
    --graph off --port 21761 --output results_off
CUDA_VISIBLE_DEVICES=3 exp -m 'Qwen3.5 HTTP ownership, Graph ON' \
    python test/benchmark/inferstate_memory/http_stress.py \
    --graph on --port 23761 --output results_on
```

The default model is `/mtc/models/Qwen3.5-27B`; override `--model-dir` as needed.
Each command starts fresh services for weak and strong ownership. The service
uses BF16, TP1, 256 request slots, 65536 KV tokens, Triton full/linear attention,
prompt logprobs, dynamic prefix caching and chunked prefill. Each variant runs:

- Two warmup requests, excluded from pressure counts.
- Decode-heavy: 32 requests, concurrency 8, 32 input and 256 output tokens.
- Prefill-heavy: 64 serial requests, 4096 input and one output token, requesting
  prompt logprobs. Observed scheduling splits the input into 3584 and 512 tokens.

The worker probe does not force GC or synchronize CUDA in the request path.
It records active/reserved CUDA memory, per-forward peaks, GC events and live
InferState weak references. Graph-owned states are counted separately. Probe
bookkeeping can affect Python GC timing, so results are specific to this setup.

For passive physical GPU sampling, start the following before the load (the
output directory must exist), then create `OUTPUT_ROOT/stop_nvml` afterward:

```bash
exp -m 'Passive NVML sampling on GPUs 2 and 3' \
    python test/benchmark/inferstate_memory/profile_nvml.py OUTPUT_ROOT
```

Copy the four variant folders into one result directory, optionally add
`nvml.csv`, and render the report without rerunning the model:

```bash
python test/benchmark/inferstate_memory/plot_http.py RESULT_DIRECTORY
```

The output includes `summary.json`, a self-contained `report.html`, and PNG/PDF
plots. Active tensor memory and sampled NVML usage are different measures;
NVML also includes allocator caches and CUDA context allocations.

### Observed HTTP results (m35 H200, Qwen3.5-27B)

| Decode Graph | Uncached Decode states, strong / weak | Long-input active CUDA peak, strong / weak (GiB) | Sampled NVML peak, strong / weak (GiB) |
| --- | --- | --- | --- |
| Off | 12 / 0 | 107.72 / 96.34 | 119.67 / 102.59 |
| On | 7 / 0 | 111.07 / 96.38 | 123.05 / 102.65 |

All 384 pressure requests succeeded; no HTTP OOM occurred. The four Decode-heavy
phases generated 32768 output tokens and executed 1024 Decode forwards each.
Graph ON held eight legitimate cached states in both ownership variants;
these are excluded from the uncached counts. Decode-only active tensor peaks
were close: 91.590 / 91.559 GiB with Graph OFF and 91.609 / 91.609 GiB with
Graph ON. The NVML difference cannot all be attributed to live Decode tensors.

## Direct-forward and controlled allocations

```bash
CUDA_VISIBLE_DEVICES=2 bash test/benchmark/inferstate_memory/reproduce.sh \
    /tmp/inferstate-forward /mtc/models/Qwen3.5-27B
```

The wrapper uses `exp`, then renders `run.py` results with `plot.py`. This is a
different workload from HTTP: it calls model forward directly, disables Graph
and clears GC/cache between suite phases. For Qwen3.5 it resets hybrid
attention state for each simulated new request.

In the uncapped prompt-logprobs phase, strong ownership completed 43 iterations
and hit CUDA OOM on attempt 44 (peak 134.99 GiB); weak ownership completed 150
(peak 93.26 GiB). Plain prefill peaks were close (88.85 / 88.83 GiB).
The equal 95 GiB allocator-budget comparison also OOMed only with strong
ownership. These iteration counts are not HTTP service request thresholds.

Use `run.py --help` for controlled allocations and explicit GC/budget settings.
Raw JSON includes configuration, hardware, source hashes, memory samples and GC
events. Always retain these records when comparing runs.
