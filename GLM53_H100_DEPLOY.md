# GLM-5.3-Flash H100/H200 deployment and validation

This branch packages LightLLM text and vision inference for GLM-5.3-Flash on
one eight-GPU H100 or H200 node. The release profile keeps the 1,048,576-token
request limit and multimodal workers while optimizing 100-way concurrency with
EP8, TP/SP mixing, prefill microbatch overlap, and CUDA graphs.

## Release image

The image is published only to the requested private registry:

```text
registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm:v1.3.0-glm53-vl-1m-c100-ep8-cc6319a4350c70e062218d85a1cbf1e8d6890cd5
```

Immutable manifest:

```text
registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm@sha256:f456207d3869996c9cfaa17df73058296e8de3720ef3d7eda4d56abb4719ff14
```

The corresponding local image is
`lightllm-glm53:vl-1m-c100-ep8` (image ID
`sha256:f5b1c4282af30f656275d9726637e8d86e7a2dd7b5a103069fb81c6bea92831d`).
The image embeds source revision `cc6319a4350c70e062218d85a1cbf1e8d6890cd5`,
the validated H100 autotune records, and the complete server command.

## Deploy on the H100 node

No command override or autotune-config mount is required:

```bash
IMAGE="registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm@sha256:f456207d3869996c9cfaa17df73058296e8de3720ef3d7eda4d56abb4719ff14"

sudo docker pull "$IMAGE"
sudo docker run -d \
  --name glm53-lightllm-vl-1m-c100-ep8 \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --security-opt label=disable \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit nofile=1048576:1048576 \
  -v /home/devsft/cache-glm53-triton:/root/.triton \
  -v /home/devsft/cache-glm53-deep-gemm:/root/.deep_gemm \
  -v /home/devsft/models/GLM-5.3-Flash:/model:ro \
  -v /home/devsft/cache-glm53-lightllm:/root/.cache \
  "$IMAGE"
```

The endpoint is `http://127.0.0.1:8002/v1`. Startup takes several minutes on
the tested host. Check it with:

```bash
sudo docker ps --filter name=glm53-lightllm-vl-1m-c100-ep8
curl --fail --show-error http://127.0.0.1:8002/v1/models
```

## Run the local image on H200

```bash
LIGHTLLM_GLM53_IMAGE=lightllm-glm53:vl-1m-c100-ep8 \
LIGHTLLM_GLM53_MODEL_DIR=/nvme/sufubao/models/GLM-5.3-Flash \
LIGHTLLM_GLM53_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-lightllm-h200 \
LIGHTLLM_GLM53_TRITON_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-triton-h200 \
LIGHTLLM_GLM53_DEEP_GEMM_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-deep-gemm-h200 \
tools/run_glm53_h100_container.sh
```

## Same-host c100 comparison

LightLLM, vLLM, and SGLang were run sequentially on the same otherwise-idle
8xH100 80 GB host on 2026-08-30. All used the same local FP8 checkpoint, TP8,
BF16 KV cache, declared 1,048,576-token context, and no speculative decoding.
The common SGLang `bench_serving` client used seed 42, temperature 0,
streaming, ignored EOS, infinite request rate, one excluded warmup request,
and random input/output lengths of 1--1,000 tokens. The fixed 1,000-request
sample contained 504,929 input and 494,908 generated tokens. Every engine
completed 1,000/1,000 requests.

| Engine | Duration | Request/s | Input tok/s | Output tok/s | Total tok/s | LightLLM lead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **LightLLM** | **233.37 s** | **4.285** | **2,163.67** | **2,120.73** | **4,284.40** | -- |
| vLLM | 270.74 s | 3.694 | 1,864.99 | 1,827.97 | 3,692.96 | **+16.02%** |
| SGLang | 328.59 s | 3.043 | 1,536.66 | 1,506.16 | 3,042.82 | **+40.80%** |

The compared server builds and material settings were:

- LightLLM: this release profile; EP8, TP/SP mixing, prefill microbatch
  overlap, 1,024-token chunks, graph batches through 104, and FlashInfer
  all-reduce disabled.
- vLLM: `vllm/vllm-openai:glm53-flash-x86_64-cu130@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703`
  (`0.1.dev20051+g487ecf187`); TP8, 256 sequences, chunked prefill, prefix
  caching, and `gpu-memory-utilization=0.90`.
- SGLang: `lmsysorg/sglang@sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf`
  (`0.0.0.dev1+gf609d677b`); TP8/EP8, DeepGEMM,
  `mem-fraction-static=0.80`, and 1,024-token chunks.

Cross-engine TTFT/ITL and client-retokenized text are intentionally omitted:
the engines expose reasoning text under different streaming fields. The API
usage token totals, successful-request count, wall time, and aggregate
throughput above are directly comparable.

Result files:

- LightLLM: `/nvme/sufubao/m39-home/results/glm53_h100_optimization/lightllm_ep8_tpsp_prefill_overlap_graph_c104_b4096_wait64_tuned_random_1k_c100_full1000.jsonl`
- vLLM: `/nvme/sufubao/m39-home/results/glm53_h100_engine_compare/vllm_h100_random_1k_c100.jsonl`
- SGLang: `/nvme/sufubao/m39-home/results/glm53_h100_engine_compare/sglang_h100_random_1k_c100.jsonl`

The LightLLM run is archived by `exp` under
`~/experiments/runs/260830-025933-usr-bin-timeout-2400-docker-run-rm-network-host-`.

## Accuracy and capability checks

The final optimized execution path was evaluated before publication. Every
evaluation was recorded with `exp`.

| Check | Result | Experiment |
| --- | --- | --- |
| GSM8K, fixed first 100, 5-shot, greedy | **99/100**, 100/100 completed, no 2,048-token truncation | `260830-030435-bin-bash-lc-export-OPENAI-API-KEY-EMPTY-PYTHONPA` |
| MMMU, fixed 100, full multimodal, greedy | **63/100**, 100/100 completed; 34 answers reached the 2,048-token cap | `260830-030533-bin-bash-lc-export-OPENAI-API-KEY-EMPTY-PYTHONPA` |
| Exact 1M-context needle | API and tokenizer both counted 1,000,000 prompt tokens; recovered `ZEBRA-4821` | `260829-211036` |
| Text smoke | Arithmetic answer `579` | `260829-210839` |
| Synthetic vision smoke | Correctly identified the red square | `260829-210908` |

The previous conservative profile scored 64/100 on the same MMMU slice; the
optimized profile's one-answer difference is within this 100-example sample,
while it reduced capped answers from 39 to 34. The multimodal and 1M-context
features remain enabled in the published command.

References: [SGLang GLM-5 benchmark recipe](https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/GLM/GLM-5.mdx),
[SGLang serving benchmark](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/base/benchmarks/autoregressive_model_benchmark.mdx),
[vLLM GLM-5.3-Flash recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash),
and [vLLM GLM-5.3 support PR](https://github.com/vllm-project/vllm/pull/53906).

## Rebuild note

`docker/Dockerfile.glm53-h100` is the complete reproducible build. The release
was produced with `docker/Dockerfile.glm53-h100-overlay` from the previous
immutable, flattened private image because Docker Hub base-metadata requests
timed out during release. The overlay replaces the complete LightLLM source,
checks the three new autotune records and import path, and pins both its source
revision and base digest in OCI labels.
