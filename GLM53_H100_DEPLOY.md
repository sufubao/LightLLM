# GLM-5.3-Flash H100/H200 deployment and validation

This branch packages LightLLM text and vision inference for GLM-5.3-Flash on
one eight-GPU H100 or H200 node. The release profile keeps the 1,048,576-token
request limit and uses pure TP8, CUDA graphs through batch 256, a 16,384-token
batch budget, fused shared experts, and no dynamic prompt cache.

## Release image

The image is published only to the requested private registry:

```text
registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm:v1.3.0-glm53-vl-1m-tp8-c256-62b8a9da25f7519822657c8126017fbc2793a08a
```

Immutable manifest:

```text
registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm@sha256:3d07b3e9964cae15001e8136f1d19bd0b655df58488546e32f1dd15ffc9dbab7
```

The corresponding local image is `lightllm-glm53:vl-1m-tp8-c256` (image ID
`sha256:34939d00288e2e52ccecd3a241185648e2929a2a6f48bec1173592d337969dff`).
The image embeds source revision
`62b8a9da25f7519822657c8126017fbc2793a08a` and the complete server command.

## Deploy on the H100 node

No command override or autotune-config mount is required:

```bash
IMAGE="registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm@sha256:3d07b3e9964cae15001e8136f1d19bd0b655df58488546e32f1dd15ffc9dbab7"

sudo docker pull "$IMAGE"
sudo docker run -d \
  --name glm53-lightllm-vl-1m-tp8-c256 \
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
sudo docker ps --filter name=glm53-lightllm-vl-1m-tp8-c256
curl --fail --show-error http://127.0.0.1:8002/v1/models
```

## Run the local image on H200

```bash
LIGHTLLM_GLM53_IMAGE=lightllm-glm53:vl-1m-tp8-c256 \
LIGHTLLM_GLM53_MODEL_DIR=/nvme/sufubao/models/GLM-5.3-Flash \
LIGHTLLM_GLM53_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-lightllm-h200 \
LIGHTLLM_GLM53_TRITON_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-triton-h200 \
LIGHTLLM_GLM53_DEEP_GEMM_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-deep-gemm-h200 \
tools/run_glm53_h100_container.sh
```

## Same-host concurrency comparison

LightLLM, vLLM, and SGLang ran sequentially on the same otherwise-idle 8xH100
80 GB node with the same local FP8 checkpoint, TP8, BF16 KV cache, declared
1,048,576-token context, no speculative decoding, and prompt/prefix caching
disabled. The unchanged SGLang `bench_serving` client used seed 42,
temperature 0, streaming, ignored EOS, infinite request rate, one excluded
warmup request, and random 1--1,000-token inputs and outputs.

The request counts for c1/c8/c16/c64/c128/c256 were
10/80/160/640/1,000/1,000. Every request completed. Each engine cell is
`output tok/s / total tok/s`; the percentage is LightLLM's lead over the
faster competing engine.

| Concurrency | LightLLM | SGLang | vLLM | Lead |
| ---: | ---: | ---: | ---: | ---: |
| 1 | **106.03 / 200.86** | 45.75 / 86.66 | 72.91 / 138.11 | **+45.44%** |
| 8 | **556.56 / 1,097.02** | 300.77 / 592.83 | 483.78 / 953.56 | **+15.04%** |
| 16 | **917.99 / 1,734.29** | 850.45 / 1,606.69 | 812.67 / 1,535.31 | **+7.94%** |
| 64 | **2,296.01 / 4,696.03** | 1,680.11 / 3,436.32 | 1,943.30 / 3,974.64 | **+18.15%** |
| 128 | **3,615.77 / 7,304.76** | 2,298.01 / 4,642.55 | 2,980.61 / 6,021.57 | **+21.31%** |
| 256 | **5,032.38 / 10,166.66** | 2,939.85 / 5,939.23 | 4,109.91 / 8,303.03 | **+22.45%** |

The lowest-margin c16 point was repeated at 1,757.52 total tok/s. Result files
are under
`/nvme/sufubao/m39-home/results/glm53_h100_optimization_round3/`; the original
SGLang and vLLM comparison is in
`/nvme/sufubao/m39-home/results/glm53_h100_engine_concurrency_sweep/`.

## Accuracy and capability checks

Every evaluation was recorded with `exp`.

| Check | Result | Experiment |
| --- | --- | --- |
| GSM8K, fixed first 100, 5-shot, greedy | **99/100**, 100/100 completed, no 2,048-token truncation | `260830-162659-bin-bash-lc-export-OPENAI-API-KEY-EMPTY-PYTHONPA` |
| MMMU, fixed 100, full multimodal, greedy | **64/100**, 100/100 completed; 36 answers reached the 2,048-token cap | `260830-163016-bin-bash-lc-export-OPENAI-API-KEY-EMPTY-HF-DATAS` |
| Exact 1M-context needle | API and tokenizer both counted 1,000,000 prompt tokens; recovered `ZEBRA-4821` | `260829-211036` |
| Synthetic vision smoke | Correctly identified the red square | `260829-210908` |

## Rebuild note

`docker/Dockerfile.glm53-h100` is the complete reproducible build. The release
was produced with `docker/Dockerfile.glm53-h100-overlay` from the previous
immutable private release because its runtime layers were already available.
The overlay replaces the complete LightLLM source and pins the source revision
and base digest in OCI labels.
