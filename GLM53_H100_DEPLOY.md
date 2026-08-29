# GLM-5.3-Flash multimodal TP8 deployment

This branch packages LightLLM text and vision inference for GLM-5.3-Flash on
one eight-GPU H100 or H200 node. The image contains the LightLLM source and
runtime dependencies. Mount the model and compiler caches from the host.

The default command serves the OpenAI-compatible API on port 8002 with:

- tensor parallel size 8;
- image encoding as eight data-parallel workers;
- a 1,048,576-token request limit;
- up to 256 active requests;
- 8,192-token chunked prefill, which bounds the DSA score matrix during
  million-token requests, with CUDA graphs disabled for this profile;
- `glm45` reasoning and `glm47` tool-call parsers.

## Build a local image

Use an immutable tag containing the full source revision:

```bash
revision="$(git rev-parse HEAD)"
created="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
version="v1.3.0-glm53-vl-1m-tp8-${revision}"

docker buildx build --load --platform linux/amd64 \
  -f docker/Dockerfile.glm53-h100 \
  --build-arg "OCI_CREATED=${created}" \
  --build-arg "OCI_REVISION=${revision}" \
  --build-arg "OCI_VERSION=${version}" \
  -t "lightllm-glm53:${version}" \
  .

docker tag "lightllm-glm53:${version}" lightllm-glm53:vl-1m-tp8
```

## Run on the local H200 node

```bash
LIGHTLLM_GLM53_IMAGE=lightllm-glm53:vl-1m-tp8 \
LIGHTLLM_GLM53_MODEL_DIR=/nvme/sufubao/models/GLM-5.3-Flash \
LIGHTLLM_GLM53_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-lightllm-h200 \
LIGHTLLM_GLM53_TRITON_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-triton-h200 \
LIGHTLLM_GLM53_DEEP_GEMM_CACHE_DIR=/nvme/sufubao/m39-home/cache/glm53-deep-gemm-h200 \
tools/run_glm53_h100_container.sh
```

## Run the published image on H100

This command was validated on one eight-GPU H100 80 GB node. The fixed
container name is `glm53-lightllm-vl-1m`. Keep `batch_max_tokens` at 8,192 and
use a conservative 1,024-token prefill chunk: a 65,536-token batch maximum
OOMed during the server's startup length check because the DSA score matrix
exceeded the H100's temporary-memory headroom. Keep FlashInfer all-reduce
disabled for this profile; without that override, the first inference request
stalled across ranks on the tested host.

```bash
IMAGE="registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm@sha256:2a664580a495215a5bfb48d96bf118a8321d7accde589e505de283d6ea5753b2"

sudo docker pull "$IMAGE"
sudo docker run -d \
  --name glm53-lightllm-vl-1m \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  -v /home/devsft/models/GLM-5.3-Flash:/model:ro \
  -v /home/devsft/cache-glm53-lightllm:/root/.cache \
  -v /home/devsft/cache-glm53-triton:/root/.triton \
  -v /home/devsft/cache-glm53-deep-gemm:/root/.deep_gemm \
  "$IMAGE" \
  /opt/sglang/bin/python -m lightllm.server.api_server \
  --model_dir /model \
  --model_name glm-5.3-flash \
  --tp 8 \
  --host 0.0.0.0 \
  --port 8002 \
  --httpserver_workers 16 \
  --mem_fraction .90 \
  --max_total_token_num 1048612 \
  --running_max_req_size 256 \
  --max_req_total_len 1048576 \
  --batch_max_tokens 8192 \
  --chunked_prefill_size 1024 \
  --linear_att_ssm_data_type bfloat16 \
  --linear_att_cache_size 256 \
  --disable_cudagraph \
  --disable_flashinfer_allreduce \
  --enable_fused_shared_experts \
  --max_image_pixels 6272000 \
  --max_image_token_count 8000 \
  --visual_tp 1 \
  --visual_dp 8 \
  --visual_infer_batch_size 8 \
  --cache_capacity 64 \
  --schedule_time_interval 0.001 \
  --prefill_coalesce_interval 0.5 \
  --reasoning_parser glm45 \
  --tool_call_parser glm47
```

Wait for the model endpoint, then stop the deployment when required:

```bash
curl --fail --show-error http://127.0.0.1:8002/v1/models
sudo docker stop --timeout 30 glm53-lightllm-vl-1m
```

## H100 validation

The published digest and command above were exercised through the
OpenAI-compatible endpoint on one eight-GPU H100 80 GB node on 2026-08-29.
Every inference, evaluation, and benchmark command was recorded with `exp`.

| Check | Result |
| --- | --- |
| Text smoke | `123 + 456` returned `579` in 15.34 s |
| Synthetic vision smoke | Identified the red square in 17.22 s |
| Exact 1M context needle | Both tokenizer and API counted exactly 1,000,000 prompt tokens; recovered `ZEBRA-4821` in 171.49 s |
| Sampled peak during 1M request | Approximately 80,063 MiB of 81,559 MiB per GPU; no OOM |
| SGLang-style latency workload | 10/10 requests; 3,309 input and 3,700 output tokens in 319.65 s; 21.93 total tok/s and 11.58 output tok/s |
| SGLang-style throughput workload | 1,000/1,000 requests; 504,929 input and 494,908 output tokens in 1,962.10 s; 509.58 total tok/s and 252.23 output tok/s |
| GSM8K | 99/100; all completed in 118.17 s and none reached the 2,048-token cap |
| MMMU vision | 64/100; all completed in 752.75 s and 39 reached the 2,048-token cap |

The comprehensive black-box checker reported 14 passes, 6 failures, and 5
skips. Core discovery, native generation and streaming, OpenAI chat streaming,
Responses API, multi-output, recovery, vision, tool parsing, and reasoning
parsing passed. Strict parity/determinism checks for completions text, seeded
token IDs, blocked-token filtering, prompt-cache hit reporting, and concurrent
output differed; Anthropic request translation returned HTTP 400. Treat the
deployment as operational for its validated native/OpenAI paths, not as a
claim that every optional compatibility path is green.

The SGLang-style result files are:

- `/nvme/sufubao/m39-home/results/glm53_h100_bench/lightllm_h100_sglang_style_c1.jsonl`;
- `/nvme/sufubao/m39-home/results/glm53_h100_bench/lightllm_h100_sglang_style_c100.jsonl`.

Relevant experiment run prefixes are `260829-210446` (health checker),
`260829-210839` (text), `260829-210908` (vision), `260829-211036`
(exact 1M), `260829-211521` (concurrency 1), `260829-212146`
(concurrency 100), `260829-215835` (GSM8K), and `260829-220106`
(MMMU).
