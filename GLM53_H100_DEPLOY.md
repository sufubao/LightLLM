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

### Same-host LightLLM, SGLang, and vLLM comparison

The three engines were measured sequentially on the same otherwise-idle
8xH100 80 GB host on 2026-08-30. All used the same local FP8 checkpoint, TP8,
BF16 KV cache, a declared 1,048,576-token context, no speculative decoding,
and an 8,192-token prefill budget. The common SGLang `bench_serving` client
used seed 42, temperature 0, streaming, ignored EOS, an infinite request rate,
and one excluded warmup request. `random-range-ratio=0` sampled input and
output lengths from 1 through 1,000 tokens. The exact c1 samples contained
3,309 input and 3,700 output tokens; the exact c100 samples contained 504,929
input and 494,908 output tokens. Every engine completed every request.

| Engine | Pinned build and material server differences |
| --- | --- |
| LightLLM | Published image above; TP8, 1,024-token chunks, CUDA graph disabled, FlashInfer all-reduce disabled |
| SGLang | `lmsysorg/sglang@sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf` (`0.0.0.dev1+gf609d677b`); TP8/EP8, DeepGEMM, `mem-fraction-static=0.80`, 1,024-token chunks |
| vLLM | `vllm/vllm-openai:glm53-flash-x86_64-cu130@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703` (`0.1.dev20051+g487ecf187`); TP8, 256 sequences, chunked prefill and prefix caching, `gpu-memory-utilization=0.90` |

| Engine | c1 duration | c1 output / total tok/s | c100 duration | c100 output / total tok/s | c100 total vs LightLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightLLM | 319.65 s | 11.58 / 21.93 | 1,962.10 s | 252.23 / 509.58 | 1.00x |
| SGLang | 91.23 s | 40.56 / 76.83 | 328.59 s | 1,506.16 / 3,042.82 | 5.97x |
| vLLM | 36.30 s | 101.92 / 193.07 | 270.74 s | 1,827.97 / 3,692.96 | 7.25x |

At c1, SGLang and vLLM delivered 3.50x and 8.81x LightLLM's total
throughput, respectively; vLLM was 2.51x SGLang. At c100, vLLM was 1.214x
SGLang. SGLang and vLLM also passed the same arithmetic smoke check by
returning `579` for `37 * 16 - 13`.

The SGLang server first failed during CUDA-graph capture at
`mem-fraction-static=0.90`; the reported run used its suggested `0.80` while
retaining the 1M context declaration and enough KV capacity for this workload.
vLLM used the dedicated pre-merge GLM-5.3 image because the ordinary public
image on the host did not register `Glm5Next`; its official recipe requires
BF16 KV on Hopper. Its startup also warned that the H100/288-expert combination
lacked a model-specific MoE tuning table. SGLang and vLLM expose reasoning
tokens under different streaming fields, and LightLLM's chunk shape is not
recognized consistently by this client, so cross-engine TTFT/ITL and
retokenized-text counts are not compared. Successful requests, API usage token
counts, wall time, and the aggregate throughput figures above are directly
comparable.

References: [SGLang GLM-5 benchmark recipe](https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/GLM/GLM-5.mdx),
[SGLang serving benchmark](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/base/benchmarks/autoregressive_model_benchmark.mdx),
[vLLM GLM-5.3-Flash recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash),
and [vLLM GLM-5.3 support PR](https://github.com/vllm-project/vllm/pull/53906).

The comprehensive black-box checker reported 14 passes, 6 failures, and 5
skips. Core discovery, native generation and streaming, OpenAI chat streaming,
Responses API, multi-output, recovery, vision, tool parsing, and reasoning
parsing passed. Strict parity/determinism checks for completions text, seeded
token IDs, blocked-token filtering, prompt-cache hit reporting, and concurrent
output differed; Anthropic request translation returned HTTP 400. Treat the
deployment as operational for its validated native/OpenAI paths, not as a
claim that every optional compatibility path is green.

The benchmark result files are:

- `/nvme/sufubao/m39-home/results/glm53_h100_bench/lightllm_h100_sglang_style_c1.jsonl`;
- `/nvme/sufubao/m39-home/results/glm53_h100_bench/lightllm_h100_sglang_style_c100.jsonl`;
- `/nvme/sufubao/m39-home/results/glm53_h100_engine_compare/sglang_h100_random_1k_c1.jsonl`;
- `/nvme/sufubao/m39-home/results/glm53_h100_engine_compare/sglang_h100_random_1k_c100.jsonl`;
- `/nvme/sufubao/m39-home/results/glm53_h100_engine_compare/vllm_h100_random_1k_c1.jsonl`;
- `/nvme/sufubao/m39-home/results/glm53_h100_engine_compare/vllm_h100_random_1k_c100.jsonl`.

Relevant experiment run prefixes are `260829-210446` (health checker),
`260829-210839` (text), `260829-210908` (vision), `260829-211036`
(exact 1M), `260829-211521` (concurrency 1), `260829-212146`
(concurrency 100), `260829-215835` (GSM8K), and `260829-220106`
(MMMU). Comparison runs are `260830-002938`, `260830-003031`, and
`260830-003246` for SGLang smoke/c1/c100, and `260830-005212`,
`260830-005226`, and `260830-005412` for vLLM smoke/c1/c100.
