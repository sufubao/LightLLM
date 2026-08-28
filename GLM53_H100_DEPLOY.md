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

Set `IMAGE` to the immutable registry tag or digest listed in the pull request.
The fixed container name is `glm53-lightllm-vl-1m`. The image default uses the
8,192-token prefill chunk validated on H200. The command below overrides the
default with a conservative 1,024-token chunk for an 80 GB H100, where the DSA
score matrix has much less temporary-memory headroom.

```bash
IMAGE=registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm:<immutable-tag>

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
  --batch_max_tokens 65536 \
  --chunked_prefill_size 1024 \
  --linear_att_ssm_data_type bfloat16 \
  --linear_att_cache_size 256 \
  --disable_cudagraph \
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

The H200 accuracy, long-context, and throughput results in the pull request are
measured through the OpenAI-compatible endpoint. H100 performance is not
inferred from H200 data, and the conservative H100 override above must be
validated independently on the target host before production traffic.
