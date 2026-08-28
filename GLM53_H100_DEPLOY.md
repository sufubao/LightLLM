# GLM-5.3-Flash on H100 TP8

This branch includes a self-contained LightLLM image variant for one eight-GPU
H100 node. The image contains the LightLLM source and runtime dependency; only
the model and compiler-cache directories are mounted from the host.

## Build

Use an immutable tag containing the full source revision:

```bash
revision="$(git rev-parse HEAD)"
created="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
version="v1.2.0-h100-tp8-${revision:0:12}"

docker buildx build --load --platform linux/amd64 \
  -f docker/Dockerfile.glm53-h100 \
  --build-arg "OCI_CREATED=${created}" \
  --build-arg "OCI_REVISION=${revision}" \
  --build-arg "OCI_VERSION=${version}" \
  -t "lightllm-glm53:${version}" \
  .
```

After verification, the host-local convenience alias may point to the same
image ID:

```bash
docker tag "lightllm-glm53:${version}" lightllm-glm53:h100-tp8
```

## Run on h100

The default image command is the measured no-speculation, concurrency-256
profile on port 8002. Run it in the foreground with:

```bash
LIGHTLLM_GLM53_IMAGE="lightllm-glm53:${version}" \
  tools/run_glm53_h100_container.sh
```

Override `LIGHTLLM_GLM53_MODEL_DIR`, `LIGHTLLM_GLM53_CACHE_DIR`, or
`LIGHTLLM_GLM53_TRITON_CACHE_DIR` when the host paths differ. In another shell,
wait for the model list endpoint:

```bash
curl --fail --show-error http://127.0.0.1:8002/v1/models
```

Stop the foreground process with `Ctrl-C`. If it was detached externally, use
`sudo docker stop --timeout 30 glm53-lightllm`.

## Measured profile

The final pre-release candidate reached 4169.40 output tokens/s at concurrency
256 with random 1024-token inputs and 256-token outputs. This is 5.74% below
the measured vLLM result, so the earlier three-percent stretch goal remains
unmet at concurrency 256. Concurrency 16 and 64 exceeded the corresponding
vLLM and SGLang measurements when run with MTP2.
