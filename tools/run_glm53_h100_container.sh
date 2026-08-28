#!/usr/bin/env bash
set -euo pipefail

image="${LIGHTLLM_GLM53_IMAGE:-lightllm-glm53:h100-tp8}"
name="${LIGHTLLM_GLM53_CONTAINER:-glm53-lightllm}"
model_dir="${LIGHTLLM_GLM53_MODEL_DIR:-/home/devsft/models/GLM-5.3-Flash}"
cache_dir="${LIGHTLLM_GLM53_CACHE_DIR:-/home/devsft/cache-glm53-lightllm}"
triton_cache_dir="${LIGHTLLM_GLM53_TRITON_CACHE_DIR:-/home/devsft/cache-glm53-triton}"

if [[ ! -d "${model_dir}" ]]; then
  echo "model directory does not exist: ${model_dir}" >&2
  exit 1
fi

mkdir -p "${cache_dir}" "${triton_cache_dir}"

exec sudo docker run --rm --name "${name}" \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  -v "${model_dir}:/model:ro" \
  -v "${cache_dir}:/root/.cache" \
  -v "${triton_cache_dir}:/root/.triton" \
  "${image}"
