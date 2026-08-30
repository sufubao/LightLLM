#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/nvme/sufubao/models/Qwen3.8-Flash-Next}"
PYTHON_BIN="${PYTHON_BIN:-/nvme/sufubao/m39-home/venv/bin/python}"
PORT="${PORT:-8123}"
HOST="${HOST:-0.0.0.0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
MAX_REQ_TOTAL_LEN="${MAX_REQ_TOTAL_LEN:-1000000}"
MAX_TOTAL_TOKEN_NUM="${MAX_TOTAL_TOKEN_NUM:-1000036}"
BATCH_MAX_TOKENS="${BATCH_MAX_TOKENS:-2048}"
RUNNING_MAX_REQ_SIZE="${RUNNING_MAX_REQ_SIZE:-8}"
# Keep one extra batch of HTTP request slots without increasing the model's
# active batch limit or its decode CUDA Graph shapes.
SHM_REQ_MAX_SIZE="${SHM_REQ_MAX_SIZE:-16}"
GRAPH_MAX_BATCH_SIZE="${GRAPH_MAX_BATCH_SIZE:-8}"
GRAPH_MAX_LEN_IN_BATCH="${GRAPH_MAX_LEN_IN_BATCH:-2048}"
SCHEDULE_TIME_INTERVAL="${SCHEDULE_TIME_INTERVAL:-0.001}"
IDLE_BATCH_COALESCE_QUIET_TIME="${IDLE_BATCH_COALESCE_QUIET_TIME:-0.002}"
IDLE_BATCH_COALESCE_MAX_WAIT="${IDLE_BATCH_COALESCE_MAX_WAIT:-0.020}"
ENABLE_VISION="${ENABLE_VISION:-0}"
ENABLE_MTP="${ENABLE_MTP:-1}"
MTP_STEP="${MTP_STEP:-3}"
ENABLE_PREFILL_CUDAGRAPH="${ENABLE_PREFILL_CUDAGRAPH:-1}"
PREFILL_CUDAGRAPH_CAPTURE_SHAPES="${PREFILL_CUDAGRAPH_CAPTURE_SHAPES:-32:1:32 256:8:32}"
PREFILL_ATT_BACKENDS="${PREFILL_ATT_BACKENDS:-auto flashinfer}"
DECODE_ATT_BACKENDS="${DECODE_ATT_BACKENDS:-auto triton}"
read -r -a PREFILL_ATT_BACKEND_ARGS <<< "${PREFILL_ATT_BACKENDS}"
read -r -a DECODE_ATT_BACKEND_ARGS <<< "${DECODE_ATT_BACKENDS}"

ARGS=(
  --model_dir "${MODEL_DIR}"
  --tp 4
  --host "${HOST}"
  --port "${PORT}"
  --data_type bfloat16
  --max_total_token_num "${MAX_TOTAL_TOKEN_NUM}"
  --batch_max_tokens "${BATCH_MAX_TOKENS}"
  --max_req_total_len "${MAX_REQ_TOTAL_LEN}"
  --running_max_req_size "${RUNNING_MAX_REQ_SIZE}"
  --shm_req_max_size "${SHM_REQ_MAX_SIZE}"
  --graph_max_batch_size "${GRAPH_MAX_BATCH_SIZE}"
  --graph_max_len_in_batch "${GRAPH_MAX_LEN_IN_BATCH}"
  --schedule_time_interval "${SCHEDULE_TIME_INTERVAL}"
  --idle_batch_coalesce_quiet_time "${IDLE_BATCH_COALESCE_QUIET_TIME}"
  --idle_batch_coalesce_max_wait "${IDLE_BATCH_COALESCE_MAX_WAIT}"
  --llm_prefill_att_backend "${PREFILL_ATT_BACKEND_ARGS[@]}"
  --llm_decode_att_backend "${DECODE_ATT_BACKEND_ARGS[@]}"
  --disable_dynamic_prompt_cache
  --disable_audio
)

if [[ "${ENABLE_PREFILL_CUDAGRAPH}" == "1" ]]; then
  read -r -a PREFILL_GRAPH_SHAPES <<< "${PREFILL_CUDAGRAPH_CAPTURE_SHAPES}"
  ARGS+=(
    --enable_prefill_cudagraph
    --prefill_cudagraph_max_handle_token 256
    --prefill_cudagraph_capture_shapes "${PREFILL_GRAPH_SHAPES[@]}"
  )
fi

if [[ "${ENABLE_VISION}" == "1" ]]; then
  ARGS+=(
    --visual_tp 4
    --visual_dp 1
    --visual_gpu_ids 0 1 2 3
  )
else
  ARGS+=(--disable_vision)
fi

if [[ "${ENABLE_MTP}" == "1" ]]; then
  ARGS+=(
    --mtp_mode eagle_with_att
    --mtp_step "${MTP_STEP}"
  )
fi

PATH="${CUDA_HOME}/bin:${PATH}" \
CUDA_HOME="${CUDA_HOME}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" \
LOADWORKER="${LOADWORKER:-4}" \
  exec "${PYTHON_BIN}" -m lightllm.server.api_server "${ARGS[@]}" "$@"
