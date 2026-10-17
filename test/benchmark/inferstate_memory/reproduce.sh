#!/usr/bin/env bash
# Reproduce the report on one available GPU; used here: m35 H200 physical GPU 2.
set -euo pipefail
cd "$(dirname "$0")/../../.."
result_dir=${1:-/tmp/inferstate-memory-results}
model_dir=${2:-/mtc/models/Qwen3.5-27B}
mkdir -p "$result_dir"
exp -m "InferState ownership A/B suite; model ${model_dir}; BF16 TP1, 4096-token prefill, default GC, Triton full+linear attention; plain and prompt logprobs, uncapped and equal allocator budget" \
    python test/benchmark/inferstate_memory/run.py \
    --suite --workload model --model-dir "$model_dir" --ownership weak \
    --gc-mode default --iterations 150 --output "$result_dir"
python test/benchmark/inferstate_memory/plot.py "$result_dir"
