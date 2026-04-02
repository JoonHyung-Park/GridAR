#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GRIDAR_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

CATEGORY="${1:-color}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running.}"
export OPENAI_API_KEY
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-benchmark/T2I-CompBench}"
export JANUS_MODEL_PATH="${JANUS_MODEL_PATH:-checkpoints/Janus_ckpt/Janus-Pro-7B}"
export ORM_MODEL_PATH="${ORM_MODEL_PATH:-checkpoints/Qwen2.5-VL-7B-Instruct}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark/T2I-CompBench/outputs}"

cd "${PROJECT_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 evaluate_gridar.py \
  benchmark="compbench_${CATEGORY}" \
  model_params.model.path="${JANUS_MODEL_PATH}" \
  orm.model_path="${ORM_MODEL_PATH}" \
  benchmark_root="${BENCHMARK_ROOT}" \
  output_root="${OUTPUT_ROOT}"
