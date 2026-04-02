#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GRIDAR_ROOT=$(cd "${PROJECT_DIR}/.." && pwd)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

PIE_BENCH_ROOT="${PIE_BENCH_ROOT:-${GRIDAR_ROOT}/benchmark/PIE-Bench}"
DATA_ROOT="${DATA_ROOT:-${PIE_BENCH_ROOT}}"
EDITAR_CKPT_ROOT="${EDITAR_CKPT_ROOT:-${GRIDAR_ROOT}/checkpoints/EditAR_ckpt}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${GRIDAR_ROOT}/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PIE_BENCH_ROOT}/outputs}"
OUTPUT_NAME="${OUTPUT_NAME:-GridAR__EditAR__PIE__N4}"

python3 "${PROJECT_DIR}/sample_edit_gridar.py" \
  --dataset-path "${DATA_ROOT}" \
  --vq-ckpt "${EDITAR_CKPT_ROOT}/pretrained_models/vq_ds16_t2i.pt" \
  --gpt-ckpt "${EDITAR_CKPT_ROOT}/editar_release.pt" \
  --t5-path "${EDITAR_CKPT_ROOT}/pretrained_models/t5-ckpt" \
  --output-dir "${OUTPUT_ROOT}" \
  --output-name "${OUTPUT_NAME}" \
  --N 4 \
  "$@"

python3 "${PROJECT_DIR}/orm_verifier.py" \
  --model_id "${QWEN_MODEL_PATH}" \
  --dataset_path "${DATA_ROOT}" \
  --output_dir "${OUTPUT_ROOT}" \
  --output_name "${OUTPUT_NAME}" \
  --key 0 1 2 3
