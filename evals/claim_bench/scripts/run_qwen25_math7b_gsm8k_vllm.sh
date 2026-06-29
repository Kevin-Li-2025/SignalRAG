#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"
OUT="${OUT:-reports/qwen25-math-7b-instruct-vllm-gsm8k}"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
MAX_NEW_MATH_TOKENS="${MAX_NEW_MATH_TOKENS:-512}"
GSM8K_N_SHOT="${GSM8K_N_SHOT:-4}"
GSM8K_TEMPLATE="${GSM8K_TEMPLATE:-evalscope}"
TEMPERATURE_MATH="${TEMPERATURE_MATH:-0.0}"
TOP_P_MATH="${TOP_P_MATH:-1.0}"
TOP_K_MATH="${TOP_K_MATH:--1}"
REPETITION_PENALTY_MATH="${REPETITION_PENALTY_MATH:-1.0}"
SEED="${SEED:-1}"
GSM8K_GATE="${GSM8K_GATE:-0.93}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

"${PYTHON_BIN}" -m claimbench.cli inspect-env
"${PYTHON_BIN}" -m claimbench.cli eval \
  --model "${MODEL}" \
  --backend vllm \
  --tasks gsm8k \
  --out "${OUT}" \
  --dtype "${DTYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --max-new-math-tokens "${MAX_NEW_MATH_TOKENS}" \
  --gsm8k-n-shot "${GSM8K_N_SHOT}" \
  --gsm8k-template "${GSM8K_TEMPLATE}" \
  --temperature-math "${TEMPERATURE_MATH}" \
  --top-p-math "${TOP_P_MATH}" \
  --top-k-math "${TOP_K_MATH}" \
  --repetition-penalty-math "${REPETITION_PENALTY_MATH}" \
  --seed "${SEED}" \
  "${LIMIT_ARG[@]}"

"${PYTHON_BIN}" -m claimbench.cli gate \
  --summary "${OUT}/summary.json" \
  --humaneval none \
  --gsm8k "${GSM8K_GATE}"
