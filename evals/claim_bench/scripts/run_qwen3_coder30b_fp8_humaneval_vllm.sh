#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"
OUT="${OUT:-reports/qwen3-coder-30b-a3b-fp8-vllm-humaneval-evalscope}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DTYPE="${DTYPE:-auto}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"
HUMANEVAL_TEMPLATE="${HUMANEVAL_TEMPLATE:-evalscope}"
HUMANEVAL_GATE="${HUMANEVAL_GATE:-0.927}"
MAX_NEW_CODE_TOKENS="${MAX_NEW_CODE_TOKENS:-384}"
TEMPERATURE_CODE="${TEMPERATURE_CODE:-0.0}"
TOP_P_CODE="${TOP_P_CODE:-1.0}"
TOP_K_CODE="${TOP_K_CODE:--1}"
REPETITION_PENALTY_CODE="${REPETITION_PENALTY_CODE:-1.0}"
SEED="${SEED:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

QUANTIZATION_ARG=()
if [[ -n "${VLLM_QUANTIZATION}" ]]; then
  QUANTIZATION_ARG=(--vllm-quantization "${VLLM_QUANTIZATION}")
fi

"${PYTHON_BIN}" -m claimbench.cli inspect-env
"${PYTHON_BIN}" -m claimbench.cli eval \
  --model "${MODEL}" \
  --backend vllm \
  --tasks humaneval \
  --out "${OUT}" \
  --dtype "${DTYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  "${QUANTIZATION_ARG[@]}" \
  --max-new-code-tokens "${MAX_NEW_CODE_TOKENS}" \
  --humaneval-template "${HUMANEVAL_TEMPLATE}" \
  --humaneval-timeout 4.0 \
  --temperature-code "${TEMPERATURE_CODE}" \
  --top-p-code "${TOP_P_CODE}" \
  --top-k-code "${TOP_K_CODE}" \
  --repetition-penalty-code "${REPETITION_PENALTY_CODE}" \
  --seed "${SEED}" \
  "${LIMIT_ARG[@]}"

"${PYTHON_BIN}" -m claimbench.cli gate \
  --summary "${OUT}/summary.json" \
  --humaneval "${HUMANEVAL_GATE}" \
  --gsm8k none
