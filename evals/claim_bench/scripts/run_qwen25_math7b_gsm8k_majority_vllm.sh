#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"
OUT="${OUT:-reports/qwen25-math-7b-instruct-vllm-gsm8k-majority}"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
GSM8K_N_SHOT="${GSM8K_N_SHOT:-4}"
GSM8K_TEMPLATE="${GSM8K_TEMPLATE:-evalscope}"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-8}"
GSM8K_TEMPERATURE="${GSM8K_TEMPERATURE:-0.7}"
GSM8K_TOP_P="${GSM8K_TOP_P:-0.95}"
GSM8K_TOP_K="${GSM8K_TOP_K:--1}"
GSM8K_REPETITION_PENALTY="${GSM8K_REPETITION_PENALTY:-1.0}"
SEED="${SEED:-1}"
GSM8K_GATE="${GSM8K_GATE:-0.94}"
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
  --max-new-math-tokens 512 \
  --gsm8k-n-shot "${GSM8K_N_SHOT}" \
  --gsm8k-template "${GSM8K_TEMPLATE}" \
  --gsm8k-samples "${GSM8K_SAMPLES}" \
  --gsm8k-selection majority \
  --temperature-math "${GSM8K_TEMPERATURE}" \
  --top-p-math "${GSM8K_TOP_P}" \
  --top-k-math "${GSM8K_TOP_K}" \
  --repetition-penalty-math "${GSM8K_REPETITION_PENALTY}" \
  --seed "${SEED}" \
  "${LIMIT_ARG[@]}"

"${PYTHON_BIN}" -m claimbench.cli gate \
  --summary "${OUT}/summary.json" \
  --humaneval none \
  --gsm8k "${GSM8K_GATE}"
