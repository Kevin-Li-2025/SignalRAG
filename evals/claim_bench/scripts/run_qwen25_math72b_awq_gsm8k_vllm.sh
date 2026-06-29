#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"
OUT="${OUT:-reports/qwen25-math-72b-instruct-awq-marlin-vllm-gsm8k}"
MODEL="${MODEL:-ShelterW/Qwen2.5-Math-72B-Instruct-AWQ}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.96}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-awq_marlin}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
MAX_NEW_MATH_TOKENS="${MAX_NEW_MATH_TOKENS:-512}"
GSM8K_N_SHOT="${GSM8K_N_SHOT:-4}"
GSM8K_TEMPLATE="${GSM8K_TEMPLATE:-evalscope}"
TEMPERATURE_MATH="${TEMPERATURE_MATH:-0.0}"
TOP_P_MATH="${TOP_P_MATH:-1.0}"
TOP_K_MATH="${TOP_K_MATH:--1}"
REPETITION_PENALTY_MATH="${REPETITION_PENALTY_MATH:-1.0}"
SEED="${SEED:-1}"
GSM8K_GATE="${GSM8K_GATE:-0.95}"
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

EAGER_ARG=()
case "${VLLM_ENFORCE_EAGER}" in
  1|true|TRUE|yes|YES) EAGER_ARG=(--vllm-enforce-eager) ;;
esac

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
  --vllm-cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB}" \
  "${EAGER_ARG[@]}" \
  "${QUANTIZATION_ARG[@]}" \
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
