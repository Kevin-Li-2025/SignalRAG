#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"
OUT="${OUT:-reports/qwen25-32b-instruct-awq-vllm}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DTYPE="${DTYPE:-auto}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-awq_marlin}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
HUMANEVAL_GATE="${HUMANEVAL_GATE:-0.85}"
GSM8K_GATE="${GSM8K_GATE:-0.93}"
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
  --tasks humaneval,gsm8k \
  --out "${OUT}" \
  --dtype "${DTYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --vllm-cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB}" \
  "${QUANTIZATION_ARG[@]}" \
  --gsm8k-n-shot 4 \
  --gsm8k-template evalscope \
  --humaneval-template claimbench \
  --humaneval-timeout 4.0 \
  --temperature-code 0.0 \
  --temperature-math 0.0 \
  "${LIMIT_ARG[@]}"

"${PYTHON_BIN}" -m claimbench.cli gate \
  --summary "${OUT}/summary.json" \
  --humaneval "${HUMANEVAL_GATE}" \
  --gsm8k "${GSM8K_GATE}"
