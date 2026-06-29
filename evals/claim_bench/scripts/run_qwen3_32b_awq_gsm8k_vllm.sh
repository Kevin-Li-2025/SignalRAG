#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"
OUT="${OUT:-reports/qwen3-32b-awq-vllm-gsm8k-thinking}"
MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-awq_marlin}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
GSM8K_GATE="${GSM8K_GATE:-0.94}"
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
  "${EAGER_ARG[@]}" \
  "${QUANTIZATION_ARG[@]}" \
  --max-new-math-tokens 2048 \
  --gsm8k-n-shot 4 \
  --gsm8k-template evalscope \
  --chat-enable-thinking \
  --temperature-math 0.6 \
  --top-p-math 0.95 \
  "${LIMIT_ARG[@]}"

"${PYTHON_BIN}" -m claimbench.cli gate \
  --summary "${OUT}/summary.json" \
  --humaneval none \
  --gsm8k "${GSM8K_GATE}"
