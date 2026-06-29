# SOTA Strategy

This repository separates verified local evidence from candidate selection.

## Current Verified Result

The checked-in evidence is a full, deterministic run on one NVIDIA L20:

- Model: `Qwen2.5-7B-Instruct`
- HumanEval Pass@1: 79.88% (131 / 164)
- GSM8K exact match: 89.99% (1187 / 1319)

This is the single-model result currently safe to quote as local evidence from this repository.

## Next Single-Model Target

The next credible single-model target is `Qwen/Qwen2.5-32B-Instruct-AWQ`.

Reasons:

- The official Qwen2.5 LLM report lists `Qwen2.5-32B-Instruct` at 88.4 HumanEval and 95.9 GSM8K.
- ModelScope hosts a 4-bit AWQ package for `Qwen2.5-32B-Instruct-AWQ` at about 19.34 GB, which is feasible on a single L20 with a short context cap.
- vLLM can convert this AWQ model to the `awq_marlin` kernel at runtime, which is much faster than forcing plain `awq`.
- The run still needs to pass this harness before it becomes local evidence, because quantization and prompt protocol can move scores.

Run:

```bash
MODEL=/home/hhai/models/modelscope/Qwen/Qwen2___5-32B-Instruct-AWQ \
PYTHON_BIN=/home/hhai/vllm-l20-exp/bin/python \
bash scripts/run_qwen25_32b_awq_vllm.sh
```

If memory is tight, lower `VLLM_MAX_MODEL_LEN` first. Do not lower `LIMIT` for evidence.

## Best Specialist Track

The strongest checked-in specialist evidence currently uses task routing:

- HumanEval: `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`
- GSM8K: `Qwen/Qwen2.5-Math-7B-Instruct`

This can be a valid system benchmark, but it must be labeled as routed. It is not a single-model claim.
The current full-split specialist evidence is:

- HumanEval Pass@1: 93.29% (153 / 164)
- GSM8K exact match: 95.91% (1265 / 1319)

Run:

```bash
MODEL=/home/hhai/models/modelscope/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
PYTHON_BIN=/home/hhai/vllm-l20-exp/bin/python \
bash scripts/run_qwen3_coder30b_fp8_humaneval_vllm.sh
```

For GSM8K-only verification of the current best math model:

```bash
MODEL=/home/hhai/models/modelscope/Qwen/Qwen2___5-Math-7B-Instruct \
PYTHON_BIN=/home/hhai/vllm-l20-exp/bin/python \
GSM8K_TEMPLATE=qwen25-math-cot \
GSM8K_N_SHOT=0 \
MAX_NEW_MATH_TOKENS=1024 \
bash scripts/run_qwen25_math7b_gsm8k_vllm.sh
```

For GSM8K-only verification of the larger AWQ math model:

```bash
MODEL=/home/hhai/models/modelscope/ShelterW/Qwen2___5-Math-72B-Instruct-AWQ \
PYTHON_BIN=/home/hhai/vllm-l20-exp/bin/python \
bash scripts/run_qwen25_math72b_awq_gsm8k_vllm.sh
```

The older routed baseline is retained because it is cheaper to rerun:

- HumanEval: `Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4`
- GSM8K: `Qwen/Qwen2.5-Math-7B-Instruct`
- HumanEval Pass@1: 90.85% (149 / 164)
- GSM8K exact match: 93.33% (1231 / 1319)

The stronger `Qwen2.5-Math-7B-Instruct` GSM8K run uses the official
Qwen2.5-Math CoT zero-shot protocol. The best checked-in variant uses 8 samples
with majority selection and is retained at 95.91% (1265 / 1319); the
deterministic one-sample run is retained separately at 95.75% (1263 / 1319).

## Absolute-SOTA Attempts

The current routed result is strong but still below the public high-water marks
reported for larger open models. The next attempts are:

- `Qwen3-Coder-30B-A3B-Instruct-FP8` for HumanEval:

```bash
MODEL=/home/hhai/models/modelscope/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
PYTHON_BIN=/home/hhai/vllm-l20-exp/bin/python \
bash scripts/run_qwen3_coder30b_fp8_humaneval_vllm.sh
```

- `Qwen2.5-72B-Instruct-AWQ` for a single-model mixed HumanEval/GSM8K run:

```bash
MODEL=/home/hhai/models/modelscope/Qwen/Qwen2___5-72B-Instruct-AWQ \
PYTHON_BIN=/home/hhai/vllm-l20-exp/bin/python \
bash scripts/run_qwen25_72b_awq_vllm.sh
```

- Completed: `Qwen2.5-Math-72B-Instruct-AWQ` for a GSM8K specialist run:

```bash
MODEL=/home/hhai/models/modelscope/ShelterW/Qwen2___5-Math-72B-Instruct-AWQ \
PYTHON_BIN=/home/hhai/vllm-l20-exp/bin/python \
bash scripts/run_qwen25_math72b_awq_gsm8k_vllm.sh
```

This run is now checked in at 94.39% (1245 / 1319). Continue treating any new
candidate as unevidenced until the full split is rerun locally and the resulting
summary is checked in.

- Completed: `Qwen2.5-Math-7B-Instruct` with the official Qwen2.5-Math CoT
  zero-shot prompt for GSM8K:

```bash
MODEL=/home/hhai/models/modelscope/Qwen/Qwen2___5-Math-7B-Instruct \
PYTHON_BIN=/home/hhai/vllm-l20-exp/bin/python \
GSM8K_TEMPLATE=qwen25-math-cot \
GSM8K_N_SHOT=0 \
MAX_NEW_MATH_TOKENS=1024 \
bash scripts/run_qwen25_math7b_gsm8k_vllm.sh
```

This run is now checked in at 95.75% (1263 / 1319) for deterministic decoding
and 95.91% (1265 / 1319) for the 8-sample majority variant.

- Completed: `Qwen3-Coder-30B-A3B-Instruct-FP8` with complete prompt and
  recommended sampled decoding for HumanEval:

```bash
MODEL=/home/hhai/models/modelscope/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
PYTHON_BIN=/home/hhai/vllm-l20-exp/bin/python \
HUMANEVAL_TEMPLATE=complete \
MAX_NEW_CODE_TOKENS=1024 \
TEMPERATURE_CODE=0.7 \
TOP_P_CODE=0.8 \
TOP_K_CODE=20 \
REPETITION_PENALTY_CODE=1.05 \
bash scripts/run_qwen3_coder30b_fp8_humaneval_vllm.sh
```

This run is now checked in at 93.29% (153 / 164).

## Invalid Shortcuts

These should not be used as reported benchmark evidence:

- Running with `LIMIT` and reporting it as a full benchmark.
- Repairing HumanEval outputs after looking at benchmark test failures.
- Selecting per-sample outputs from multiple attempts while still calling it Pass@1.
- Copying official model-card scores into this repository without rerunning this harness.
