# LLM Benchmark Harness

> Migrated from
> [`Kevin-Li-2025/llm-claim-bench`](https://github.com/Kevin-Li-2025/llm-claim-bench)
> into SignalRAG as `evals/claim_bench/`. This subproject remains isolated:
> its dependencies, scripts, evidence summaries, and tests are kept under this
> directory so the main SignalRAG app is not forced to install benchmark-only
> GPU dependencies.

Auditable HumanEval and GSM8K harness for single-GPU LLM inference benchmarking.
The runs in this repository are inference-only reproductions with public model
weights. They do not train, fine-tune, or modify model checkpoints.

## Verified L20 Result

Full run on one NVIDIA L20 with `Qwen2.5-7B-Instruct`, vLLM, BF16, deterministic decoding, cached full benchmark datasets, and no sample limit:

- HumanEval Pass@1: **79.88%** (131 / 164)
- GSM8K exact match: **89.99%** (1187 / 1319)

This run is the checked-in mixed-task baseline:

- `evidence/qwen25-7b-instruct-vllm-mixed-summary.json`
- `evidence/qwen25-7b-instruct-vllm-mixed-summary.md`

Full per-sample predictions are generated under `reports/` and intentionally not committed.

## Best Verified Specialist Results

These are full-split specialist inference runs on the same L20. They are valid as
task-specialist or routed-system scores, not as a single-model mixed benchmark:

- HumanEval Pass@1: **93.29%** (153 / 164), `Qwen3-Coder-30B-A3B-Instruct-FP8`, complete prompt with recommended sampled decoding
- GSM8K exact match: **95.91%** (1265 / 1319), `Qwen2.5-Math-7B-Instruct`, official Qwen2.5-Math CoT prompt with majority over 8 samples

The `Qwen2.5-Coder-32B-Instruct-GPTQ-Int4` HumanEval reproduction is retained
separately: **91.46%** (150 / 164) with fixed sampled decoding, and **90.85%**
(149 / 164) greedy. The `Qwen2.5-Math-72B-Instruct-AWQ` GSM8K run is retained as
a larger math baseline at **94.39%** (1245 / 1319), the deterministic official
Qwen2.5-Math CoT run is retained at **95.75%** (1263 / 1319), and the older
EvalScope-style `Qwen2.5-Math-7B-Instruct` run is retained at **93.33%** (1231 / 1319).

### Qwen3-Coder Prompt Sweep

Full-split prompt sweep on `Qwen3-Coder-30B-A3B-Instruct-FP8`, one L20, vLLM,
temperature `0.0`, no sample limit:

| Prompt template | HumanEval Pass@1 |
| --- | ---: |
| `evalscope` | **92.07%** (151 / 164) |
| `complete` + recommended sampling | **93.29%** (153 / 164) |
| `complete` | **91.46%** (150 / 164) |
| `concise` | **90.24%** (148 / 164) |
| `claimbench` | **89.63%** (147 / 164) |
| `raw` | **53.05%** (87 / 164) |

The sweep confirmed that the EvalScope-style instruction is the strongest fixed
prompt among the tested variants for this model and harness.

Checked-in summaries:

- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-evalscope-summary.json`
- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-evalscope-summary.md`
- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-complete-recommended-s1-summary.json`
- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-complete-recommended-s1-summary.md`
- `evidence/qwen25-coder-32b-instruct-gptq-marlin-vllm-humaneval-evalscope-t02-s1-summary.json`
- `evidence/qwen25-coder-32b-instruct-gptq-marlin-vllm-humaneval-evalscope-t02-s1-summary.md`
- `evidence/qwen25-coder-32b-instruct-gptq-marlin-vllm-humaneval-evalscope-summary.json`
- `evidence/qwen25-coder-32b-instruct-gptq-marlin-vllm-humaneval-evalscope-summary.md`
- `evidence/qwen25-math-72b-instruct-awq-marlin-vllm-gsm8k-full-u099-l4096-eager-summary.json`
- `evidence/qwen25-math-72b-instruct-awq-marlin-vllm-gsm8k-full-u099-l4096-eager-summary.md`
- `evidence/qwen25-math-7b-instruct-vllm-gsm8k-qwen25-cot-summary.json`
- `evidence/qwen25-math-7b-instruct-vllm-gsm8k-qwen25-cot-summary.md`
- `evidence/qwen25-math-7b-instruct-vllm-gsm8k-qwen25-cot-majority8-summary.json`
- `evidence/qwen25-math-7b-instruct-vllm-gsm8k-qwen25-cot-majority8-summary.md`
- `evidence/qwen25-math-7b-instruct-vllm-gsm8k-summary.json`
- `evidence/qwen25-math-7b-instruct-vllm-gsm8k-summary.md`
- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-claimbench-sweep-summary.json`
- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-claimbench-sweep-summary.md`
- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-complete-sweep-summary.json`
- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-complete-sweep-summary.md`
- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-concise-sweep-summary.json`
- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-concise-sweep-summary.md`
- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-raw-sweep-summary.json`
- `evidence/qwen3-coder-30b-a3b-fp8-vllm-humaneval-raw-sweep-summary.md`

To verify the checked-in summaries without rerunning inference:

```bash
PYTHON_BIN=python bash scripts/check_evidence.sh
```

## SOTA Candidate Track

The remaining single-L20 targets are stronger quantized or MoE candidates:

- `Qwen3-Coder-30B-A3B-Instruct-FP8` for HumanEval.
- `Qwen2.5-72B-Instruct-AWQ` for a high single-model mixed run.

These remaining targets are candidate-selection signals only; they are not local evidence until
this harness reruns the full split with no `LIMIT`.

Completed candidate evidence:

- `Qwen2.5-Math-7B-Instruct` GSM8K specialist, one L20, vLLM, official
  `qwen25-math-cot` zero-shot prompt, 8 samples with majority selection:
  **95.91%** (1265 / 1319).
- `Qwen2.5-Math-72B-Instruct-AWQ` GSM8K specialist, one L20, vLLM
  `awq_marlin`, no CPU offload, max model length 4096, eager execution:
  **94.39%** (1245 / 1319).

To run the candidate:

```bash
MODEL=/path/to/Qwen3-Coder-30B-A3B-Instruct-FP8 \
PYTHON_BIN=python \
bash scripts/run_qwen3_coder30b_fp8_humaneval_vllm.sh
```

The SOTA candidate gates are stricter than the baseline gates. A result only
becomes a verified benchmark result after a full, no-limit run writes a checked-in
summary.

See `configs/sota_candidates.json` and `docs/sota_strategy.md` for the single-model
and routed-system candidate plan.

For a maximum-score system benchmark, use `scripts/run_qwen25_32b_routed_vllm.sh`.
It runs HumanEval and GSM8K with different 32B models, then writes a combined
summary explicitly labeled as a routed system rather than a single-model score.
For a HumanEval-only code specialist run, use
`scripts/run_qwen25_coder32b_humaneval_vllm.sh`. For a GSM8K-only math specialist
run, use `scripts/run_qwen25_math7b_gsm8k_vllm.sh`.

## Reproduce On The GPU Machine

```bash
cd llm-claim-bench
python3.11 -m venv .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,vllm]"

MODEL=/path/to/Qwen2.5-7B-Instruct \
PYTHON_BIN=python \
bash scripts/run_qwen25_7b_mixed_vllm.sh
```

On the reference L20 machine, the exact model path used was:

```bash
MODEL=/home/hhai/models/modelscope/Qwen/Qwen2___5-7B-Instruct \
PYTHON_BIN=/home/hhai/vllm-l20-exp/bin/python \
bash scripts/run_qwen25_7b_mixed_vllm.sh
```

Use `LIMIT=4` for a fast smoke test. Do not report limited smoke-test outputs as
benchmark results.

## Evaluation Protocol

HumanEval:

- Loads `openai/openai_humaneval` from Hugging Face Datasets, with fallback to `openai_humaneval`.
- Uses the model tokenizer chat template for the instruction-tuned model.
- Uses the selected HumanEval instruction template. The mixed-task baseline uses
  `claimbench`; the current best code-specialist run uses `evalscope`.
- Generates one completion per task with temperature `0.0`.
- Executes `prompt + completion + test + check(entry_point)` in an isolated Python subprocess with a timeout.
- Reports Pass@1 as `passed / total`.

GSM8K:

- Loads `gsm8k/main` from Hugging Face Datasets.
- Uses a 4-shot EvalScope-style prompt and asks for the final answer in `\boxed{}`.
- Uses the model tokenizer chat template for the instruction-tuned model.
- Generates one completion per task with temperature `0.0`.
- Runs vLLM generation in harness-controlled chunks. `batch_size` is an
  effective scheduling control, not just report metadata; full-split runs should
  not send the entire GSM8K test set to one `llm.generate()` call.
- Extracts the final numeric answer from `####`, `\boxed{}`, `answer is`, or the last numeric expression.
- Reports exact match over the full test split.

Optional GSM8K self-consistency:

- `--gsm8k-samples N` generates `N` sampled completions per problem.
- `--gsm8k-selection majority` selects the most frequent extracted answer, breaking ties by earliest sampled occurrence.
- Self-consistency runs are labeled separately in `summary.json` / `summary.md` and should not be mixed with the deterministic single-sample score.

The scripts record model id, decoding settings, limits, Python/Torch/Transformers versions, CUDA device metadata, and per-sample outputs so the result can be audited.

Safety note: HumanEval necessarily executes generated Python. This harness blocks common destructive calls and uses subprocess timeouts, but it is not a hard security sandbox. For untrusted model outputs, run it in a disposable container or VM.
