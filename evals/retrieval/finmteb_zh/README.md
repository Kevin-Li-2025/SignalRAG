# FinMTEB ZH Reranker Evaluation

[![Corrected GPU rerun](https://img.shields.io/badge/result-corrected_GPU_rerun-green)](RESULTS.md)
[![CI](https://img.shields.io/github/checks-status/Kevin-Li-2025/finmteb-zh-reranker-sota/main?label=CI)](https://github.com/Kevin-Li-2025/finmteb-zh-reranker-sota/commits/main)
[![CD](https://img.shields.io/badge/CD-release_workflow_configured-blue)](.github/workflows/release.yml)
[![Model](https://img.shields.io/badge/Model-Qwen3--Reranker--8B-black)](https://huggingface.co/Qwen/Qwen3-Reranker-8B)

Finance-domain Chinese reranking evaluation for FinanceMTEB `Reranking_zh`.

## Result

This project targets the FinanceMTEB Chinese reranking slice:

- `FinanceMTEB/FinEvaRetrieval-reranking`
- `FinanceMTEB/DISCFinLLM-reranking`

The historical `0.997807` MAP result is **invalidated**. The old
pipeline placed positives first, broke RRF/metric ties by stable input order,
and did not consistently reject truncated score vectors.

The corrected RTX 4090 rerun selected each strategy on train only, froze test,
and evaluated both BF16 and bitsandbytes NF4. Every arm passed four candidate-
order seeds with identical metrics.

| Precision | FinEva MAP | DISC MAP | Macro MAP | Macro nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 0.990566 | 1.000000 | 0.995283 | 0.996518 |
| NF4 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

NF4-minus-BF16 macro MAP is +0.004717 on these frozen splits. No general NF4
superiority or SOTA claim is supported: FinEva has 53 test queries and DISC has
19. The evaluator now uses deterministic
label-independent candidate ordering, equal midranks for tied RRF inputs,
tie-aware expected MAP/MRR/nDCG, and strict vector-length checks.

See `RESULTS.md` for exact scores, commands, and environment details. The
auditable GPU bundle is in
[`reports/corrected_gpu_matrix_v1/`](reports/corrected_gpu_matrix_v1/EVIDENCE.md).
The public snapshot comparison is recorded in
`reports/public_reranking_zh_snapshot_comparison.md`.

## Method

The historical evaluation path was:

1. Reproduce the official leaderboard snapshot.
2. Run a zero-shot Qwen3 reranker with train-only score-mode and rank-fusion selection.
3. Freeze the selected strategy, then evaluate on untouched test splits.

Corrected frozen setup:

- Base model: `Qwen/Qwen3-Reranker-8B`
- Inference: BF16 and bitsandbytes NF4 on NVIDIA RTX 4090
- Score mode: raw `true` token logit
- Fusion: per-query z-score linear blend with lexical features
- Selection policy: train-only search/CV, frozen test evaluation
- Protocol: `protocols/finmteb_zh_corrected_gpu_v1.json`

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/finmteb_sota/` | Dataset loading, scoring, metrics, lexical features, score caching |
| `scripts/search_blend_strategy.py` | Train-only score-mode and RRF strategy search |
| `scripts/eval_blend_strategy.py` | Frozen strategy evaluation on test |
| `scripts/validate_blend_strategy_cv.py` | Train-only CV audit |
| `reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1_test.json` | Current best test result |
| `reports/public_reranking_zh_snapshot_comparison.md` | Public leaderboard snapshot comparison |

## CI/CD

GitHub Actions CI runs on every push and pull request to `main`:

- Python 3.10 and 3.12
- `ruff check .`
- `python -m compileall -q src scripts tests`
- `python -m pytest -q`

The CI badge at the top of this README tracks `.github/workflows/ci.yml`.

The release workflow runs on `v*` tags or manual dispatch:

- builds wheel and source distribution
- builds a clean source zip without caches or score-cache files
- uploads artifacts
- publishes a GitHub Release for version tags

## Setup

Use Python 3.10, 3.11, or 3.12 on the GPU machine.

```bash
cd /Users/yinxiaogou/Documents/resume/finmteb-sota-reranker
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[train]"
```

Login to Hugging Face if you plan to push the adapter:

```bash
huggingface-cli login
```

## Leaderboard Snapshot

```bash
python scripts/finmteb_leaderboard_snapshot.py
```

Current target from the official `benchmark.xlsx` snapshot:

- `Reranking_zh` top average is about `0.9931`.
- `FinEvaReranking` is saturated at `1.0000`.
- `DISCFinLLMReranking` top visible score is about `0.9956`.

The public snapshot is retained as context only. The corrected tie-aware protocol
is not claimed as a like-for-like leaderboard submission.

## Current Best Zero-Shot Run

Search score mode and rank-fusion blends on train only:

```bash
python scripts/search_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --score-mode true_logit \
  --cache-tag qwen3_8b_true_logit \
  --keep-candidates 25 \
  --output reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1.json
```

Audit the frozen choice on train-only CV:

```bash
python scripts/validate_blend_strategy_cv.py \
  --tasks zh \
  --split train \
  --cache-tag qwen3_8b_true_logit \
  --folds 7 \
  --seed 20260526 \
  --keep-candidates 20 \
  --output reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_nested_cv_v1.json
```

Then freeze the train-selected strategy for test:

```bash
python scripts/eval_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split test \
  --strategy-file reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1.json \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --score-mode true_logit \
  --cache-tag qwen3_8b_true_logit \
  --output reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1_test.json
```

## Previous Baseline

The earlier instruction/alpha search is still useful as a simple reproduction
baseline.

```bash
python scripts/search_instruction_alpha.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --output reports/qwen3_reranker_8b_train_search.json
```

Then freeze the selected instruction and alpha for test:

```bash
python scripts/eval_qwen3_reranker.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split test \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --tuning-file reports/qwen3_reranker_8b_train_search.json \
  --output reports/qwen3_reranker_8b_test.json
```

If you want only alpha tuning with a fixed instruction:

```bash
python scripts/eval_qwen3_reranker.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --lexical-grid \
  --output reports/qwen3_reranker_8b_train_tune.json
```

If memory is tight, use `Qwen/Qwen3-Reranker-4B` first.

## QLoRA Training

```bash
accelerate launch scripts/train_qwen3_reranker_lora.py \
  --config configs/l20_qwen3_reranker_8b.yaml
```

Evaluate the adapter:

```bash
python scripts/search_instruction_alpha.py \
  --model Qwen/Qwen3-Reranker-8B \
  --adapter outputs/qwen3-reranker-8b-finmteb-lora \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --output reports/qwen3_8b_lora_train_search.json
```

```bash
python scripts/eval_qwen3_reranker.py \
  --model Qwen/Qwen3-Reranker-8B \
  --adapter outputs/qwen3-reranker-8b-finmteb-lora \
  --tasks zh \
  --split test \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --tuning-file reports/qwen3_8b_lora_train_search.json \
  --output reports/qwen3_8b_lora_test.json
```

## Guardrails

- Do not tune on test splits.
- Tune the lexical blend on train or validation only, then freeze it.
- Report per-task MAP/MRR/nDCG, not just the average.
- Save every command and commit hash before publishing.
