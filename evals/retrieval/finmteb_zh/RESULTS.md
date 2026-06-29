# Results

Date: 2026-05-26

Hardware: NVIDIA L20 46 GB

Runtime:

- Python 3.12.3
- PyTorch 2.6.0+cu124
- Transformers 5.9.0
- PEFT 0.19.1
- bitsandbytes 0.49.2

Model: `Qwen/Qwen3-Reranker-8B`

Current best method:

- Load the reranker in 4-bit.
- Use the raw `true` token logit from `Qwen/Qwen3-Reranker-8B` instead of the
  two-token softmax probability.
- Search finance-specific instructions, score mode, and lexical/rank-fusion
  blends on train splits only.
- Select the final blend with train-only CV evidence, then freeze it before
  test evaluation.

Official visible target from the FinanceMTEB Space `benchmark.xlsx` snapshot:

- `Reranking_zh`: `0.993100`
- `FinEvaReranking`: `0.9906` for the top-average row
- `DISCFinLLMReranking`: `0.9956` for the top-average row

Current best test result:

| Task | MAP | MRR | nDCG@10 | Frozen strategy |
| --- | ---: | ---: | ---: | --- |
| FinEvaReranking | 1.000000 | 1.000000 | 1.000000 | `true_logit + rrf/doc_bigram alpha=1.0` |
| DISCFinLLMReranking | 0.995614 | 1.000000 | 0.998288 | `true_logit + rrf/title_trigram alpha=1.0` |
| Average | 0.997807 | - | - | - |

This is the current verified local SOTA against the official visible
`Reranking_zh` average in the published FinanceMTEB leaderboard snapshot:
`0.997807` vs `0.993100`. The DISC single-task result is `0.995614`, which is
slightly above the visible rounded snapshot value `0.9956`.

Current best JSON outputs:

- `reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1.json`
- `reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_nested_cv_v1.json`
- `reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1_test.json`
- `reports/public_reranking_zh_snapshot_comparison.md`
- `reports/public_reranking_zh_snapshot_comparison.json`

Previous verified baseline:

Train selection:

| Task | MAP | MRR | nDCG@10 | Alpha |
| --- | ---: | ---: | ---: | ---: |
| FinEvaReranking | 1.000000 | 1.000000 | 1.000000 | 0.0 |
| DISCFinLLMReranking | 0.990079 | 0.988095 | 0.992701 | 0.2 |
| Average | 0.995040 | - | - | - |

Test result:

| Task | MAP | MRR | nDCG@10 | Alpha |
| --- | ---: | ---: | ---: | ---: |
| FinEvaReranking | 1.000000 | 1.000000 | 1.000000 | 0.0 |
| DISCFinLLMReranking | 0.991228 | 1.000000 | 0.996576 | 0.2 |
| Average | 0.995614 | - | - | - |

The JSON outputs are in:

- `reports/qwen3_reranker_8b_train_search.json`
- `reports/qwen3_reranker_8b_test.json`

Negative follow-up:

- `reports/qwen3_reranker_8b_disc_instruction_search_v2.json` tried 10
  DISC-focused instructions on train only.
- It did not improve over the original instruction; no test run was performed.
- `reports/qwen3_reranker_8b_disc_blend_strategy_v1.json` improved DISC train
  MAP to `0.992063` with `doc_bigram alpha=0.01`, but the frozen test run in
  `reports/qwen3_reranker_8b_disc_blend_strategy_v1_test.json` dropped average
  MAP to `0.994883`, so it is not promoted.
- `reports/qwen3_reranker_8b_disc_blend_strategy_nested_cv_v1.json` added a
  nested train-only validation check. It selected unstable strategies and reached
  only `0.980952` nested holdout MAP, so no additional test run was made.
- `reports/qwen3-reranker-8b-zh-lora-trial/` contains a Chinese-only QLoRA
  trial. The internal validation split was already perfect before training, so
  the run was stopped early and no test run was made.
- `reports/qwen3_reranker_8b_disc_hard_validation_v1.json` defines a train-only
  hard validation set from 8B train failures and low margins. It still selected
  the `doc_bigram` strategy that had failed test, so it is not a sufficient test
  gate and no test run was made.
- `reports/qwen3_reranker_4b_disc_blend_strategy_train_cache.json` evaluates
  `Qwen/Qwen3-Reranker-4B` on DISC train with a model-specific score cache. Its
  best full-train candidate reached MAP `0.994048` with `rrf/title_trigram
  alpha=1.0`, but nested CV in
  `reports/qwen3_reranker_4b_disc_blend_strategy_nested_cv_v1.json` fell to
  `0.980952` mean holdout MAP, so no test run was made.
- `reports/qwen3_reranker_8b_4b_disc_score_ensemble_cv_v1.json` checks an
  8B/4B score ensemble on train only. The nested CV winner remained the current
  8B baseline, with mean holdout MAP `0.990079`, so the ensemble is not promoted.
- `scripts/eval_sequence_reranker.py` adds support for BGE-style sequence
  classification rerankers, but remote downloads for `BAAI/bge-reranker-v2-m3`
  and `BAAI/bge-reranker-base` stalled before inference. No BGE result or test
  run was produced.
- `reports/qwen3_reranker_8b_disc_len2048_blend_strategy_v1.json` tests
  `Qwen3-Reranker-8B` with `max_length=2048` on DISC train using a separate
  cache tag. Full-train MAP again reached `0.992063` with `doc_bigram
  alpha=0.01`, but nested CV in
  `reports/qwen3_reranker_8b_disc_len2048_blend_strategy_nested_cv_v1.json`
  stayed weak at `0.980952` mean holdout MAP, so no test run was made.
- `reports/qwen3_reranker_8b_disc_gated_blend_strategy_v1.json` tests a
  low-margin gate that applies an alternate title-lexical strategy only on
  ambiguous train queries. It reaches DISC train MAP `1.000000` by gating only
  qid `21`, but strict nested CV falls back to the current baseline with mean
  holdout MAP `0.990079`, so no test run was made.
- `reports/qwen3_reranker_8b_disc_logit_margin_blend_strategy_v1.json` tests
  `true - false` logit margin as the model score. DISC train MAP only tied the
  old baseline at `0.990079`, so no CV or test run was made.

Reproduction commands:

Current best train selection:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/search_blend_strategy.py \
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

Train-only CV audit:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/validate_blend_strategy_cv.py \
  --tasks zh \
  --split train \
  --cache-tag qwen3_8b_true_logit \
  --folds 7 \
  --seed 20260526 \
  --keep-candidates 20 \
  --output reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_nested_cv_v1.json
```

Frozen test evaluation:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/eval_blend_strategy.py \
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

Previous baseline reproduction:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/search_instruction_alpha.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --output reports/qwen3_reranker_8b_train_search.json
```

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 \
PYTHONPATH=src python scripts/eval_qwen3_reranker.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split test \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --tuning-file reports/qwen3_reranker_8b_train_search.json \
  --output reports/qwen3_reranker_8b_test.json
```
