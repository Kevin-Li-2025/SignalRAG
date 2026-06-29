# Arabic Retrieval Lab

Last updated: 2026-06-05

Benchmark-first Arabic retrieval and embedding research lab. The goal is to
build a genuinely strong standalone Arabic retriever checkpoint, but the repo is
deliberately conservative about public claims: results are reported only for the
same benchmark, split, and metric actually evaluated.

## TL;DR

Current best completed result:

| Result type | System | Benchmark | Main metric | Score |
| --- | --- | --- | --- | ---: |
| Retrieval system | BGE-M3 hybrid r100 + `BAAI/bge-reranker-v2-m3` min-max blend | MIRACL Arabic dev | nDCG@10 | 0.828605 |

This is a strong Arabic retrieval-system result. It is not a standalone Arabic
embedding-model SOTA claim, because the final ranking uses a hybrid retriever
plus cross-encoder reranker scores. The standalone-student work is active, but
no trained checkpoint has passed the repo's robustness gate yet.

Short version:

- Strongest system: `BAAI/bge-m3` dense+sparse+multi-vector retrieval,
  reranked by `BAAI/bge-reranker-v2-m3`, then score-blended.
- Strongest standalone baseline in this repo: frozen BGE-M3 hybrid first stage,
  `0.801110` nDCG@10 on MIRACL Arabic dev.
- Best standalone-student signal so far: v82 moved the selected diagnostic
  surface from sparse+ColBERT to model-card fusion, but still failed the strict
  four-slice gate.
- Current decision: do not publish or scale any v58-v82 student checkpoint.

## What This Repo Is

This is a research artifact for Arabic retrieval and embedding experiments. It
contains:

- reproducible MIRACL/MTEB-style evaluation code;
- derived experiment records with exact metrics and decisions;
- public-baseline comparisons against BGE-M3, E5, Qwen3, and an Arabic-specific
  embedding model;
- reranking and score-blending experiments;
- student-distillation attempts and explicit rejection gates;
- research notes that record negative results instead of hiding them.

It intentionally does not contain raw datasets, generated embeddings, TREC run
files, remote logs, caches, credentials, or model checkpoints.

## Claim Boundary

Safe wording today:

> A reproducible Arabic retrieval research lab with a strong MIRACL Arabic dev
> retrieval-system result using BGE-M3 hybrid retrieval, cross-encoder
> reranking, and score blending.

Do not claim today:

- "Arabic embedding SOTA model"
- "new standalone Arabic embedding model"
- "formal public leaderboard SOTA"

Those claims require a trained standalone checkpoint and same-task evidence
across official or public leaderboard surfaces. The current best score is a
system result, not a self-owned embedding checkpoint.

## Current Internal Leaderboard

Benchmark: `mteb/MIRACLRetrieval`, subset `ar`, split `dev`, revision
`9c09abc13478308c27598f350e31d8f06b9b5481`.

| Type | Method | nDCG@10 | MAP@10 | MRR@10 | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Score-blended retrieval system | BGE-M3 hybrid r100 + `BAAI/bge-reranker-v2-m3` min-max blend | 0.828605 | 0.769687 | 0.842686 | Best completed result; selected with tune/holdout split |
| Reranked retrieval system | BGE-M3 hybrid r100 + `BAAI/bge-reranker-v2-gemma` | 0.823758 | 0.764017 | 0.839831 | Better than reranker-only v47, worse than v56 score blend |
| Reranked retrieval system | BGE-M3 hybrid r100 + `BAAI/bge-reranker-v2-m3` | 0.817731 | 0.756480 | 0.835130 | Full-dev cross-encoder reranking |
| Hybrid first-stage retriever | `BAAI/bge-m3` dense+sparse+ColBERT r100 | 0.801110 | 0.738021 | 0.819846 | Frozen candidate generator |
| Dense retriever | `BAAI/bge-m3` | 0.785230 | 0.720700 | 0.808149 | Strong dense-only baseline |
| Dense retriever | `intfloat/multilingual-e5-large-instruct` | 0.764560 | 0.695560 | 0.794536 | Strong multilingual baseline |
| Dense retriever | `Qwen/Qwen3-Embedding-8B` | 0.700164 | 0.623864 | 0.719286 | Expensive direct dense baseline |
| Dense retriever | `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` | 0.613070 | 0.532950 | 0.640482 | Arabic-specific model, weak on this retrieval setup |

The v56 blend was selected on alternating tune queries and improved the held-out
half of dev by `+0.013449` nDCG@10 over reranker-only. That is stronger evidence
than a full-dev-only sweep, but it is still a dev-set system result.

## Standalone Student Status

The project is trying to distill the best retrieval stack into a standalone
retriever. The current conclusion is negative but useful:

| Version | Idea | Result | Decision |
| --- | --- | --- | --- |
| v58 | Dense-only BGE-M3 MarginMSE student | Large regression on fixed diagnostic | Reject |
| v61/v62 | Official FlagEmbedding M3 head-only KD | Best diagnostic gain `+0.003575`, below `+0.005` gate | Reject |
| v69a | Tiny full-encoder BGE-M3 smoke | best-vs-best `-0.000256` | Reject |
| v70a/v71a | LoRA adapter smokes | Weak or slice-specific gains | Reject |
| v72a/v74 | Surface-aware LoRA with multi-split check | promising held-out signal, unstable across slices | Reject |
| v77/v80 | Multi-surface and failure-aware rows | sparse+ColBERT movement, model-card/base-best still weak | Reject |
| v82 | Anti-regression teacher rows + LoRA | model-card mean `+0.004190`, best-vs-best `+0.001203`; strict gate failed | Reject |

The useful lesson from v82: explicit anti-regression supervision is better than
plain row reweighting, because the selected same-weight surface moved from
sparse+ColBERT to the model-card fusion surface. It still is not enough. The
next model attempt should change the objective, not simply scale v82.

## Evaluation Philosophy

This repo uses conservative gates because BGE-M3 is already strong. A useful
student must improve the robust public fusion surface, not just a weak ablation.

Current student gate:

- evaluate on four query-disjoint stride slices of the same MIRACL Arabic
  diagnostic candidate pool;
- require at least `+0.005` nDCG@10 on best-vs-best and model-card surfaces;
- inspect per-query regressions;
- reject slice-specific gains and sparse+ColBERT-only improvements;
- run no full-dev evaluation, checkpoint upload, or public claim unless the
  diagnostic gate passes first.

## Research Sources

The repo's benchmark and model choices are based on high-signal public sources:

- [MIRACL](https://github.com/project-miracl/miracl): multilingual information
  retrieval benchmark with Arabic train/dev/test splits and standard IR
  metrics.
- [MTEB MIRACLRetrieval](https://huggingface.co/datasets/mteb/MIRACLRetrieval):
  MTEB task wrapper used for reproducible local evaluation.
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3): multilingual retriever
  exposing dense, sparse, and multi-vector retrieval modes.
- [BGE-M3 technical report](https://arxiv.org/abs/2402.03216): describes the
  unified dense/sparse/multi-vector model and self-knowledge distillation.
- [FlagEmbedding finetune docs](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder):
  official training path used for M3 teacher-score distillation experiments.

## Repository Layout

```text
.
|-- README.md                  # This entry point
|-- RESEARCH.md                # Detailed research log and experiment narrative
|-- TECHNICAL_REPORT.md        # Short report-style summary
|-- MODEL_CARD.md              # Draft model-card surface for future checkpoints
|-- experiments/               # Small derived JSON summaries only
|-- scripts/                   # Evaluation, reranking, distillation, diagnostics
|-- configs/                   # Training/evaluation configs
|-- tests/                     # Unit tests for data prep, gates, and evaluators
|-- src/                       # Lightweight project modules
|-- kaggle_entrypoint.py       # MTEB/Kaggle-style evaluation entrypoint
```

## Reproducibility

Install lightweight local dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run local validation:

```bash
python3 -m compileall -q scripts src tests
PYTHONPATH=scripts:src python3 -m unittest discover -s tests -v
python3 - <<'PY'
import json
from pathlib import Path
for path in sorted(Path("experiments").glob("*.json")):
    json.loads(path.read_text())
print("experiment JSON records parsed")
PY
git diff --check
```

Useful script entry points:

```bash
# Build/score BGE-M3 hybrid MIRACL candidates.
python scripts/run_miracl_bge_m3_hybrid.py --help

# Rerank a frozen candidate run with a cross-encoder or FlagEmbedding reranker.
python scripts/rerank_miracl_candidates.py --help

# Sweep score-level blends over an existing ranked_by_query.json.
python scripts/sweep_reranker_score_blend.py --help

# Compute and sweep BGE-M3 dense/sparse/ColBERT fusion weights.
python scripts/sweep_bge_m3_hybrid_weights.py --help

# Run official FlagEmbedding M3 finetuning in no-DDP mode.
python scripts/run_flagembedding_m3_no_ddp.py --help

# Check whether a student clears the strict multi-split diagnostic gate.
python scripts/check_student_multisplit_gate.py --help

# Analyze per-query student regressions across diagnostic surfaces.
python scripts/analyze_student_surface_failures.py --help
```

The full MIRACL run files, ranked JSON files, generated teacher JSONL files,
model checkpoints, caches, logs, and embeddings are intentionally not committed.

## Important Experiment Records

The full chronological record is in `experiments/` and `RESEARCH.md`. The most
important derived records are:

- `experiments/2026-06-04-miracl-official-ar-bge-m3-hybrid-r100-v43.json`
- `experiments/2026-06-05-bge-reranker-v2-m3-miracl-ar-v47.json`
- `experiments/2026-06-05-bge-reranker-score-blend-v56.json`
- `experiments/2026-06-05-bge-reranker-v2-gemma-full-v57.json`
- `experiments/2026-06-05-student-distill-v58-result.json`
- `experiments/2026-06-05-official-m3-no-ddp-student-smokes-v61-v62.json`
- `experiments/2026-06-05-v72-surface-aware-teacher-and-smoke.json`
- `experiments/2026-06-05-v74-v72a-stride4-stability.json`
- `experiments/2026-06-05-v77-multisurface-lora-smoke.json`
- `experiments/2026-06-05-v80-failure-aware-lora-smoke.json`
- `experiments/2026-06-05-v82-antiregression-lora-smoke.json`

## Data And Artifact Policy

Committed:

- source code;
- configs;
- tests;
- small derived JSON summaries;
- research notes and reports.

Not committed:

- credentials or tokens;
- raw datasets;
- raw Kaggle or remote run outputs;
- generated embeddings;
- model checkpoints;
- Hugging Face, Kaggle, or dataset caches;
- large TREC run files and ranked JSON artifacts.

## Next Step

Do not scale v82 directly. The next credible standalone-model attempt should
keep the anti-regression idea but change the objective more directly:

- add explicit pair/list constraints against base model-card winners;
- raise model-card positive separation without degrading base-best rankings;
- build a validation-aware loss outside the stock M3 KD objective;
- continue requiring the four-slice gate and per-query regression analysis
  before any full-dev evaluation or checkpoint publication.
