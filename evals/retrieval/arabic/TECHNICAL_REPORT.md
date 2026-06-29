# Technical Report: Arabic Retrieval Lab

Last updated: 2026-06-06

## Summary

This project builds a benchmark-first Arabic retrieval and embedding research
pipeline. The strongest completed result is a MIRACL Arabic dev retrieval
system:

| System | nDCG@10 | MAP@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| BGE-M3 hybrid r100 + `BAAI/bge-reranker-v2-m3` min-max score blend | 0.828605 | 0.769687 | 0.842686 |
| BGE-M3 hybrid r100 + `BAAI/bge-reranker-v2-gemma` reranker-only | 0.823758 | 0.764017 | 0.839831 |
| BGE-M3 hybrid r100 + `BAAI/bge-reranker-v2-m3` reranker-only | 0.817731 | 0.756480 | 0.835130 |
| BGE-M3 hybrid r100 first stage | 0.801110 | 0.738021 | 0.819846 |
| BGE-M3 dense-only | 0.785230 | 0.720700 | 0.808149 |

The result is strong, but it is a retrieval-system result, not a new standalone
Arabic embedding model. A standalone model claim would require a trained
checkpoint that performs retrieval without a cross-encoder reranker and passes
robust same-task validation.

## Research Question

Can we build a defensible Arabic retrieval or embedding system that improves
over strong public multilingual and Arabic-specific baselines under the same
benchmark, split, and metric definitions?

Current answer:

- Yes, for a MIRACL Arabic dev retrieval system inside this repo's reproducible
  evaluation setup.
- Not yet, for a standalone Arabic embedding/retriever checkpoint.
- Not yet, for formal public leaderboard SOTA.

## Evaluation Surface

Primary benchmark:

- Dataset/task: `mteb/MIRACLRetrieval`
- Subset: Arabic (`ar`)
- Split: `dev`
- Revision: `9c09abc13478308c27598f350e31d8f06b9b5481`
- Main metric: nDCG@10
- Secondary metrics: MAP@10, MRR@10, Recall@10, Precision@10, hit rate@10

MIRACL Arabic is useful because it is a high-signal multilingual retrieval task
with native relevance judgments and a large corpus. It is still one task
family, so it cannot alone justify broad Arabic embedding SOTA claims.

## Baselines

The repo tracks several public baseline families:

- `BAAI/bge-m3`: multilingual dense/sparse/multi-vector retriever.
- `intfloat/multilingual-e5-large-instruct`: strong instruction-style
  multilingual dense retriever.
- `Qwen/Qwen3-Embedding-8B`: high-capacity multilingual dense baseline.
- `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2`: Arabic
  triplet/Matryoshka model; weaker in this MIRACL retrieval setup.
- `BAAI/bge-reranker-v2-m3`: fast multilingual cross-encoder reranker.
- `BAAI/bge-reranker-v2-gemma`: larger FlagEmbedding LLM reranker.

## Best Retrieval System

The current best system has three stages:

1. First-stage retrieval:
   Use BGE-M3 hybrid scoring over MIRACL Arabic corpus candidates. This combines
   dense, sparse, and ColBERT-style multi-vector evidence.
2. Cross-encoder reranking:
   Rerank the frozen top-100 candidates per query with
   `BAAI/bge-reranker-v2-m3`.
3. Score blending:
   Blend per-query min-max-normalized first-stage and reranker scores:

```text
0.65 * normalized_reranker_score + 0.35 * normalized_first_stage_score
```

The blend was selected on alternating tune queries and evaluated on the
remaining holdout queries before reporting the full-dev score.

## Key Results

### v43: BGE-M3 Hybrid First Stage

The frozen BGE-M3 hybrid r100 setup scored:

- nDCG@10: `0.801110`
- MAP@10: `0.738021`
- MRR@10: `0.819846`

This became the stable first-stage candidate generator.

### v47: BGE Reranker v2-M3

Reranking the v43 top-100 candidate pool with `BAAI/bge-reranker-v2-m3` scored:

- nDCG@10: `0.817731`
- MAP@10: `0.756480`
- MRR@10: `0.835130`
- Gain over v43: `+0.016621` nDCG@10

This established that cross-encoder reranking adds real headroom beyond the
hybrid first-stage retriever.

### v56: Score Blend

A disciplined blend sweep over the v47 ranked output selected min-max
normalization with reranker weight `alpha=0.65`.

Tune/holdout:

- Tune nDCG@10: `0.826880`
- Holdout nDCG@10: `0.830329`
- Holdout gain over reranker-only: `+0.013449`

Full dev:

- nDCG@10: `0.828605`
- MAP@10: `0.769687`
- MRR@10: `0.842686`
- Full gain over v47: `+0.010874`
- Full gain over v43: `+0.027495`

This is the strongest completed result in the repo.

### v57: BGE Reranker v2-Gemma

The larger `BAAI/bge-reranker-v2-gemma` full-dev rerank completed:

- nDCG@10: `0.823758`
- MAP@10: `0.764017`
- MRR@10: `0.839831`

It beats the v47 reranker-only setup by `+0.006028` nDCG@10, but trails the v56
score blend by `-0.004846`. It is useful as a teacher-diversity candidate, but
not the current best system.

## Standalone Student Work

The project is also trying to distill the retrieval stack into a standalone
Arabic retriever. That work has not yet produced a publishable checkpoint.

Current diagnostic rule:

- evaluate the student on four query-disjoint MIRACL Arabic diagnostic slices;
- require at least `+0.005` nDCG@10 over frozen BGE-M3 on both best-vs-best and
  model-card fusion surfaces;
- inspect per-query regressions;
- reject sparse+ColBERT-only or slice-specific improvements.

Important student results:

| Version | Idea | Diagnostic result | Decision |
| --- | --- | --- | --- |
| v58 | Dense-only MarginMSE student | large regression | reject |
| v61/v62 | Official FlagEmbedding M3 head-only KD | best gain `+0.003575`, below gate | reject |
| v69a | Tiny full-encoder smoke | best-vs-best `-0.000256` | reject |
| v72a/v74 | Surface-aware LoRA | promising held-out signal, unstable across slices | reject |
| v77/v80 | Multi-surface/failure-aware LoRA | weak model-card/base-best movement | reject |
| v82 | Anti-regression LoRA | model-card `+0.004190`, best-vs-best `+0.001203` | reject |

v82 is the most useful negative result: explicit anti-regression supervision
moved the selected same-weight surface from sparse+ColBERT to the model-card
fusion surface, but still failed the strict gate. This suggests the next attempt
should change the objective, not simply scale the same recipe.

## Negative Results And Lessons

The project intentionally records failed experiments because they shape the
research direction:

- Dense-only training on BGE-M3 with MIRACL teacher data regressed.
- MarginMSE training against hybrid teacher scores also regressed.
- Qwen3-Reranker-0.6B did not beat the first-stage BGE-M3 hybrid gate.
- Head-only sparse/ColBERT distillation produced unstable or too-small gains.
- LoRA adapter smokes trained cleanly, but did not robustly improve the
  model-card/base-best surfaces.

Practical lesson: the near-term problem is not GPU plumbing. It is objective
design. The student needs a training signal that directly preserves and improves
the strong BGE-M3 fusion surface rather than only moving weak ablations.

## Interpretation

The strongest evidence today is for a retrieval system:

```text
BGE-M3 hybrid retrieval -> BGE reranking -> score blending
```

This is valuable for Arabic information retrieval and as a teacher candidate
for distillation. It is not yet evidence that a new Arabic embedding model has
been trained.

The next credible model-building step is:

1. Keep the v56 retrieval system as the strongest teacher reference.
2. Preserve the v82 anti-regression insight.
3. Replace or augment stock M3 KD with direct pair/list constraints against
   base model-card winners.
4. Validate with the four-slice gate before any full-dev evaluation.
5. Only publish a checkpoint after same-task evidence shows robust gains.

## Public Claim Guidance

Safe project description:

> A benchmark-first Arabic retrieval and embedding research lab with a strong
> MIRACL Arabic retrieval-system result using BGE-M3 hybrid retrieval,
> cross-encoder reranking, and disciplined score blending.

Avoid:

> Arabic embedding SOTA model.

Avoid until hidden/public benchmark evidence supports it:

> Formal SOTA.
