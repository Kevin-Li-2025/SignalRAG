# Arabic Embedding Research Notes

Last updated: 2026-06-05

This file is the detailed research log. For a concise project overview, start
with `README.md`. For a short narrative summary suitable for review, see
`TECHNICAL_REPORT.md`.

## Claim Discipline

The requested end state is an Arabic embedding model that can credibly compete
for SOTA. This requires a benchmark definition before training. The current
project uses MTEB/MMTEB-Arabic compatible evaluation as the public comparison
surface because it provides repeatable task loaders, metrics, and public model
results.

Acceptable claim wording by evidence level:

- `baseline`: evaluated public models, no trained model yet.
- `competitive`: trained model beats common multilingual baselines on the
  selected Arabic tasks.
- `public-reference SOTA-level`: trained model beats the strongest tracked
  public reference in the same harness and metric set.
- `formal SOTA`: only if a public leaderboard or paper-style comparison confirms
  the result under identical tasks, splits, and metrics.

## Strong Public Baselines To Evaluate

| Model | Why Track It |
| --- | --- |
| `Swan-Large` / ArabicMTEB references | Paper-level Arabic-centric target; availability needs verification before direct reproduction. |
| `Qwen/Qwen3-Embedding-8B` | Strong multilingual ceiling reference; too large for cheap fine-tuning. Must use official instructions/prompts. |
| `Qwen/Qwen3-Embedding-4B` | Strong multilingual reference, possible high-end inference baseline. Must use official instructions/prompts. |
| `Qwen/Qwen3-Embedding-0.6B` | Practical trainable base, but smoke run underperformed; retest only with official prompted evaluation before choosing it as base. |
| `BAAI/bge-m3` | Multilingual dense/sparse/multi-vector reference with broad retrieval strength; dense-only MTEB underuses its hybrid retrieval design. |
| `intfloat/multilingual-e5-large-instruct` | Strong instruction-style multilingual baseline; query instructions are required for fair retrieval evaluation. |
| `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` | Arabic-specific Matryoshka triplet model; strong STS/pure-Arabic reference, not yet a broad all-task winner. |

## Web Research Snapshot

Sources checked on 2026-06-01:

- MTEB paper: https://arxiv.org/abs/2210.07316
- MTEB org/leaderboard entry point: https://huggingface.co/mteb and
  https://huggingface.co/spaces/mteb/leaderboard
- Swan and ArabicMTEB paper page: https://huggingface.co/papers/2411.01192
- Qwen3 Embedding 0.6B model card:
  https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Multilingual E5 Large Instruct model card:
  https://huggingface.co/intfloat/multilingual-e5-large-instruct
- BGE-M3 model card: https://huggingface.co/BAAI/bge-m3
- Arabic Triplet Matryoshka V2 model card:
  https://huggingface.co/Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2
- GATE paper: https://arxiv.org/abs/2505.24581

Additional source refresh on 2026-06-03:

- MTEB organization and leaderboard entry point:
  https://huggingface.co/mteb
- MIRACLRetrieval task card:
  https://huggingface.co/datasets/mteb/MIRACLRetrieval
- Swan and ArabicMTEB paper page:
  https://huggingface.co/papers/2411.01192
- Qwen3 Embedding official technical note:
  https://qwenlm.github.io/blog/qwen3-embedding/
- BGE-M3 official model card:
  https://huggingface.co/BAAI/bge-m3

Additional source refresh on 2026-06-04:

- BGE-M3 official model card:
  https://huggingface.co/BAAI/bge-m3
- FlagEmbedding official repository:
  https://github.com/FlagOpen/FlagEmbedding
- BGE-M3 official documentation:
  https://bge-model.com/bge/bge_m3.html
- MIRACL official project page:
  https://project-miracl.github.io/
- MIRACL official Hugging Face topics/qrels:
  https://huggingface.co/datasets/miracl/miracl
- FlagEmbedding official embedder fine-tuning examples:
  https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
- SentenceTransformers loss reference:
  https://sbert.net/docs/package_reference/sentence_transformer/losses.html
- SentenceTransformers hard-negative utility reference:
  https://sbert.net/docs/package_reference/util/hard_negatives.html
- BGE-M3 paper:
  https://arxiv.org/abs/2402.03216

Additional source refresh after v41 on 2026-06-04:

- BGE fine-tuning tutorial:
  https://bge-model.com/tutorial/7_Finetuning/7.1.2.html
- FlagEmbedding official repository:
  https://github.com/FlagOpen/FlagEmbedding
- FlagEmbedding fine-tuning README:
  https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/master/examples/finetune/embedder/README.md
- Swan and ArabicMTEB paper:
  https://arxiv.org/abs/2411.01192

Additional source refresh on 2026-06-05:

- NVIDIA Llama-Embed-Nemotron-8B model card:
  https://huggingface.co/nvidia/llama-embed-nemotron-8b
- Jina Embeddings v5 text-small model card:
  https://huggingface.co/jinaai/jina-embeddings-v5-text-small
- Qwen3-Reranker-8B model card:
  https://huggingface.co/Qwen/Qwen3-Reranker-8B
- Qwen3-Reranker-4B model card:
  https://huggingface.co/Qwen/Qwen3-Reranker-4B
- BGE reranker v2 documentation:
  https://bge-model.com/tutorial/5_Reranking/5.2.html
- BGE reranker v2-M3 model card:
  https://huggingface.co/BAAI/bge-reranker-v2-m3
- BGE reranker v2-Gemma model card:
  https://huggingface.co/BAAI/bge-reranker-v2-gemma

Additional v63 source refresh on 2026-06-05:

- BGE-M3 paper:
  https://arxiv.org/abs/2402.03216
- BGE-M3 model card:
  https://huggingface.co/BAAI/bge-m3
- BGE fine-tuning tutorial:
  https://bge-model.com/tutorial/7_Finetuning/7.1.2.html
- BGE-M3 FlagEmbedding modeling API:
  https://bge-model.com/API/finetune/embedder/encoder_only/m3/modeling.html
- Qwen3 Embedding official repository:
  https://github.com/QwenLM/Qwen3-Embedding
- Swan and ArabicMTEB arXiv paper:
  https://arxiv.org/abs/2411.01192
- Swan and ArabicMTEB Hugging Face paper page:
  https://huggingface.co/papers/2411.01192
- MTEB organization and results/submission entry points:
  https://huggingface.co/mteb and
  https://github.com/embeddings-benchmark/results

Additional v64 source refresh on 2026-06-05:

- SentenceTransformers loss documentation:
  https://sbert.net/docs/package_reference/sentence_transformer/losses.html
- FlagEmbedding M3 modeling source documentation:
  https://bge-model.com/_modules/FlagEmbedding/finetune/embedder/encoder_only/m3/modeling.html
- Score-distribution KD paper:
  https://arxiv.org/abs/2604.04734

v64 research implications:

- BGE-M3's public strength is still tied to dense+sparse+ColBERT behavior, so
  head-only student training must preserve the strong fusion surfaces, not only
  improve weak sparse-heavy ablations.
- SentenceTransformers' cached large-batch and guide-model losses are relevant
  alternatives if we later return to dense or dual-encoder training, but v58
  showed naive small MarginMSE is unsafe for this project.
- The score-distribution KD result is especially actionable here: our current
  teacher rows are top-candidate-heavy, and the next student attempt should
  inspect and stratify teacher score distributions before another GPU training
  run.

Research implications:

1. The original MTEB paper explicitly argues that no single embedding method
   dominates across all task families. The project should report task-family
   scores, not only one average.
2. The strongest Arabic-specific benchmark target appears to be ArabicMTEB from
   the Swan paper: cross-lingual, multi-dialectal, multi-domain, and
   multi-cultural evaluation over eight tasks and 94 datasets. A defensible
   Arabic SOTA claim should compare against this target if the benchmark/model
   artifacts are obtainable.
3. Swan-Large is the most relevant paper-level Arabic-centric reference because
   the paper reports it beating Multilingual-E5-large on most Arabic tasks. The
   public accessibility of Swan/ArabicMTEB artifacts still needs verification,
   so it is a claim target, not yet a reproducible baseline in this repo.
4. `intfloat/multilingual-e5-large-instruct` requires query-side task
   instructions for retrieval. Our current raw `SentenceTransformer` smoke
   results are useful for relative sanity checks, but they are not the fairest
   retrieval/reranking numbers for E5.
5. `Qwen/Qwen3-Embedding-0.6B` is instruction-aware, supports Matryoshka-style
   output dimensions, and the model card says instructions usually improve
   downstream results. The weak smoke result does not fully rule it out for
   retrieval until prompted evaluation is run.
6. `BAAI/bge-m3` supports dense, sparse, and multi-vector retrieval across more
   than 100 languages and long inputs. A dense-only MTEB run is a conservative
   baseline, not the upper bound for BGE-M3 in retrieval.
7. `Arabic-Triplet-Matryoshka-V2` and the GATE paper are highly relevant for
   Arabic STS. Their own model card emphasizes STS17/STS22.v2 strength and
   Matryoshka plus MultipleNegativesRankingLoss training. That matches our
   smoke result: excellent pure-Arabic STS, weaker broad multilingual behavior.
8. The MTEB organization remains the right public comparison surface because it
   maintains active leaderboard infrastructure, large result datasets, and
   task-level dataset cards.
9. MIRACL is a high-signal retrieval benchmark, not a throwaway task: the task
   card describes multilingual ad hoc retrieval across 18 languages and native
   relevance judgments. Our Arabic subset result should therefore carry more
   weight than compact sanity tasks like Sadeem.
10. ArabicMTEB/Swan remains the most relevant Arabic-centric SOTA target: the
    benchmark spans eight tasks and 94 datasets, and Swan-Large is reported as
    outperforming Multilingual-E5-large on most Arabic tasks. Reproducible
    Swan/ArabicMTEB artifacts should be actively searched before making any
    final SOTA claim.
11. Qwen3 Embedding is now a mandatory strong reference for SOTA-level claims:
    the official note reports the 8B embedding model as No. 1 on the MTEB
    multilingual leaderboard at release, and the series supports instructions,
    Matryoshka-style dimensions, long context, and dedicated rerankers.
12. BGE-M3's official model card emphasizes dense, sparse, and multi-vector
    retrieval. Our current MTEB runs use dense embedding mode only, so BGE-M3's
    best hybrid retrieval configuration may be stronger than the numbers in
    this repo.
13. BGE-M3 is the most practical next branch because the official model exposes
    dense vectors, lexical sparse weights, and ColBERT-style multi-vector
    scoring from one checkpoint. This lets us improve retrieval without first
    committing to expensive model training.
14. A full Arabic MIRACL ColBERT cache would be large and operationally risky:
    the corpus has 2,061,414 documents, and storing token-level vectors for all
    documents is much heavier than the dense cache. The practical compromise is
    dense+sparse full-corpus candidate generation followed by official
    dense+sparse+ColBERT reranking on a small candidate set.
15. The first BGE-M3 hybrid run should use the official weighted score shape:
    `[0.4, 0.2, 0.4]` for dense, sparse, and ColBERT. After a clean full run,
    tune candidate size and weights only if the result is competitive and the
    change is reported as an ablation, not as a hidden benchmark trick.
16. Teacher-data construction should use the MIRACL `train` split, not the
    `dev` split used for evaluation, to avoid contaminating the benchmark.
17. The teacher-data format should preserve hard negatives and teacher scores:
    FlagEmbedding's preparation flow uses hard-negative mining and can attach
    teacher scores for knowledge distillation, so generated rows should include
    `pos`, `neg`, `pos_scores`, and `neg_scores`.
18. Generated teacher JSONL is derived training data, not a reproducibility
    record. Keep it in `outputs/` or remote `remote_outputs/`; commit only the
    construction script and small non-text summaries.
19. The MTEB MIRACL task mirror used for our evaluation exposes the Arabic
    query/qrel loader as `dev` only. For training-data construction, use
    official `miracl/miracl` train topics/qrels and reuse the same MIRACL
    Arabic corpus through the cached MTEB corpus loader when practical.
20. SentenceTransformers documents `MultipleNegativesRankingLoss` as suitable
    for positive pairs, triplets, and n-tuples with hard negatives, but the
    official docs also distinguish teacher-score distillation losses such as
    `MarginMSELoss` when gold similarities come from a teacher model. Because
    v38 already stores BGE-M3 hybrid `pos_scores` and `neg_scores`, plain MNR
    is likely an incomplete use of the data.
21. SentenceTransformers' hard-negative guidance and the newer hardness options
    for MNR make explicit hard negatives useful, but the v39 smoke shows that
    merely adding one hard negative via the generic trainer path is not enough
    to beat the base BGE-M3 sample result.
22. The BGE-M3 paper and official implementation emphasize self-knowledge
    distillation from multiple retrieval functions. That aligns better with
    our hybrid teacher scores than a pair-only objective.
23. Llama-Embed-Nemotron-8B is now a mandatory high-end multilingual reference:
    its model card reports state-of-the-art multilingual MTEB performance as of
    2025-10-21 and exposes a full recipe/dataset/code release. It should be
    evaluated as a dense external ceiling before any broad SOTA claim.
24. Jina v5 text-small is a new compact multilingual distillation reference:
    the model card reports a 677M Qwen3-0.6B-Base model distilled from
    Qwen3-Embedding-4B plus task-specific contrastive losses. This is a strong
    signal that our practical student path should use teacher distillation and
    task-targeted mixtures, not naive self-training.
25. Qwen3-Reranker models are instruction-aware, multilingual rerankers with
    0.6B/4B/8B sizes. They are not embedding models, so they cannot support a
    pure embedding-model SOTA claim, but they are a legitimate retrieval-system
    stress test over the frozen BGE-M3 hybrid top-100 candidate pool.
26. BGE reranker v2-M3 is a 568M multilingual cross-encoder based on BGE-M3 and
    is the lowest-risk first reranking ablation because it is fast, public, and
    aligned with the current BGE-M3 hybrid first stage.
27. BGE reranker v2-Gemma is a larger multilingual FlagEmbedding LLM reranker.
    It is a legitimate next reranker-system gate, but it has higher operational
    cost and required a non-Xet download path on the remote L20 host.
28. Score-level blending is a low-cost system-improvement route because v47
    cross-encoder reranking does not use the original BGE-M3 hybrid score after
    ranking. A cautious blend must use a tune/holdout split; otherwise, it is
    just dev-set tuning.
29. The v56 min-max score blend is the strongest completed MIRACL Arabic dev
    retrieval-system result in this repo: `0.828605` nDCG@10. It improved both
    tune and holdout halves against reranker-only, which makes it stronger
    evidence than a full-dev-only tweak.
30. This still does not prove an Arabic embedding-model SOTA. The result is a
    retrieval stack over MIRACL dev: BGE-M3 hybrid candidates, BGE reranker
    scores, and score blending. The next model-training branch should distill
    this stronger teacher into a standalone retriever and validate beyond
    MIRACL dev.
31. Swan/ArabicMTEB remains the right Arabic-centric SOTA reference, but it is
    not yet a reliable reproducibility surface for this repo. The arXiv paper
    points to `https://github.com/UBC-NLP/swan`, while the Hugging Face paper
    discussion contains later user comments asking where the Swan models and
    ArabicMTEB data can be found. Until the benchmark artifacts are actually
    obtainable, use Swan as a paper-level target and do not claim direct
    same-harness comparison.
32. The BGE-M3 paper directly supports the current standalone-student direction:
    it trains a single model to combine dense retrieval, sparse retrieval, and
    multi-vector retrieval, with self-knowledge distillation integrating
    relevance scores from different retrieval functions. This matches why the
    v61/v62 official M3 path improved over the custom v59 head-only recipe, even
    though it did not yet clear the scaling gate.
33. The most suspicious unresolved technical gap is mode-weight and temperature
    alignment. The BGE-M3 model card demonstrates inference score fusion with
    `[0.4, 0.2, 0.4]`, while the FlagEmbedding M3 training model API defaults
    `compute_score` to dense `1.0`, sparse `0.3`, and ColBERT `1.0`; the
    fine-tuning tutorial also exposes score temperature and recommends
    teacher-score distillation through `pos_scores` and `neg_scores`. Before
    spending more GPU on larger v62-style runs, isolate whether the student
    checkpoint only looks weak because the train and evaluation score surfaces
    are misaligned.
34. Qwen3 Embedding remains a mandatory high-end external baseline because the
    official repository reports the 8B model as No. 1 on MTEB multilingual at
    release and emphasizes instruction-aware retrieval. It should not replace
    the standalone-student goal, but any later model card must compare against
    Qwen3 where feasible and must use query instructions fairly.
35. MTEB submission research reinforces the distinction between a model and a
    retrieval stack. Public MTEB results are submitted through result files and
    model metadata, but the current v56 system is a hybrid retriever plus
    reranker blend. A standalone checkpoint can be submitted as a model; a
    multi-component system should be reported separately unless its wrapper is
    deterministic, documented, and acceptable for the target benchmark.

## 2026-06-05 SOTA Push Plan

Decision: keep the current BGE-M3 hybrid r100 run frozen as the strong
first-stage retrieval teacher, then test cross-encoder reranking and new dense
ceilings before launching more training.

Rationale:

- The current official-format dev score, `0.801110` nDCG@10, is already in the
  same band as the BGE-M3 paper's Arabic MIRACL `All` reference. More hidden
  weight twiddling is unlikely to create a large, defensible gain.
- The previous dense-only BGE-M3 training smokes regressed, and the official
  FlagEmbedding unified trainer was unstable in the current remote environment.
- A reranker-over-candidates experiment is the fastest way to discover whether
  there is headroom above the BGE-M3 hybrid top-100 ordering before spending GPU
  hours on a student model.
- If reranking improves the score, use the improved ordering as a stronger
  teacher for a trainable student. If reranking hurts, keep BGE-M3 hybrid as the
  teacher and move to broader ArabicMTEB/Swan-style evaluation.
- After v56, the strongest confirmed teacher is the min-max blend of BGE-M3
  hybrid scores and BGE-reranker-v2-m3 scores, not reranker-only v47.

## 2026-06-05 v63 Research Decision

Do not launch a larger student training job yet. The best next step is a
controlled diagnostic that separates three effects before more training:

1. Candidate-pool and metric surface: continue using the fixed v52-v54
   same-candidate MIRACL Arabic diagnostic gate so any gain is attributable to
   the checkpoint or scoring recipe, not to a changed retrieval pool.
2. Weight surface: re-evaluate base BGE-M3 and the best v62 head checkpoint
   under identical weight grids, including the model-card weights
   `[0.4, 0.2, 0.4]`, the normalized training-default shape `[1.0, 0.3, 1.0]`,
   and sparse-heavy/dense-heavy/ColBERT-heavy ablations.
3. Training surface: only if the weight sweep shows the student has a stable
   same-weight advantage should a new small v63 training smoke run. The first
   candidates are lower learning rate, score-temperature alignment, and a
   staged sparse-then-ColBERT curriculum. A larger row count alone is not a
   research design.

The gate remains unchanged: at least `+0.005` nDCG@10 over the base model on
the same candidate-pool diagnostic before any full-dev evaluation, checkpoint
upload, or stronger public wording. A gain caused only by tuning fusion weights
on the base model is a system-scoring insight, not a standalone-student result.

## 2026-06-05 v63 Diagnostic Result

Implementation:

- Added `scripts/sweep_bge_m3_hybrid_weights.py` to compute BGE-M3 component
  scores once for a fixed candidate pool, then sweep dense/sparse/ColBERT
  fusion weights for base and student-head checkpoints.
- Evaluated base BGE-M3 and the v62 official no-DDP head checkpoint on the
  fixed 200-query / top-100 MIRACL Arabic same-candidate diagnostic.
- Remote summaries remain under `remote_outputs/v63-bge-m3-weight-diagnostic`
  and `remote_outputs/v63-bge-m3-weight-diagnostic-expanded`; only the derived
  interpretation is committed.

Findings:

- The model-card surface `[0.4, 0.2, 0.4]` still does not clear the gate:
  v62 scored `0.792839` versus base `0.789204`, delta `+0.003635`.
- The default-grid best same-weight gain was `+0.005495` under sparse-heavy
  `[0.25, 0.5, 0.25]`, but the absolute score there was only `0.785990`,
  below stronger base surfaces.
- The expanded-grid largest same-weight gain was `+0.011573` under
  sparse+ColBERT-only `[0.0, 0.5, 0.5]`, but this was recovery from a weak base
  score: student `0.785669` versus base `0.774096`.
- The strongest base surface in the grid was ColBERT-heavy
  `[0.25, 0.15, 0.6]` at `0.792419`.
- The strongest v62 surface in the expanded grid was `[0.15, 0.15, 0.7]` at
  `0.793258`, only `+0.000839` over the strongest base surface.

Decision:

- Do not scale v62 or publish its checkpoint. The sparse/ColBERT head learned a
  localized correction, but not a robust standalone-model improvement.
- The next student experiment should target this specific gap: preserve or
  improve the model-card/base-best surface while keeping the sparse/ColBERT
  gains. Plausible next designs are a staged sparse+ColBERT curriculum with a
  base-best anchor, lower learning rate, and explicit validation on both
  model-card and ColBERT-heavy surfaces.
- Any future gate should require both same-weight gain and best-vs-best gain,
  because a same-weight gain on a weak fusion surface can overstate model
  progress.

## 2026-06-05 v64 Lower-LR/Temperature Smokes

Setup:

- Ran two official FlagEmbedding no-DDP BGE-M3 unified head-only smokes on the
  remote L20, still freezing the encoder and training only `sparse_linear` and
  `colbert_linear`.
- Both runs used the v50 BGE-reranker-v2-m3 teacher rows, 2,048 rows, group
  size 9, batch size 4, one epoch, `m3_kd_loss`, and the same fixed 200-query /
  top-100 MIRACL Arabic candidate-pool diagnostic.
- v64a used learning rate `3e-6` and temperature `0.02`; v64b used learning
  rate `3e-6` and temperature `0.05`.
- Remote summaries remain under
  `remote_outputs/v64a-lr3e-6-t002-weight-diagnostic/summary.json` and
  `remote_outputs/v64b-lr3e-6-t005-weight-diagnostic/summary.json`; only the
  derived interpretation is committed.

Results:

| Run | LR | Temp | Train loss | Student best | Best-vs-best delta | Model-card delta | Best same-weight delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v64a | 3e-6 | 0.02 | 0.649000 | 0.793058 | +0.000639 | +0.001918 | +0.010557 |
| v64b | 3e-6 | 0.05 | 0.701291 | 0.791406 | -0.001013 | +0.002202 | +0.002432 |

Interpretation:

- v64a confirms the v63 pattern: the head can repair weak sparse-heavy or
  sparse+ColBERT surfaces, but it does not robustly beat the strongest base
  surface. Its largest same-weight gain was `+0.010557`, but best-vs-best was
  only `+0.000639`.
- v64b did not help. A higher temperature slightly improved the model-card
  surface but damaged ColBERT-heavy and sparse-heavy surfaces enough that the
  best student score trailed the best base score by `-0.001013`.
- Do not scale v64, publish its checkpoints, or run a full-dev evaluation. The
  evidence now points away from nearby official head-only hyperparameter
  sweeps.

Decision:

- The next step should be a v65 teacher-data distribution diagnostic, not more
  GPU training. Inspect teacher score histograms, query-level entropy, positive
  and negative strata, and held-out coverage.
- If the distribution is top-heavy, generate stratified rows that mix positives,
  hard negatives, middle-score negatives, and easy negatives before launching
  another official M3 head or broader student run.
- A future student gate should still require both `+0.005` best-vs-best
  nDCG@10 and `+0.005` model-card-surface gain on an independent diagnostic
  before scaling.

## 2026-06-05 v65 Teacher Score Distribution Diagnostic

Setup:

- Added `scripts/analyze_teacher_score_distribution.py`.
- Ran it on the v50 MIRACL Arabic train teacher rows scored by
  `BAAI/bge-reranker-v2-m3`.
- The diagnostic is non-training: it writes only a score-distribution summary
  under `remote_outputs/student-distill-v65-score-distribution`, and no raw
  teacher rows, checkpoints, logs, or caches are committed.

Key measurements:

- Rows with scores: `3,495`; positive pairs: `4,949`; negative pairs: `27,960`.
- Negative sources: `12,889` judged negatives and `15,071` unjudged candidates.
- Best positive above max negative: `75.9%` of rows.
- All positives above all negatives: `62.3%` of rows.
- Teacher best-positive minus max-negative margin: mean `2.466`, p25 `0.120`,
  p50 `2.613`, p75 `4.910`.
- Pair-level best-positive minus each-negative margins are much easier: mean
  `6.274`, p50 `6.342`; `19,671 / 27,960` negative pairs are in the `4_plus`
  margin bin.
- Teacher margin and BGE-M3 hybrid margin are strongly correlated on these
  rows: Pearson `0.829`.
- Teacher softmax targets are too sharp at low temperatures:
  mean normalized entropy is `0.238` at temperature `1.0`, `0.103` at `0.5`,
  `0.0379` at `0.2`, and `0.0189` at `0.1`; median entropy at `0.2` is near
  zero.

Interpretation:

- The v50 teacher rows are useful, but not ideal for student learning. They
  contain hard and disagreement cases at the row level, yet the individual
  negative-pair distribution is dominated by easy negatives.
- This can explain the v62/v64 pattern: the sparse/ColBERT heads learn local
  repairs on weak fusion surfaces, but the training rows do not expose enough
  calibrated score spectrum to improve the strongest base/model-card surfaces.
- The next student attempt should not start by increasing row count. It should
  first generate stratified teacher rows that cover hard, middle-score, and
  easy negative margins, and use a softer teacher target or score rescaling
  before softmax.

Decision:

- No new GPU training yet.
- Implement a stratified teacher-row builder next. Required properties:
  query-disjoint train/held-out split, explicit negative-margin strata,
  preserved raw teacher scores, and summary checks showing healthier entropy
  and margin coverage than v50/v58.
- Only after that should another official M3 no-DDP smoke run, with the same
  `+0.005` best-vs-best and model-card-surface gate.

Implementation:

- `scripts/rerank_miracl_candidates.py` consumes an existing TREC run file,
  reloads the same MIRACL Arabic query/qrel/corpus revision, reconstructs only
  the candidate texts needed for the top-100 pool, scores query-document pairs
  with `FlagReranker`, a direct transformers sequence-classification reranker,
  or `sentence_transformers.CrossEncoder`, writes a reranked run file, and
  recomputes nDCG@10/MAP@10/MRR@10 on the same queries.
- First experiment: BGE-M3 hybrid r100 candidates plus
  `BAAI/bge-reranker-v2-m3`, starting with a 200-query dev sample. If it beats
  the first-stage score on the identical sample, scale to all 2,896 dev queries.
- Second experiment only if resources permit: Qwen3-Reranker-0.6B/4B/8B over
  the same candidate file with an Arabic retrieval instruction. Use sample
  gating before full reranking because 289,600 pairs with an 8B cross-encoder
  can be expensive.

Result:

- v46 used a 200-query same-query gate with `BAAI/bge-reranker-v2-m3` over the
  frozen BGE-M3 hybrid r100 candidates. It improved nDCG@10 from `0.793838` to
  `0.799917`, a `+0.006079` absolute gain, so the ablation was scaled.
- v47 completed the full 2,896-query Arabic MIRACL dev split. The reranked run
  scored nDCG@10 / main score `0.817731`, MAP@10 `0.756480`, MRR@10 `0.835130`,
  recall@10 `0.908944`, and hit-rate@10 `0.958564`.
- The full v47 same-query gain over the frozen BGE-M3 hybrid r100 first stage is
  `+0.016621` nDCG@10 (`0.817731` vs `0.801110`), with 289,600 rerank pairs in
  `1069.6s` on the L20.
- This is now the strongest completed Arabic MIRACLRetrieval dev result in this
  repository. It is still a retrieval-system result, not a standalone embedding
  model claim, because it depends on a cross-encoder final reranker.

Current decision:

- Use the v47 reranked ordering as the best available teacher candidate for a
  trainable Arabic embedding student.
- v48 tested `Qwen/Qwen3-Reranker-0.6B` through the official
  sentence-transformers CrossEncoder path on the same 200-query gate. It scored
  only `0.705272` nDCG@10 versus the frozen first-stage `0.793838`, so it was
  rejected.
- v49 then tested the model-card transformers causal yes/no logits path with
  the official English retrieval instruction. It improved to `0.758853`, but
  still trailed the same first-stage score by `-0.034985` nDCG@10.
- Do not scale Qwen3-Reranker-0.6B to full dev. Use v47 BGE-reranker-v2-M3 as
  the current best reranker teacher. A Qwen3-Reranker-4B gate is optional, but
  the more direct next step is student distillation from the v47 teacher.

2026-06-05 source refresh:

- BGE-M3 documentation again supports the hybrid route: the model exposes dense,
  sparse, and multi-vector scoring, and its intended final ranking can combine
  all three scores. Source: https://bge-model.com/bge/bge_m3.html
- BGE fine-tuning documentation treats `pos_scores` and `neg_scores` as
  first-class knowledge-distillation inputs. Source:
  https://bge-model.com/tutorial/7_Finetuning/7.1.2.html
- SentenceTransformers' `MarginMSELoss` supports teacher-score margin
  distillation over `(query, positive, negative)` triples, but our v39/v40
  evidence says dense-only BGE-M3 fine-tuning is the wrong surface for a hybrid
  teacher unless the student/evaluator is also dense-only. Source:
  https://sbert.net/docs/package_reference/sentence_transformer/losses.html
- MIRACL official metadata confirms Arabic is large enough to be meaningful for
  train/dev iteration: `2,061,414` Arabic passages, `3,495` train queries, and
  `2,896` dev queries. Source: https://github.com/project-miracl/miracl
- Qwen3-Embedding-8B remains a mandatory public reference because its model card
  reports the No.1 MTEB multilingual position at release, but our same-task
  dense and reranker gates still point to BGE-M3 hybrid plus BGE reranker as the
  better immediate Arabic MIRACL route. Source:
  https://huggingface.co/Qwen/Qwen3-Embedding-8B

Reranker-scored train teacher data:

- v50 rescored the full v38 MIRACL Arabic train teacher JSONL with
  `BAAI/bge-reranker-v2-m3` through the same sequence-classification backend
  that produced the v47 dev gain.
- Input: official MIRACL Arabic train-derived v38 rows, not dev rows. This
  avoids using dev labels or dev text as training supervision.
- Scale: `3,495` train rows, `4,949` positive pairs, `27,960` negative pairs,
  and `32,909` total reranker-scored pairs.
- Runtime: `130.4s` on the L20, `252.4` pairs/s, with batch size `32`.
- Mean reranker positive score: `6.251401`; mean negative score: `0.751260`;
  mean positive-minus-negative score: `+5.500141`.
- Best-positive-minus-best-negative mean margin: `+2.466470`.
- Quality caveat: `2,653 / 3,495` rows (`75.9%`) have the best positive above
  the max selected hard negative. The remaining `24.1%` are hard/ambiguous
  rows and should be filtered, downweighted, or used as pairwise ranking
  challenges instead of blindly treated as clean positives.
- The hybrid-margin versus reranker-margin Pearson correlation is `0.829177`,
  so the teacher surfaces broadly agree, but the reranker produces a much
  sharper supervised signal for distillation.

Current decision after v50: treat the generated v50 JSONL as the strongest
available local training target. The next smoke should not repeat generic
dense-only BGE-M3 training on all rows. Use either high-confidence v50 margins
for a compact dense student, or a custom BGE-M3 sparse/ColBERT head-only
distillation loop that preserves the hybrid retrieval surface. In either case,
validate first on a fixed MIRACL Arabic dev sample and only scale if base
performance is not degraded.

Filtered dense-only smoke:

- Source refresh before launch checked SentenceTransformers hard-negative
  mining and MarginMSE documentation plus BGE-M3 M3 source docs. The relevant
  implication was to filter hard negatives by margin and avoid raw cross-encoder
  logit scales as direct labels.
- v51 prepared `512` high-confidence rows from the v50 teacher JSONL. Filters:
  `min_best_margin=1.0`, `min_all_margin=0.0`, four hard negatives per query.
  The selected rows were extremely high confidence: mean best margin `7.958442`,
  mean all-positive-above-max-negative margin `6.632727`.
- v51 transformed reranker margins with `tanh(margin / 4.0)`, producing target
  margins with mean `0.972301`, min `0.907761`, and max `0.999683`.
- Training used dense-only `SentenceTransformer("/home/hhai/hf-models/BAAI-bge-m3")`,
  `MarginMSELoss`, batch size `4`, one epoch, and a conservative learning rate
  `5e-6`.
- Environment note: the first v51 training attempt failed before training
  because the earlier v42 Deepspeed probe left `deepspeed` installed. The
  remote host has no CUDA toolkit path, so `accelerate` importing Deepspeed
  raised `CUDA_HOME does not exist`. Removing Deepspeed from the remote
  virtualenv restored the generic trainer.
- Training completed in `157.3s` with `train_loss=0.344934`.
- Evaluation used a deterministic `200` query / `5000` document
  positive-plus-negatives MIRACL Arabic dev sample. This is still not a full
  MIRACL result, but it is larger than the v39/v40 smoke sample.
- Base BGE-M3 dense score: `0.976015` nDCG@10. v51 final checkpoint:
  `0.586807` nDCG@10, a collapse of `-0.389208`.

Current decision after v51: dense-only full-encoder BGE-M3 training is rejected
again, now decisively. The failure does not invalidate v50 teacher data; it
shows the student surface is wrong. Next, implement a custom head-only BGE-M3
distillation smoke: freeze the XLM-R encoder, train only `sparse_linear` and
`colbert_linear`, load the head checkpoint into the hybrid scorer, and compare
against base BGE-M3 hybrid on an identical MIRACL Arabic sample before any full
run. Do not upload or publish the v51 checkpoint.

Head-only BGE-M3 sparse/ColBERT distillation:

- Source refresh on 2026-06-05 rechecked BGE-M3 official documentation and
  source: BGE-M3 is explicitly dense+sparse+multi-vector, and the M3 model
  exposes separate `sparse_linear` and `colbert_linear` heads. MIRACL remains a
  high-signal public retrieval surface because the official paper uses
  nDCG@10/Recall@100, and Qwen3 remains a mandatory high-end public reference.
- `scripts/train_bge_m3_head_distill.py` implements the custom smoke path:
  freeze the encoder, keep dense scores fixed, train only sparse/ColBERT heads,
  and save only a head checkpoint in remote outputs. The checkpoint is not
  committed.
- `scripts/rerank_miracl_candidates_bge_m3_hybrid.py` evaluates trained heads
  by re-scoring the identical first 200 queries from the frozen BGE-M3 hybrid
  r100 top-100 candidate run. The base same-candidate score is `0.788829`
  nDCG@10.
- v52 used 128 rows, unscaled v50 reranker margins, learning rate `1e-4`, and
  no head anchor. It trained, but the sample score collapsed to `0.707574`
  nDCG@10 (`-0.081254`). The target scale was too aggressive.
- v53 used 128 rows, margin scale `0.1`, learning rate `1e-5`, and head L2
  anchor `1.0`. It scored `0.789181` nDCG@10, only `+0.000352` over base.
- v54 expanded the conservative v53 setup to 512 rows / 2048 triples. It
  scored `0.787002` nDCG@10 (`-0.001827`), so the v53 gain is not robust.

Current decision after v52-v54: the custom head-only path is technically viable
but not yet a robust model improvement. Do not scale v52, v53, or v54 to full
MIRACL dev and do not publish their checkpoints. The strongest completed
Arabic MIRACL route remains the v47 retrieval system: BGE-M3 hybrid r100
candidate generation plus `BAAI/bge-reranker-v2-m3`. Any next student attempt
needs a stronger validation loop and less brittle target construction before
using more GPU time.

Success criterion:

- Same-query sample gain must exceed random shuffle noise before scaling.
- Full dev gain must beat the current `0.801110` official-format BGE-M3 hybrid
  r100 score before it is described as a better retrieval system.
- A score blend must improve a held-out half of dev before it is treated as a
  real system improvement. v56 satisfies this with holdout `+0.013449` nDCG@10
  versus reranker-only.
- Any "Arabic embedding model SOTA" claim still requires a trained embedding
  checkpoint and broader ArabicMTEB/MTEB-family validation, not only a
  reranker-based retrieval run.

## 2026-06-04 BGE-M3 Hybrid Plan

Decision: pivot from large dense-only baselines to BGE-M3 hybrid
dense+sparse+multi-vector evaluation.

Rationale:

- The completed dense Arabic MIRACL table already picked a strong retrieval
  family: BGE-M3 beats E5, Qwen3-Embedding-8B, and Arabic Triplet on the same
  full corpus and metric.
- Qwen3-Embedding-8B was expensive on the L20 and did not beat BGE-M3 in direct
  dense mode. More dense-only model chasing is unlikely to be the fastest route
  to a top Arabic retrieval result.
- BGE-M3's official design includes sparse lexical matching and ColBERT
  interaction. Dense-only MTEB evaluation is therefore a lower-bound baseline,
  not the model's intended retrieval ceiling.

Implementation:

- `scripts/run_miracl_bge_m3_hybrid.py` performs full-corpus dense+sparse
  candidate generation on MIRACL Arabic.
- It keeps only top candidates per query in memory, avoiding raw dataset,
  checkpoint, embedding, or multi-vector cache commits.
- It reloads candidate texts and reranks query-document pairs with
  `BGEM3FlagModel.compute_score` using dense+sparse+ColBERT weights.
- The first full run uses `candidate_top_k=50`, `rerank_limit=50`,
  `max_length=256`, and `max_passage_length=256`. If it beats the dense
  BGE-M3 baseline, the next ablation should test `rerank_limit=100` or a small
  weight sweep.

Success criterion:

- Beat dense BGE-M3 full Arabic MIRACL nDCG@10 `0.785230` on the same `ar`
  subset, split, and metric.
- Keep the public claim narrow: "BGE-M3 hybrid improves over our dense BGE-M3
  MIRACL Arabic baseline" unless same-task public references show a broader
  SOTA-level claim.

Result:

- v33 completed the first full BGE-M3 hybrid run with `candidate_top_k=50`,
  `rerank_limit=50`, and dense/sparse/ColBERT weights `[0.4, 0.2, 0.4]`.
- Main score / nDCG@10: `0.800791`.
- This beats dense BGE-M3 `0.785230` by `+0.015561` on the same MIRACL Arabic
  dataset revision, subset, split, and top-k metric.
- It also beats E5 by `+0.036231`, Qwen3-Embedding-8B direct dense by
  `+0.100626`, and Arabic Triplet by `+0.187721` nDCG@10.
- Runtime was about `1.07h`: `3352.7s` for dense+sparse candidate generation
  over 2,061,414 corpus documents and `500.3s` for 144,800 hybrid rerank pairs.

Current decision: BGE-M3 hybrid r50 is the strongest completed full Arabic
MIRACL retrieval result in this repo. Run r100 next to test whether a larger
candidate set improves or saturates the result. If r100 is stable or better,
run a small weight ablation before using hybrid BGE-M3 as the retrieval teacher.

Follow-up r100 result:

- v34 doubled the candidate and rerank depth to `candidate_top_k=100` and
  `rerank_limit=100` with the same `[0.4, 0.2, 0.4]` weights.
- Main score / nDCG@10: `0.801025`.
- This is the new best completed full Arabic MIRACL retrieval result in the
  repo, but it only improves over r50 by `+0.000234` nDCG@10.
- Rerank cost roughly doubled from 144,800 to 289,600 query-document pairs and
  from `500.3s` to `1012.7s`.

Current decision after r100: candidate depth is close to saturated. Do not run
r200 before testing weights. The next experiment should be a small
dense/sparse/ColBERT weight ablation, starting with a ColBERT-heavier setting
such as `[0.3, 0.2, 0.5]`, preferably at r50 to keep runtime controlled unless
the final teacher config requires r100.

Official-format MIRACL package:

- v43 generated official EvalAI-style TREC run files for Arabic dev/test-a with
  the frozen BGE-M3 hybrid r100 `[0.4, 0.2, 0.4]` configuration.
- `ar_dev.txt` validates structurally with 2,896 queries, 100 hits per query,
  and 289,600 lines. Recomputing from the run file gives main score /
  nDCG@10 `0.801110`, MAP@10 `0.738021`, MRR@10 `0.819846`, recall@10
  `0.893223`, and hit-rate@10 `0.948895`.
- `ar_test-a.txt` validates structurally with 936 queries, 100 hits per query,
  and 93,600 lines. Test-a has no local qrels in this workflow, so no local
  score is available.
- The zip contains exactly `miracl_submission/ar_dev.txt` and
  `miracl_submission/ar_test-a.txt`.
- EvalAI challenge 1881 currently reports `is_active=false` and
  `is_frozen=true`, so the artifact is prepared and validated but not uploaded.
  Treat this as a ready-to-submit retrieval-system package, not a formal
  leaderboard rank.

ColBERT-heavy ablation:

- v35 tested r50 with weights `[0.3, 0.2, 0.5]`.
- Main score / nDCG@10: `0.800816`.
- This is effectively tied with default r50 (`+0.000026`) and below default
  r100 (`-0.000209`).
- Hit rate improved slightly to `0.950276`, but MAP and MRR were lower than the
  default r50/r100 runs, so the main ranking signal did not improve.

Current decision after v35: increasing ColBERT weight alone is not the missing
gain. Test whether lexical sparse matching is underweighted next, using a
sparse-heavy r50 setting such as `[0.4, 0.3, 0.3]`. If that does not improve,
default r100 or default r50 is sufficient as the teacher scoring setup.

Sparse-heavy ablation:

- v36 tested r50 with weights `[0.4, 0.3, 0.3]`.
- Main score / nDCG@10: `0.798118`.
- This is worse than default r50 by `-0.002673` and worse than default r100 by
  `-0.002907`, while still beating the dense-only BGE-M3 baseline by
  `+0.012888`.

Current decision after v36: lexical sparse is not underweighted in the default
mix; increasing sparse weight is harmful on this MIRACL Arabic setup. Run one
final dense-heavy sanity check `[0.5, 0.2, 0.3]`. If it does not beat default
r100 `0.801025`, stop weight search and use default BGE-M3 hybrid r100 as the
teacher scoring setup.

Dense-heavy ablation:

- v37 tested r50 with weights `[0.5, 0.2, 0.3]`.
- Main score / nDCG@10: `0.800583`.
- This is below default r50 by `-0.000208` and below default r100 by
  `-0.000442`, while still beating dense-only BGE-M3 by `+0.015353`.

Current decision after v37: stop BGE-M3 hybrid weight search for now. The
default r100 `[0.4, 0.2, 0.4]` setup remains the best tracked full Arabic
MIRACLRetrieval result in this repo and should be used as the retrieval teacher
for Arabic hard-negative and teacher-score data construction.

Teacher-data construction start:

- Use `scripts/build_miracl_bge_m3_teacher_data.py`.
- Source split: MIRACL Arabic `train`.
- Query/qrel source: official `miracl/miracl`, because the MTEB task mirror
  used for evaluation exposes only `dev`.
- Corpus source can reuse the cached MTEB MIRACL Arabic corpus; document ids
  correspond to the same MIRACL Arabic passage collection.
- Teacher: BGE-M3 hybrid default r100 `[0.4, 0.2, 0.4]`.
- Output: generated JSONL with query text, positive passages, hard negatives,
  teacher scores, doc ids, and negative-source labels.
- Policy: do not commit JSONL teacher data, raw corpus text, model snapshots,
  embeddings, or cache files.

Teacher-data construction result:

- v38 completed the full MIRACL Arabic `train` build with BGE-M3 hybrid r100
  `[0.4, 0.2, 0.4]`.
- Corpus count: `2,061,414`; train queries: `3,495`; positive qrels: `6,217`;
  judged-negative qrels: `19,165`.
- Rows written: `3,495`; skipped queries: `0`; missing positive doc ids: `0`.
- Selected negatives: `27,960`, split between `12,889` judged negatives and
  `15,071` unjudged candidate negatives.
- Mean positive teacher score: `0.608119`; mean negative teacher score:
  `0.527708`; mean margin: `+0.080411`.
- Runtime on the L20 was `4922.7s`: `3603.8s` for full-corpus candidate
  generation and `1318.9s` for teacher scoring `349,546` pairs.
- The generated `teacher_train.jsonl` is raw derived training data and remains
  in remote `remote_outputs`; only the small summary is committed.

Current decision after v38: superseded by v39/v40/v50. Plain dense-only training
smokes regressed, and v50 now provides a stronger reranker-scored teacher
surface. Do not claim SOTA from teacher-data construction alone; it is the
input to the training step, not a benchmark result.

Training smoke result:

- v39 trained `BAAI/bge-m3` on 64 rows sampled from the v38 teacher JSONL with
  plain `MultipleNegativesRankingLoss`, batch size `4`, one epoch, and no
  Matryoshka wrapper.
- The first attempt exposed a dependency mismatch:
  `sentence-transformers==5.5.1` with `transformers==4.44.2` failed before
  training because `Trainer` did not accept `processing_class`. Upgrading the
  remote virtualenv to `transformers==4.57.6` fixed the training stack.
- Training completed in `15.55s`, with `train_loss=0.515743`.
- A fixed MIRACL Arabic dev diagnostic sample of 50 queries and 1000 documents
  scored base BGE-M3 at `0.985402` nDCG@10 and the v39 smoke checkpoint at
  `0.976880`, a delta of `-0.008523`.
- This sample is intentionally not a full benchmark result, but it is a useful
  go/no-go signal: do not scale plain MNR self-training from this setup.

Current decision after v39: implement a teacher-score-aware objective before
larger training. The next smoke should use `pos_scores` and `neg_scores`
directly, for example by expanding each row into query/positive/negative
triples with target teacher margins and training with `MarginMSELoss`, or by
switching to the official FlagEmbedding BGE-M3 distillation path.

Teacher-score smoke result:

- The training script now supports `loss: margin_mse`. In this mode, each
  teacher row expands into query/positive/negative triples with label
  `pos_score - neg_score`.
- v40 used the same first 64 v38 teacher rows and expanded them into 512
  MarginMSE triples.
- Training completed in `44.52s`, with `train_loss=0.001281`.
- The fixed MIRACL Arabic dev diagnostic sample scored the v40 checkpoint at
  `0.973496` nDCG@10 versus base BGE-M3 `0.985402`, a delta of `-0.011906`.
  It also trailed the v39 plain MNR smoke by `-0.003383`.

Current decision after v40: stop generic dense-only SentenceTransformer
fine-tuning of BGE-M3 for now. The smoke result likely reflects a mismatch
between the target teacher and the student surface: the teacher scores came
from BGE-M3 hybrid dense+sparse+ColBERT retrieval, while the saved
SentenceTransformer checkpoint is evaluated as dense-only. The next serious
training branch should either use FlagEmbedding's BGE-M3 finetuning path so
dense, sparse, and ColBERT behavior are trained together, or train a separate
dense student with a dense-only teacher/evaluation target.

Official FlagEmbedding M3 unified KD smoke:

- Source refresh on 2026-06-04 checked the BGE-M3 model card, the
  FlagEmbedding fine-tuning examples, the fine-tuning README, and the BGE-M3
  paper. The important implication is clear: BGE-M3's strongest behavior is
  not just dense embedding. It is a dense+sparse+multi-vector retrieval model,
  and its training recipe uses self-knowledge distillation across retrieval
  modes.
- The official FlagEmbedding data format matches the v38 teacher file:
  `query`, `pos`, `neg`, optional `pos_scores`, and optional `neg_scores`.
  The installed M3 trainer also exposes the right levers:
  `knowledge_distillation`, `unified_finetuning`, `use_self_distill`, and
  `fix_encoder`.
- v41 attempted this path on the remote L20 using `FlagEmbedding==1.3.5`,
  `torch==2.6.0+cu124`, `transformers==4.57.6`, and `accelerate==1.13.0`.
  Plain python startup failed because the dataset loader expects an initialized
  distributed process group. Switching to `torchrun --nproc_per_node=1`
  fixed startup.
- The 64-row smoke then exposed a bad `sub_batch_size=3` interaction with
  ColBERT tensors of different sequence lengths. Removing sub-batching got the
  run to the first training step, but it exited with SIGSEGV. A minimized
  16-row, 1-negative, shorter-text, bf16, no-self-distill run also reached the
  first step and then exited with SIGSEGV.

Current decision after v41: do not scale the installed official M3 trainer in
this environment. The method is still the right research direction, but this
package stack is not stable enough for production training. Next viable paths:
create an isolated official FlagEmbedding finetune environment with compatible
dependencies and Deepspeed, or implement a custom head-only BGE-M3 distillation
loop that avoids the unstable Trainer path and validates on the same hybrid
MIRACL sample.

Deepspeed environment probe:

- The official BGE finetuning tutorial says to install `FlagEmbedding` with the
  `finetune` extra and shows `torchrun` plus a Deepspeed config. The README
  confirms the same training data shape we already have: `query`, `pos`,
  `neg`, and optional teacher `pos_scores` / `neg_scores`.
- v42 installed `deepspeed==0.19.1` into the remote virtualenv and added a
  reproducible `configs/ds_stage0.json`.
- The remote has an NVIDIA L20 with CUDA 12.4 runtime and driver, but no CUDA
  toolkit path and no `nvcc`. Deepspeed import fails before training with
  `CUDA_HOME does not exist, unable to compile CUDA op(s)`. Setting
  `DS_BUILD_OPS=0` does not bypass this compatibility check.

Current decision after v42: do not spend the next research round on CUDA
toolkit or container setup unless official FlagEmbedding training becomes the
only remaining path. The higher-leverage next experiment is custom BGE-M3
head-only distillation: freeze the encoder, keep the dense representation
fixed, train only `sparse_linear` and `colbert_linear`, optimize against both
MIRACL Arabic qrel labels and the v38 hybrid teacher distribution, then
evaluate using the same full MIRACL Arabic hybrid scorer. This preserves the
surface that currently wins (`0.801025` nDCG@10) and avoids the unstable
Trainer/Deepspeed path.

Immediate methodological correction: after the currently running expanded
baseline finishes, run a prompted/registry-controlled evaluation for E5 and
Qwen using MTEB's model registry or explicit query prompts. Treat the current
bare `SentenceTransformer` retrieval numbers as lower-bound references.

## Training Hypothesis

Best first recipe:

1. Do not train until the expanded baseline and prompted/registry-controlled
   E5/Qwen evaluation are complete.
2. If pure Arabic STS remains the target, start from the Arabic Matryoshka/GATE
   family. If retrieval and cross-lingual QA are the target, start from E5 or
   BGE-M3 depending on the expanded benchmark.
3. Use Arabic query-positive pairs from QA, retrieval, paraphrase, NLI, and
   translated instruction-retrieval data.
4. Mine in-batch and hard negatives using a strong teacher selected per task
   family, not one global teacher.
5. Train with MultipleNegativesRankingLoss, optionally wrapped with Matryoshka
   dimensions `[1024, 768, 512, 256]`.
6. Re-evaluate against all baselines using the same MTEB/ArabicMTEB task list,
   prompts, splits, metrics, and model revisions.

## First Smoke Run

Kaggle kernel `likevin2005/arabic-embedding-sota-lab` completed memory-safe
smoke runs on 2026-06-01. The runs used MTEB `2.12.30`, `batch_size=1`, and
`max_seq_length=256` on ArEntail, STS17, and STS22.v2. Derived summaries are
stored in `experiments/`; raw Kaggle outputs are intentionally not committed.

| Model | Task | Primary Score | Arabic Subset Score(s) |
| --- | --- | ---: | --- |
| `BAAI/bge-m3` | ArEntail | 0.891028 | default=0.891028 |
| `BAAI/bge-m3` | STS17 | 0.796679 | ar-ar=0.807459, en-ar=0.694256 |
| `BAAI/bge-m3` | STS22.v2 | 0.687026 | ar=0.628047 |
| `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` | ArEntail | 0.922311 | default=0.922311 |
| `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` | STS17 | 0.299716 | ar-ar=0.853052, en-ar=0.258296 |
| `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` | STS22.v2 | 0.345622 | ar=0.642056 |
| `Qwen/Qwen3-Embedding-0.6B` | ArEntail | 0.771287 | default=0.771287 |
| `Qwen/Qwen3-Embedding-0.6B` | STS17 | 0.785389 | ar-ar=0.753294, en-ar=0.701115 |
| `Qwen/Qwen3-Embedding-0.6B` | STS22.v2 | 0.674304 | ar=0.619630 |
| `intfloat/multilingual-e5-large-instruct` | ArEntail | 0.872725 | default=0.872725 |
| `intfloat/multilingual-e5-large-instruct` | STS17 | 0.833980 | ar-ar=0.815918, en-ar=0.786442 |
| `intfloat/multilingual-e5-large-instruct` | STS22.v2 | 0.686614 | ar=0.632818 |

Aggregate read:

- Pure Arabic smoke mean: Arabic Triplet Matryoshka V2 `0.805806`, BGE-M3
  `0.775511`, multilingual-e5-large-instruct `0.773820`,
  Qwen3-Embedding-0.6B `0.714737`.
- Arabic-involving smoke mean including STS17 `en-ar`: BGE-M3 `0.755197`,
  multilingual-e5-large-instruct `0.776976`, Qwen3-Embedding-0.6B `0.711332`,
  Arabic Triplet Matryoshka V2 `0.668929`.
- Task primary-score mean across the three smoke tasks:
  multilingual-e5-large-instruct `0.797773`, BGE-M3 `0.791578`,
  Qwen3-Embedding-0.6B `0.743660`, Arabic Triplet Matryoshka V2 `0.522550`.

This supports a narrow conclusion: the Arabic-specific checkpoint is a strong
pure-Arabic reference, but it is not a robust multilingual embedding baseline.
Qwen3-Embedding-0.6B did not beat BGE-M3, E5, or Arabic Triplet on this smoke
set, so it should not be the default first fine-tuning base without more
evidence. The next run should expand the Arabic text-task benchmark for the top
references before launching a training run.

## Expanded Baseline Execution Notes

The first expanded attempt combined STS, NLI, reranking, and retrieval in one
Kaggle run. It stayed in `RUNNING` without producing downloadable v7 outputs,
so the benchmark was split to reduce risk:

- Batch 1: STS/NLI tasks only: `ArEntail`, `XNLI`, `HUMESTS22`, `STS17`,
  `STS22.v2`.
- Batch 2: reranking tasks: `NamaaMrTydiReranking`, `MIRACLReranking`.
- Batch 3: retrieval tasks: start with `SadeemQuestionRetrieval`, then add
  larger retrieval tasks only after runtime is known.

This split keeps each Kaggle version diagnosable and avoids losing all progress
to one long retrieval run.

The STS/NLI split completed as v8:

| Model | 5-Task Mean | STS Mean | NLI Mean | Arabic-Only Mean |
| --- | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3` | 0.754627 | 0.701237 | 0.834713 | 0.682070 |
| `intfloat/multilingual-e5-large-instruct` | 0.748363 | 0.712956 | 0.801473 | 0.667930 |
| `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` | 0.578534 | 0.372088 | 0.888205 | 0.716767 |

Current decision: BGE-M3 is the best unprompted five-task STS/NLI baseline;
E5 remains the strongest STS-family baseline; Arabic Triplet is the strongest
NLI/pure-Arabic baseline. Before training, run prompted or MTEB-registry
controlled evaluation for E5 and Qwen because the current harness uses bare
`SentenceTransformer` loading.

The first prompted/registry attempt combined E5 and Qwen in one v9 run. It
stayed in `RUNNING` without publishing v9 outputs, so the prompt-control check
was split again:

- v10: E5 registry wrapper only on the STS/NLI task set.
- v11: Qwen registry wrapper only if v10 completes cleanly.

The combined v9 run eventually completed before v10 could be launched:

| Model | 5-Task Mean | STS Mean | NLI Mean | Arabic-Only Mean |
| --- | ---: | ---: | ---: | ---: |
| `intfloat/multilingual-e5-large-instruct` | 0.762929 | 0.717883 | 0.830498 | 0.668478 |
| `Qwen/Qwen3-Embedding-0.6B` | 0.744820 | 0.696892 | 0.816712 | 0.678347 |

Prompt-control impact:

- E5 five-task mean improved from `0.748363` unprompted to `0.762929`
  prompted/registry-controlled.
- Qwen improved on shared smoke tasks: ArEntail `+0.063212`, STS17
  `+0.061573`; STS22.v2 moved `-0.010201`.
- Prompted E5 is now the strongest observed five-task STS/NLI overall baseline.

Current decision: run reranking and retrieval batches before training. Training
data construction should use E5/BGE as broad teachers and Arabic Triplet as an
NLI/Arabic semantic teacher.

## Reranking Execution Notes

The first reranking batch combined `NamaaMrTydiReranking` and
`MIRACLReranking` across E5, BGE-M3, and Arabic Triplet. It stayed in
`RUNNING` without publishing v10 outputs, so reranking is being split into
single-task batches:

- v11: `NamaaMrTydiReranking` only across the three tracked models.
- Next: `MIRACLReranking` only, once v11 completes or exposes a blocker.

The v11 Namaa reranking run completed:

| Model | Main Score / MAP@10 | NDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| `intfloat/multilingual-e5-large-instruct` | 0.896080 | 0.922010 | 0.896080 |
| `BAAI/bge-m3` | 0.792860 | 0.844080 | 0.792860 |
| `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` | 0.644340 | 0.731470 | 0.644340 |

Current reranking decision: E5 is the clear Namaa winner and now has the
strongest evidence as the broad Arabic baseline/teacher candidate. Run
`MIRACLReranking` next before retrieval or training-data construction.

The v12 `MIRACLReranking` run attempted all MIRACL reranking subsets across E5,
BGE-M3, and Arabic Triplet. Kaggle stopped it after it exceeded the max allowed
execution duration. The output listing still showed only the prior v11 Namaa
artifacts, so this is a workflow-size blocker rather than a model result.

The v13 correction restricted `MIRACLReranking` to the `ar` subset and ran only
`intfloat/multilingual-e5-large-instruct`. Kaggle quota was exhausted, so the
run used the approved remote L20 GPU. v14 and v15 repeated the same setup for
`BAAI/bge-m3` and
`Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2`. All three
completed:

| Model | Subset | Main Score / NDCG@10 | MAP@10 | MRR@10 | Eval Time |
| --- | --- | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3` | ar | 0.792010 | 0.737030 | 0.798404 | 2820.7s |
| `intfloat/multilingual-e5-large-instruct` | ar | 0.774970 | 0.715230 | 0.786406 | 2697.1s |
| `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` | ar | 0.673330 | 0.597860 | 0.676159 | 1558.6s |

Current MIRACL decision: BGE-M3 leads E5 on the harder Arabic MIRACL subset by
`+0.017040` nDCG@10 and Arabic Triplet by `+0.118680`, while E5 remains the
clear winner on Namaa reranking. This makes the reranking picture
task-dependent rather than a single global winner. Arabic Triplet is faster but
not a competitive broad reranking teacher on MIRACL Arabic. Move next to
retrieval batches or training-data construction with task-family teachers:
BGE-M3 for MIRACL-style retrieval/reranking, E5 for Namaa-style reranking and
prompted STS/NLI, and Arabic Triplet for Arabic semantic/NLI data.

## Retrieval Execution Notes

The v16 retrieval batch used `SadeemQuestionRetrieval` with subset `default`
across E5, BGE-M3, and Arabic Triplet. This is a compact retrieval sanity check
before larger Arabic MIRACL/MKQA retrieval runs:

| Model | Main Score / NDCG@10 | MAP@10 | MRR@10 | Eval Time |
| --- | ---: | ---: | ---: | ---: |
| `intfloat/multilingual-e5-large-instruct` | 0.696100 | 0.597790 | 0.600753 | 50.4s |
| `BAAI/bge-m3` | 0.694210 | 0.595360 | 0.592824 | 105.0s |
| `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` | 0.643610 | 0.545480 | 0.538965 | 37.3s |

Current retrieval decision: E5 narrowly wins Sadeem by `+0.001890` nDCG@10 over
BGE-M3, which is too small to pick a retrieval teacher from this task alone.
Arabic Triplet is fastest but trails by `0.052490`. Run `MIRACLRetrieval` on
the Arabic subset next; if runtime is acceptable, follow with `MKQARetrieval`
Arabic.

The v17 retrieval batch used `MIRACLRetrieval` with subset `ar` across the same
three tracked models:

| Model | Subset | Main Score / NDCG@10 | MAP@10 | MRR@10 | Eval Time |
| --- | --- | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3` | ar | 0.785230 | 0.720700 | 0.808149 | 6789.8s |
| `intfloat/multilingual-e5-large-instruct` | ar | 0.764560 | 0.695560 | 0.794536 | 3495.0s |
| `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` | ar | 0.613070 | 0.532950 | 0.640482 | 2482.3s |

Current MIRACL retrieval decision: BGE-M3 is the clear dense baseline winner on
the Arabic subset, beating E5 by `+0.020670` nDCG@10 and Arabic Triplet by
`+0.172160`. The later BGE-M3 hybrid r50 and r100 runs improved the same full
MIRACL Arabic comparison to `0.800791` and `0.801025` nDCG@10, making hybrid
BGE-M3 the strongest completed retrieval-family result in the repo. The first
weight ablation, ColBERT-heavy r50 `[0.3,0.2,0.5]`, did not materially improve
over default r50 and remained below default r100. Sparse-heavy r50
`[0.4,0.3,0.3]` was worse. This makes default hybrid BGE-M3 the current
MIRACL-style teacher candidate, pending one final dense-heavy sanity check. For
broader public claims, still search for reproducible Swan/ArabicMTEB references
before model-training claims.

The first Qwen3-Embedding-8B full Arabic MIRACL retrieval attempt was stopped
after repeated batch-size increases because it remained uneconomic on the
current L20 dense MTEB workflow. A smaller v30 diagnostic was run instead with
500 queries, a 10k corpus, all 982 positives for those queries, and random
negative documents:

| Model | Sample | Main Score / NDCG@10 | MAP@10 | MRR@10 | Recall@10 | Corpus Encode |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-Embedding-8B` | 10k corpus / 500 queries / 982 positives | 0.981477 | 0.971469 | 0.989800 | 0.990600 | 210.8s, 47.44 texts/s |

This sample score is deliberately not comparable to the full MIRACL table:
putting every selected query's positives into a 10k corpus with random negatives
makes retrieval much easier than the full 2.06M-document Arabic corpus. The
result is still useful as an engineering diagnostic. It confirms that the
Qwen3-Embedding-8B path runs cleanly, that batch size 62 fits the L20 for this
direct evaluator, and that corpus encoding speed is roughly 47.44 texts/s at
256 tokens. At that observed rate, the full Arabic corpus alone would take
about 12.1 hours to encode before full evaluator overhead.

The v32 cached full-corpus run then completed for Qwen3-Embedding-8B on the
same Arabic MIRACL retrieval split:

| Model | Evaluator | Main Score / NDCG@10 | MAP@10 | MRR@10 | Recall@10 | Corpus Encode |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-Embedding-8B` | direct dense cached exact top-k | 0.700164 | 0.623864 | 0.719286 | 0.817561 | 45686.5s, 45.12 texts/s |

This is the first completed full-corpus Qwen3-Embedding-8B Arabic MIRACL result
in the repo. It does not beat the completed dense baselines: BGE-M3 remains
higher by `+0.085066` nDCG@10 and E5 remains higher by `+0.064396`, while
Qwen3-8B is higher than Arabic Triplet by `+0.087094`. The engineering result
is still valuable: the cache workflow completed the 2.06M-document corpus in
about 12.69 hours and scored all 2,896 queries against the full corpus in
83.4s.

Current Qwen3-8B decision: do not claim Qwen3-Embedding-8B is the strongest
Arabic MIRACL dense baseline from this direct run. Because Qwen3 is
instruction-aware, the fair next Qwen-specific check is prompt/template
research and a prompt-controlled cached evaluation if the official guidance
differs from the direct `encode_queries`/`encode_corpus` path. For immediate
full MIRACL-style retrieval teacher selection, BGE-M3 hybrid is now stronger
than the completed dense baselines. r100 confirms the gain but shows candidate
depth is nearly saturated. ColBERT-heavy weighting did not improve the best
score, and sparse-heavy weighting was worse, leaving dense-heavy as the last
small ablation before freezing the teacher configuration.

## Standalone Student Distillation

The next phase is explicitly not "another retrieval stack." The goal is to
compress the strongest teacher behavior into a standalone retriever checkpoint
that can be loaded as one model and evaluated against the same MIRACL Arabic
retrieval gates.

Research refresh on 2026-06-05:

- SentenceTransformers documents `MarginMSELoss` for teacher-score margin
  distillation over `(query, passage_one, passage_two)` triples. This matches
  our reranker-scored teacher rows better than treating the teacher labels as
  binary positives and negatives.
- The BGE-M3 paper frames the model around dense, sparse, and multi-vector
  retrieval with self-knowledge distillation. This supports using the current
  BGE-M3 hybrid+rereanker teacher as a first training target rather than
  inventing a new architecture.
- FlagEmbedding's BGE-M3 implementation accepts teacher scores and exposes
  separate dense, sparse, and ColBERT losses. That remains the right scaling
  direction, but prior v52-v54 head-only experiments showed that we need a
  stronger train/held-out validation loop before trying another full BGE-M3
  unified training path.

v58 creates that loop:

| Split | Rows | Triples | Unique Queries | Mean Best Margin | Mean All Margin | Mean Target Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1000 | 4000 | 1000 | 6.429477 | 5.054223 | 0.939690 |
| held-out | 200 | 800 | 200 | 4.519829 | 3.529903 | 0.853148 |

The split source is MIRACL Arabic `train`, not dev. The script
`scripts/prepare_student_distill_splits.py` rejects non-train source rows by
default, hashes query ids into deterministic train/held-out partitions, and
fails if query ids overlap. The raw JSONL contains generated training text and
stays outside git. The committed artifact is only
`experiments/2026-06-05-student-distill-v58-setup.json`.

The queued v58 smoke uses `BAAI/bge-m3`, `MarginMSELoss`, learning rate
`1e-6`, effective batch size `16`, and one epoch. It must beat the frozen
BGE-M3 dense diagnostic before any larger student training is justified. Even
if it improves, it is still only a standalone-model candidate until it is
validated on full MIRACL Arabic dev and broader ArabicMTEB/MTEB-family tasks.

v58 completed and failed this gate:

| Model | Sample | nDCG@10 | MAP@10 | MRR@10 | Recall@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen `BAAI/bge-m3` | 200 queries / 5000 docs | 0.974790 | 0.965366 | 0.983750 | 0.983000 |
| v58 dense student | 200 queries / 5000 docs | 0.831359 | 0.791704 | 0.900250 | 0.842208 |

The delta is `-0.143431` nDCG@10. This rejects scaling or publishing v58. The
result is more controlled than v51 because the train/held-out split is cleaner
and the learning rate is lower, but it points to the same underlying issue:
plain dense-only `SentenceTransformer` MarginMSE does not preserve the
hybrid/reranker behavior we are trying to distill. The next standalone model
attempt should either fix the official FlagEmbedding BGE-M3 teacher-score path
or use a custom objective/evaluator that preserves dense, sparse, and ColBERT
signals instead of saving a dense-only checkpoint.

v59 design decision:

- Stop dense-only BGE-M3 student training until the objective changes.
- Preserve the retrieval surface that made BGE-M3 strong: dense, sparse, and
  ColBERT/multi-vector scoring.
- Use raw per-query teacher distributions instead of saturated tanh margins:
  the v58 train targets had mean `0.939690`, which leaves little ranking-shape
  information for hard negatives.
- Prefer a listwise KL/cross-entropy target over each query's positive plus
  hard negatives, with an anchor term against base BGE-M3 hybrid scores.
- Evaluate through the BGE-M3 hybrid path, not dense-only
  `SentenceTransformer.encode`.

The listwise hybrid-head objective plumbing is now implemented in
`scripts/train_bge_m3_head_distill.py` with local unit tests: it can extract raw
teacher score distributions, extract base BGE-M3 hybrid scores, train with
`listwise_kl`, and add a KL anchor against the base hybrid distribution. The
reader now also accepts raw v50 reranker-teacher rows that have
`pos_scores`/`neg_scores` but no `target_margins`; it derives margins only as
fallback bookkeeping while keeping the raw score distribution for listwise KL.
That avoids depending on the saturated v51 tanh-margin targets for v59.

The matching diagnostic gate is now wired in
`scripts/rerank_miracl_candidates_bge_m3_hybrid.py`: a checkpoint evaluation can
load a fixed baseline summary, compute same-candidate deltas, write a
`diagnostic_gate` block, and optionally fail the process with
`--require-gate-pass`. The first pass/fail gate should require at least
`+0.005` nDCG@10 over base on the v52-v54 candidate-pool diagnostic; a
v53-style `+0.000352` gain is too small to scale. This makes the next
permitted GPU step a smallest-possible v59 smoke, not another full training
run.

The first v59 smoke used raw v50 reranker teacher rows, `128` train rows,
`512` triples, `listwise_kl`, teacher temperature `6.0`, base-anchor weight
`0.5`, and head L2 anchor `1.0`. It trained cleanly, with mean loss
`0.126910`, but the same-candidate diagnostic score was only `0.788880`
nDCG@10 versus the base `0.788829`, a delta of `+0.000051`. The gate required
`+0.005`, so v59 is rejected for scaling. The useful result is negative but
actionable: raw listwise targets and anchoring prevent collapse, yet this small
head-only setup still cannot move the frozen BGE-M3 hybrid ordering enough to
justify full MIRACL dev or checkpoint publication.

## BGE Reranker v2 Gemma Full Run

v57 completed the full MIRACL Arabic dev reranking pass with
`BAAI/bge-reranker-v2-gemma` over the frozen BGE-M3 hybrid r100 candidate pool:

| System | nDCG@10 | MAP@10 | MRR@10 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| BGE-M3 hybrid r100 first stage | 0.801110 | 0.738021 | 0.819846 | 0.893223 |
| BGE-M3 hybrid r100 + BGE reranker v2-m3 | 0.817731 | 0.756480 | 0.835130 | 0.908944 |
| BGE-M3 hybrid r100 + BGE reranker v2-Gemma | 0.823758 | 0.764017 | 0.839831 | 0.912179 |
| BGE-M3 hybrid r100 + BGE reranker v2-m3 score blend | 0.828605 | 0.769687 | 0.842686 | 0.918030 |

This resolves the v55b gate: Gemma is genuinely stronger than reranker v2-m3
reranker-only on the full dev set, but it still trails the v56 score-blended
system by `0.004846` nDCG@10. Therefore v56 remains the strongest completed
retrieval-system result and the best current teacher reference. Gemma should be
kept as a teacher-diversity candidate, not promoted to the primary teacher
unless a held-out score blend or ensemble improves over v56.

## FlagEmbedding M3 No-DDP Probe

The 2026-06-05 research refresh revisited the official BGE-M3 training path
because the custom v59 head-only objective was stable but did not move the
same-candidate diagnostic enough to justify scaling. The high-signal sources
remain the BGE-M3 paper, the FlagEmbedding finetuning tutorial, and the
official M3 modeling implementation: BGE-M3's trainable surface is dense,
sparse, and multi-vector, and the official trainer supports teacher-score
distillation with `pos_scores` and `neg_scores`.

v60 created an isolated overlay environment on the remote L20 to avoid the
newest local Trainer stack while reusing the existing CUDA torch and
FlagEmbedding install:

| Package | Version |
| --- | --- |
| torch | 2.6.0+cu124 |
| FlagEmbedding | 1.3.5 |
| transformers | 4.44.2 |
| accelerate | 0.34.2 |
| datasets | 2.21.0 |
| tokenizers | 0.19.1 |

The torchrun path still failed. A dense-only KD probe under
`python -m torch.distributed.run --nproc_per_node=1` produced a faulthandler
trace in `torch.nn.parallel.DistributedDataParallel` parameter-shape
verification through Accelerate `prepare_model`. This explains why v41 and the
first v60 retry were unstable: the practical blocker is DDP initialization on
this remote binary stack, not the BGE-M3 unified objective itself.

The no-DDP wrapper in `scripts/run_flagembedding_m3_no_ddp.py` patches only
uninitialized `torch.distributed.get_rank()` and `get_world_size()` calls to
return single-process values. With that wrapper:

| Probe | Objective | Result |
| --- | --- | --- |
| v60c | official dense KD, `kl_div` | completed 1 step, loss `1.340862` |
| v60d | official unified dense+sparse+ColBERT head-only KD, `m3_kd_loss` | completed 1 step, loss `0.011697` |

This is an engineering unblock, not a model result. The next GPU step should
be a gated official M3 no-DDP head-only smoke over v50 teacher rows, evaluated
against the existing v52-v54 same-candidate diagnostic. The first scaling gate
is unchanged: at least `+0.005` nDCG@10 over the frozen BGE-M3 hybrid base.
No v60 checkpoint, raw run file, log, cache, or generated teacher data should
be committed or uploaded.

## Official M3 No-DDP Student Smokes

v61 and v62 used the recovered no-DDP FlagEmbedding path for actual gated
student smokes. Both used v50 teacher rows, froze the BGE-M3 encoder, trained
only the sparse and ColBERT heads, and evaluated on the fixed 200-query
same-candidate diagnostic against the v52 base summary.

| Run | Rows | Group | Train loss | nDCG@10 | Delta vs base | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v59 custom listwise head smoke | 128 | 5 | 0.126910 | 0.788880 | +0.000051 | fail |
| v61 official no-DDP M3 | 512 | 5 | 0.422248 | 0.790555 | +0.001726 | fail |
| v62 official no-DDP M3 | 2048 | 9 | 0.645689 | 0.792404 | +0.003575 | fail |

This is the first encouraging standalone-student signal in the project: the
official M3 trainer clearly moves the same candidate pool more than the custom
v59 head-only loop. It is still not enough. The predeclared gate requires
`+0.005` nDCG@10 before broader training, full-dev evaluation, checkpoint
publication, or any standalone-model claim. The next attempt should improve
the training design rather than blindly scale the same recipe: likely knobs are
teacher-score temperature, mode weighting, a separate tune/held-out split for
student hyperparameters, lower learning rate, and staged sparse-then-ColBERT
curriculum.

## Stratified Teacher Rows And v66/v67 Smokes

The v65 score-distribution diagnostic made the next GPU step conditional on
teacher-data redesign. The 2026-06-05 source refresh checked the BGE-M3 model
card, FlagEmbedding's M3 modeling/fine-tuning documentation, SentenceTransformers
loss documentation, and recent score-distribution KD work. The practical
conclusion was narrow: keep the official BGE-M3 dense+sparse+ColBERT training
surface, but control the row distribution before adding more GPU.

`scripts/prepare_stratified_teacher_rows.py` now builds query-disjoint teacher
splits from the v50 BGE-reranker-v2-m3 rows. It samples hard negatives
(`negative`, `0_to_0.5`, `0.5_to_1`), middle negatives (`1_to_2`, `2_to_4`),
and easy negatives (`4_plus`), preserves original teacher and BGE-M3 hybrid
scores, and can require a minimum best-positive-minus-max-negative margin. The
builder is covered by local unit tests and commits only code plus summaries,
not training text, run files, logs, or checkpoints.

v66 used a focused hard/middle/easy split with 512 train rows and 80 held-out
rows. It had clean negative coverage, but it was too adversarial: only `28.9%`
of train rows and `30.0%` of held-out rows had the teacher's best positive
above the max negative. The official no-DDP M3 head-only smoke trained for one
epoch with `lr=3e-6`, group size `5`, and loss `1.156953`. On the fixed
200-query same-candidate diagnostic, the best base surface was `0.792493`
nDCG@10 and the best student surface was `0.792837`, a best-vs-best gain of
only `+0.000343`. The best same-weight gain was larger under a weak
`sparse_heavy` surface (`+0.007092`), but that is not enough for a standalone
model gate because the strong ColBERT-heavy surface barely moved.

v67 added a qrel-consistency filter with `min_best_margin=0.05` and used a
smaller 160-row train split. This made the teacher distribution cleaner:
`100%` of train and held-out rows had the teacher's best positive above the max
negative, with mean best margins around `0.13` to `0.14`. The training loss was
lower (`0.964898`), but the diagnostic still failed: base best was `0.792493`,
student best was `0.792302`, and best-vs-best delta was `-0.000192`. The best
same-weight gain was `+0.002263` under `dense_sparse`, below the `+0.005`
gate.

This tells us two things. First, v66/v67 are useful diagnostics because they
show the student can move specific sparse or sparse+dense fusion surfaces.
Second, they are not model results: neither produces a robust standalone
checkpoint that beats frozen BGE-M3 on the strongest same-candidate surface.
The next allowed training design should be a curriculum rather than another
blind scale-up: start with qrel-consistent hard rows, add broader middle/easy
rows gradually, tune on a query-disjoint split, and require both best-vs-best
and model-card-surface gains of at least `+0.005` nDCG@10 before full-dev
evaluation or checkpoint publication.

## v68 Curriculum Distillation Design

The v68 research refresh adds three constraints from high-signal sources. First,
BGE-M3's public strength is multi-mode, so the student path must keep the
official dense+sparse+ColBERT objective and evaluate the same fusion surfaces.
Second, curriculum-learning work for dense retrieval distillation supports
controlling teacher-row difficulty rather than mixing all hard examples at
once. Third, score-distribution KD work argues against a hard-negative-only
view: the student needs a range of teacher scores and enough entropy to learn
the teacher's ordering shape.

The practical script change is `--max-best-margin` in
`scripts/prepare_stratified_teacher_rows.py`. Combined with the existing
`--min-best-margin`, this lets us build bounded margin bands for curriculum
training:

- `v68a-boundary-hard-clean`: best margin `0.05` to `0.5`, one hard, one
  middle, and two easy negatives. This is the clean version of v67's hard rows.
- `v68b-middle-score-spectrum`: best margin `0.5` to `4.0`, more middle-score
  negatives. This addresses v65's missing teacher score spectrum.
- `v68c-broad-anchor`: best margin `0.05` to `8.0`, more easy anchors and a
  lower learning rate. This should regularize the head update back toward the
  stable BGE-M3 hybrid surface.

The v68 execution result below keeps the same gate: at least `+0.005` nDCG@10
on both best-vs-best and model-card-surface diagnostics before full-dev
evaluation, scaling, checkpoint upload, or any standalone-model claim.

## v68 Curriculum Split And Smoke Result

Fresh source refresh for the execution pass rechecked the BGE-M3 model card,
the BGE-M3 technical report, the FlagEmbedding M3 modeling and fine-tuning
documentation, curriculum distillation work, and score-distribution KD work.
The conclusion stayed the same: preserve the BGE-M3 dense+sparse+ColBERT
surface, but avoid hard-negative-only training because it loses teacher-score
structure.

Remote split generation used the v50 BGE-reranker-v2-m3 teacher rows from the
MIRACL Arabic train split. The generated split summaries were:

| Split | Train rows | Held-out rows | Required mix | Train entropy @0.2 | Decision |
| --- | ---: | ---: | --- | ---: | --- |
| v68a boundary-hard-clean | 94 | 9 | hard 1 / middle 1 / easy 2 | 0.494201 | Too small |
| v68b middle-score-spectrum | 46 | 8 | hard 1 / middle 3 / easy 2 | 0.497637 | Too small |
| v68c broad-anchor | 65 | 8 | hard 1 / middle 3 / easy 4 | 0.428363 | Too small and mix-incomplete |
| v68d qrel-spectrum-g4 | 512 | 96 | middle 2 / easy 2 | 0.212741 | Small smoke allowed |
| v68e confident-spectrum-g4 | 512 | 80 | middle 2 / easy 2 | 0.212382 | Held back; redundant without a new hypothesis |

The important data lesson is that the intended hard-negative curriculum is too
sparse under the current v50 teacher rows. Once query-disjointness, qrel
consistency, bounded margins, and required hard/middle/easy counts are enforced,
the hard-negative bands produce only a few dozen usable rows. Relaxing hard
negatives gives enough rows, but the teacher targets become low-entropy and
confidence-heavy.

v68d ran the smallest official FlagEmbedding no-DDP BGE-M3 unified head-only
smoke:

- Base model: `BAAI/bge-m3`.
- Trainable parameters: `sparse_linear` and `colbert_linear`; encoder frozen.
- Rows: `512`, group size `5`, one epoch, `lr=2e-6`, temperature `0.02`.
- Training: `256` steps in `19.04s`, train loss `0.895486`.
- Evaluation: same fixed 200-query / top-100 MIRACL Arabic candidate-pool
  diagnostic, comparing base and student across identical fusion weights.

Result:

| Surface | Base nDCG@10 | Student nDCG@10 | Delta |
| --- | ---: | ---: | ---: |
| Best-vs-best (`colbert_heavy`) | 0.792493 | 0.792039 | -0.000454 |
| Model-card weights | 0.788973 | 0.789420 | +0.000447 |
| Training-default weights | 0.791008 | 0.791465 | +0.000457 |
| Dense+sparse | 0.780468 | 0.781804 | +0.001335 |
| Sparse-heavy | 0.780495 | 0.782026 | +0.001531 |

Decision:

- v68d fails the standalone-student gate. Do not scale, publish, upload, or run
  full-dev evaluation for this checkpoint.
- v68e is not worth launching as a same-family smoke without a new hypothesis:
  it is also middle/easy-only and has similarly low entropy.
- The repeated pattern across v63-v68 is now clear. Head-only BGE-M3
  sparse/ColBERT KD can repair weak sparse-heavy or dense+sparse fusion
  surfaces, but it does not move the strongest BGE-M3 surface enough to justify
  a standalone model claim.
- The next design should change a larger factor: validation-aware
  full-encoder/adapter training, a mixed v56/v57 teacher with calibrated
  score distributions, or broader Arabic benchmark tune/held-out coverage
  before another GPU run.

## v69 Blended Teacher And Full-Encoder/Adapter Design

The v69 research refresh checked the same high-signal M3 sources plus PEFT
LoRA documentation. Two details matter for the next move:

- FlagEmbedding M3 has an official `fix_encoder` training argument. That means
  a tiny full-encoder smoke is available through the recovered no-DDP path, but
  it must be guarded by very small batch size and a strict diagnostic gate.
- The official M3 path does not expose a simple BGE-M3 LoRA flag. LoRA remains
  a plausible adapter route, but it needs explicit local integration and
  trainable-parameter tests before any remote GPU run.

The more immediate v69 change is teacher-target calibration. v56 showed that a
per-query min-max blend of BGE-reranker-v2-m3 scores and BGE-M3 hybrid scores
with reranker weight `0.65` was stronger than reranker-only on both tune and
holdout dev halves. That does not make v56 a standalone model, but it is a
useful teacher-design clue: the student target should not be only sharp
reranker logits or only frozen hybrid scores.

`scripts/prepare_blended_teacher_scores.py` now implements this train-side
teacher transform. For each MIRACL train teacher row, it:

- Reads existing `pos_scores`/`neg_scores` from the reranker teacher.
- Reads `bge_m3_hybrid_pos_scores`/`bge_m3_hybrid_neg_scores`.
- Applies row-level min-max normalization to both score families.
- Writes new `pos_scores`/`neg_scores` as
  `0.65 * reranker_norm + 0.35 * hybrid_norm` by default.
- Preserves the original reranker scores and blend metadata.
- Writes only generated remote JSONL and a small summary; raw text rows are not
  committed.

`scripts/sweep_bge_m3_hybrid_weights.py` now also accepts
`--student-model-path`. This matters because a full-encoder or adapter-merged
checkpoint must be loaded as a full model, not as the older head-only
`--student-head-checkpoint` path.

The summary-only v69 blended teacher diagnostic completed on the remote L20:

| Measurement | Value |
| --- | ---: |
| Input rows | 3,495 |
| Written rows | 3,495 |
| Skipped rows | 0 |
| Positive above max negative | 0.773963 |
| Best-margin mean | 0.213304 |
| Best-margin p50 | 0.283532 |
| Entropy @ temp 1.0 | 0.978091 |
| Entropy @ temp 0.5 | 0.906094 |
| Entropy @ temp 0.2 | 0.522769 |
| Entropy @ temp 0.1 | 0.196761 |
| Mean positive probability @ temp 0.2 | 0.562496 |
| Hybrid-margin correlation | 0.864316 |

Compared with v68d, this is a much healthier teacher target: v68d had only
512 train rows, entropy `0.212741` at temperature `0.2`, and mean positive
probability `0.910569`. The blended v69 target keeps all v50 rows and is less
one-hot, while preserving positive-above-max-negative around the original v50
level.

This is still not a model result. It only passes the teacher-distribution gate
for a tiny full-encoder smoke.

Proposed first full-encoder smoke if the summary passes:

- `BAAI/bge-m3`, official no-DDP `m3_kd_loss`, `unified_finetuning=True`.
- `fix_encoder=False`, not head-only.
- `128` train rows, group size `5`, batch size `1`, gradient accumulation `8`.
- Learning-rate candidates `2e-7` and `5e-7`; temperature candidates `0.05`
  and `0.1`.
- Same fixed 200-query / top-100 MIRACL Arabic candidate-pool diagnostic.

The gate remains unchanged. A checkpoint is not eligible for full-dev
evaluation, upload, or stronger claims unless it beats frozen BGE-M3 by at
least `+0.005` nDCG@10 on both best-vs-best and model-card-surface diagnostics.

## v69a Full-Encoder Smoke Result

v69a tested the smallest useful version of the v69 idea: use the healthier
v56-style blended teacher rows, but let the BGE-M3 encoder move instead of
only training sparse/ColBERT heads.

Run configuration:

- Base model: `BAAI/bge-m3`.
- Launcher: official FlagEmbedding M3 no-DDP wrapper.
- Objective: `m3_kd_loss` with `unified_finetuning=True`.
- Trainable surface: full encoder plus M3 heads, `fix_encoder=False`.
- Train rows: `128`; group size `5`; batch size `1`; gradient accumulation
  `8`.
- LR `2e-7`; temperature `0.05`; query length `64`; passage length `128`.
- Runtime `22.0936s`; train loss `0.715388`; no OOM.

The fixed same-candidate diagnostic finished on the 200-query / top-100 MIRACL
Arabic pool:

| Surface | Base nDCG@10 | Student nDCG@10 | Delta |
| --- | ---: | ---: | ---: |
| model_card | 0.788811 | 0.788829 | +0.000018 |
| training_default | 0.791008 | 0.791410 | +0.000401 |
| dense_heavy | 0.788818 | 0.788164 | -0.000655 |
| sparse_heavy | 0.780924 | 0.780657 | -0.000268 |
| colbert_heavy | 0.792749 | 0.792493 | -0.000256 |
| dense_sparse | 0.781518 | 0.782517 | +0.000999 |
| dense_colbert | 0.786870 | 0.786868 | -0.000002 |
| sparse_colbert | 0.788258 | 0.788124 | -0.000134 |
| colbert_only | 0.786549 | 0.787183 | +0.000634 |

Gate:

- Base best: `colbert_heavy`, nDCG@10 `0.792749`.
- Student best: `colbert_heavy`, nDCG@10 `0.792493`.
- Best-vs-best delta: `-0.000256`.
- Model-card delta: `+0.000018`.
- Best same-weight gain: `dense_sparse`, `+0.000999`.
- Required gate: `+0.005` on both best-vs-best and model-card surfaces.

Decision:

- v69a fails the standalone-student gate.
- Do not scale, publish, upload, or run full-dev evaluation for this checkpoint.
- The useful result is negative: blended teacher distributions are healthier,
  and full-encoder training is technically feasible on the L20, but a tiny
  full-encoder smoke still does not move the strongest BGE-M3 fusion surface.
- Do not spend more GPU on nearby LR/temperature variants without changing a
  larger design factor. The next serious attempt should first implement either
  locally testable adapter/LoRA integration with trainable-parameter accounting
  or a stronger validation-aware objective before another remote run.

## v70 LoRA Adapter Plumbing And Target Audit

The v70 refresh checked the BGE-M3 model card, the BGE-M3 technical report,
FlagEmbedding M3 docs, the FlagEmbedding fine-tuning tutorial, and Hugging Face
PEFT LoRA docs. The conclusion is narrow: adapter training is a plausible next
larger design change, but only if it stays on BGE-M3's dense+sparse+ColBERT
surface and is gated exactly like v63-v69.

Implementation changes:

- `scripts/bge_m3_lora_utils.py` provides lightweight target preset resolution,
  linear-module discovery, and LoRA trainable-parameter estimates without
  importing torch or PEFT at module import time.
- `scripts/inspect_bge_m3_lora_targets.py` loads a BGE-M3 encoder and writes a
  small target-audit JSON before any training.
- `scripts/run_flagembedding_m3_no_ddp.py` now supports wrapper-only LoRA
  options such as `--lora-enable`, `--lora-target-preset`, `--lora-r`, and
  `--lora-report-json`. These options are stripped before forwarding to the
  official FlagEmbedding M3 CLI.
- The wrapper monkeypatches `EncoderOnlyEmbedderM3Runner.get_model()` and wraps
  the returned encoder with PEFT. This is the correct insertion point because
  FlagEmbedding creates the encoder via `AutoModel.from_pretrained(...)` and
  then attaches separate `sparse_linear` and `colbert_linear` heads.
- A narrow compatibility shim handles remote `peft 0.19.1` probing
  `torch.distributed.tensor.DTensor` when the installed torch build lacks that
  attribute.

Remote audit on the cached `BAAI/bge-m3` snapshot:

| LoRA preset | Selected modules | Est. trainable params | Encoder fraction |
| --- | ---: | ---: | ---: |
| attention_qv | 48 | 1,572,864 | 0.002770 |
| attention_qkv | 72 | 2,359,296 | 0.004155 |
| all_linear | 144 | 7,077,888 | 0.012466 |

Actual PEFT wrapping smoke:

- Preset: `attention_qv`, rank `16`, alpha `32`, dropout `0.05`,
  task type `FEATURE_EXTRACTION`.
- Resulting class: `PeftModelForFeatureExtraction`.
- Before wrapping: `567,754,752` total encoder parameters.
- After wrapping: `569,327,616` total parameters and `1,572,864` trainable
  parameters.

Decision:

- v70 is an engineering result, not a model or metric result.
- The next adapter smoke, if run, should use `attention_qv` first because it is
  the smallest validated surface. Do not jump to all-linear before a q/value
  adapter shows a real diagnostic signal.
- The first possible training smoke is v70a: v69 blended teacher rows,
  official M3 no-DDP `m3_kd_loss`, `unified_finetuning=True`,
  `fix_encoder=False`, LoRA q/value rank 16, 128 rows, group size 5, batch size
  1, gradient accumulation 8, LR no higher than `5e-5` for the first adapter
  smoke, and the same 200-query same-candidate diagnostic.
- The same gate applies: no full-dev evaluation, checkpoint upload, publication,
  or SOTA wording unless the student beats frozen BGE-M3 by at least `+0.005`
  nDCG@10 on both best-vs-best and model-card surfaces.

## v70a Attention Q/Value LoRA Smoke Result

v70a ran the first minimal adapter training attempt after the v70 plumbing:
`attention_qv` LoRA rank `16`, alpha `32`, dropout `0.05`, official
FlagEmbedding no-DDP M3 `m3_kd_loss`, `unified_finetuning=True`,
`fix_encoder=False`, v69 blended teacher rows, 128 training rows, group size 5,
batch size 1, gradient accumulation 8, learning rate `5e-5`, and temperature
`0.05`.

Two engineering fixes were needed and are now covered by tests:

- The no-DDP wrapper now imports the real `torch.distributed.tensor` module
  when available before falling back to a minimal `DTensor` placeholder. This
  avoids breaking PyTorch's own DTensor submodule imports during optimizer
  construction.
- `scripts/sweep_bge_m3_hybrid_weights.py` applies the same PEFT/DTensor
  compatibility patch before loading a full student adapter directory through
  `BGEM3FlagModel`.

Training completed on the remote L20:

| Setting | Value |
| --- | ---: |
| LoRA selected modules | 48 |
| LoRA trainable params | 1,572,864 |
| Total params after wrapping | 569,327,616 |
| Train runtime | 22.9605s |
| Train loss | 0.717402 |

The fixed 200-query MIRACL Arabic same-candidate diagnostic produced:

| Surface | Base nDCG@10 | v70a nDCG@10 | Delta |
| --- | ---: | ---: | ---: |
| Best-vs-best | 0.792749 (`colbert_heavy`) | 0.791962 (`training_default`) | -0.000787 |
| Model-card | 0.788811 | 0.791940 | +0.003129 |
| Sparse-heavy | 0.780924 | 0.787181 | +0.006257 |

Decision:

- v70a fails the standalone-student gate.
- The adapter route is now end-to-end functional, but this exact q/value
  recipe still only repairs a weaker sparse-heavy ablation surface. It does not
  move the strongest frozen BGE-M3 fusion surface.
- Do not scale, full-dev evaluate, publish, upload, or call this checkpoint
  SOTA.
- The next standalone attempt needs a larger design change: richer train rows,
  a real tune/held-out adapter sweep, qkv/all-linear LoRA only with explicit
  regularization and gates, or a stronger objective aimed directly at the
  model-card/base-best surfaces.

## v71 Validation-Aware Adapter Gate

The v71 refresh deliberately did not launch another GPU training job. The
fresh source check reinforced the same design constraint:

- The `BAAI/bge-m3` model card presents BGE-M3 as dense+sparse+multi-vector and
  recommends hybrid retrieval plus reranking for strong retrieval systems:
  https://huggingface.co/BAAI/bge-m3
- FlagEmbedding's finetuning docs support teacher-score distillation through
  `pos_scores`/`neg_scores`, `m3_kd_loss`, `unified_finetuning`,
  `use_self_distill`, and `fix_encoder` controls:
  https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
- The BGE-M3 technical report identifies self-knowledge distillation from
  different retrieval modes and efficient large-batch training as central to
  the model's strength: https://arxiv.org/abs/2402.03216
- The MIRACL official repository confirms Arabic is a high-signal retrieval
  setting: 2,061,414 passages, 3,495 train queries, and 2,896 dev queries with
  29,197 dev judgments: https://github.com/project-miracl/miracl

The practical conclusion is that v70a should not be scaled. It proved the LoRA
adapter route can train and load, but its gain was on a weaker sparse-heavy
surface while best-vs-best regressed. The next adapter run must not be selected
and judged on the same diagnostic queries.

Implemented v71 gate plumbing:

- `scripts/sweep_bge_m3_hybrid_weights.py` now supports `--query-id-file`,
  `--query-offset`, and `--query-stride`. This lets the fixed BGE-M3 hybrid
  candidate-pool diagnostic be split into deterministic query-disjoint tune and
  held-out subsets. The summary records query ids and split parameters.
- `scripts/check_student_diagnostic_gate.py` combines tune and held-out weight
  sweep summaries. It requires all of the following to clear the configured
  threshold, currently `+0.005` nDCG@10:
  tune best-vs-best, tune model-card, held-out best-vs-best, held-out
  model-card, and held-out performance for the same-weight surface selected on
  tune.
- Unit tests cover deterministic query splitting, query-id files, missing
  query rejection, and rejection of tune-only gains.

Decision:

- v71 is validation infrastructure, not a standalone model result.
- Do not full-dev evaluate, scale, upload, publish, or use SOTA wording for any
  future adapter/student checkpoint unless it passes this tune-plus-heldout
  gate first.
- The next allowed GPU smoke should be materially different from v70a, for
  example attention-qkv or all-linear LoRA with explicit regularization and the
  v69 blended teacher rows. It must run both tune and held-out sweeps and then
  `scripts/check_student_diagnostic_gate.py`; a tune-only win is not enough.

## v71a Attention QKV LoRA Smoke Result

v71a tested the first adapter run under the v71 gate:

- LoRA preset: `attention_qkv`, rank `16`, alpha `32`, dropout `0.10`.
- Selected modules: `72`; trainable parameters: `2,359,296`.
- Base/trainer: `BAAI/bge-m3`, official FlagEmbedding M3 no-DDP wrapper,
  `m3_kd_loss`, `unified_finetuning=True`, `fix_encoder=False`.
- Teacher rows: v69 blended teacher rows, first `512` rows, group size `5`.
- Learning rate: `2e-5`, temperature `0.05`, batch size `1`, gradient
  accumulation `8`.
- Training completed without OOM in `99.2877` seconds with train loss
  `0.690338`.

The first online evaluation attempt hit a Hugging Face mirror 403 while loading
an incidental `BAAI/bge-m3/imgs/.DS_Store` asset. The adapter config already
pointed at the local cached BGE-M3 snapshot, so the evaluation was rerun with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; no retraining was needed.

The v71 gate result:

| Split | Surface | Base nDCG@10 | v71a nDCG@10 | Delta |
| --- | --- | ---: | ---: | ---: |
| Tune | Best-vs-best | 0.795555 (`colbert_heavy`) | 0.793099 (`colbert_heavy`) | -0.002456 |
| Tune | Model-card | 0.791324 | 0.792786 | +0.001462 |
| Held-out | Best-vs-best | 0.789432 (`colbert_heavy`) | 0.790037 (`model_card`) | +0.000605 |
| Held-out | Model-card | 0.786623 | 0.790037 | +0.003414 |
| Held-out | Best same-weight | 0.774363 (`sparse_heavy`) | 0.780230 (`sparse_heavy`) | +0.005866 |

All five predeclared gate criteria failed:

- tune best-vs-best
- tune model-card
- held-out best-vs-best
- held-out model-card
- held-out delta for the same-weight surface selected on tune

Decision:

- v71a fails the standalone-student gate.
- Do not scale, full-dev evaluate, publish, upload, or claim this checkpoint.
- The result is still useful: attention-qkv LoRA is technically stable and
  again improves a sparse-heavy ablation, but it does not move the strong
  BGE-M3 fusion surfaces enough. The next attempt should change the objective
  or data design more substantially, rather than only expanding LoRA target
  modules.

## v72 Surface-Aware Teacher Rows And LoRA Smoke

The v72 round changed the data design rather than sweeping another nearby LoRA
target. The fresh source check stayed consistent with the earlier conclusion:

- `BAAI/bge-m3` is a dense+sparse+multi-vector retriever, and its public model
  card recommends hybrid retrieval plus reranking for strong systems:
  https://huggingface.co/BAAI/bge-m3
- FlagEmbedding's embedder finetuning path supports teacher-score KD through
  `pos_scores`/`neg_scores`, `unified_finetuning`, and `m3_kd_loss`:
  https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
- The BGE-M3 technical report emphasizes multi-function retrieval and
  self-knowledge distillation: https://arxiv.org/abs/2402.03216
- MIRACL Arabic remains a high-signal but dev-selection-sensitive benchmark:
  https://github.com/project-miracl/miracl

Implemented:

- Added `scripts/prepare_surface_aware_teacher_rows.py`.
- Added `tests/test_prepare_surface_aware_teacher_rows.py`.
- The new builder creates query-disjoint train/held-out teacher rows whose
  target scores are row-level min-max blends of reranker and BGE-M3 hybrid
  scores. It prioritizes hybrid false-positive negatives: negatives that the
  frozen BGE-M3 hybrid surface scores too close to or above the positive while
  the teacher target still separates them.

Remote v72 teacher-data diagnostic:

| Item | Value |
| --- | ---: |
| Input rows | 3,495 |
| Candidate rows with usable hybrid false positives | 425 |
| Train rows | 355 |
| Held-out rows | 70 |
| Query overlap | 0 |
| Train hybrid-false-positive negatives | 2,141 |
| Held-out hybrid-false-positive negatives | 414 |

This is intentionally narrow. It says the current v69 teacher rows contain a
real but small set of queries where the frozen hybrid surface is making the
kind of mistake we want the student to learn. It is not a scale-ready full
training corpus.

v72a then trained one small smoke:

- Base/trainer: `BAAI/bge-m3`, official FlagEmbedding M3 no-DDP wrapper.
- Adapter: attention-qkv LoRA, rank `16`, alpha `32`, dropout `0.10`.
- Trainable parameters: `2,359,296`.
- Teacher rows: v72 surface-aware rows.
- Train rows: `355`; group size: `9`.
- Learning rate: `2e-5`; temperature: `0.05`; gradient accumulation: `8`.
- Runtime: `73.8499s`; train loss: `1.202936`; no OOM.

The v71 tune/held-out gate result:

| Split | Surface | Base nDCG@10 | v72a nDCG@10 | Delta |
| --- | --- | ---: | ---: | ---: |
| Tune | Best-vs-best | 0.795555 (`colbert_heavy`) | 0.794262 (`colbert_heavy`) | -0.001293 |
| Tune | Model-card | 0.791324 | 0.793345 | +0.002021 |
| Tune | Best same-weight | 0.789425 (`sparse_colbert`) | 0.791857 (`sparse_colbert`) | +0.002433 |
| Held-out | Best-vs-best | 0.789943 (`colbert_heavy`) | 0.796421 (`sparse_colbert`) | +0.006478 |
| Held-out | Model-card | 0.786298 | 0.792526 | +0.006228 |
| Held-out | Tune-selected same-weight | 0.787092 (`sparse_colbert`) | 0.796421 (`sparse_colbert`) | +0.009329 |

Decision:

- v72a fails the strict standalone-student gate because tune best-vs-best and
  tune model-card did not clear `+0.005`.
- Do not scale, full-dev evaluate, upload, publish, or call this checkpoint
  SOTA.
- The signal is still materially better than v71a: all held-out criteria pass.
  The next design should stabilize this surface-aware signal across multiple
  query-disjoint slices before any larger run. Good options are expanding the
  surface-aware corpus, adding a second tune split, balancing model-card and
  ColBERT-heavy targets directly, or constructing train rows from more teacher
  diversity rather than scaling v72a as-is.

## v73 Multi-Split Stability Gate

The v72a result is the first student smoke with a real held-out signal, but it
is not reliable enough: one diagnostic half improved and the other did not.
v73 therefore adds a stricter split-stability gate before any further GPU
training or full-dev evaluation.

Implemented:

- Added `scripts/check_student_multisplit_gate.py`.
- Added `tests/test_check_student_multisplit_gate.py`.
- The gate accepts repeated `--split-summary label=path` arguments, selects the
  same-weight surface with the highest mean delta across supplied splits, and
  aggregates best-vs-best, model-card, and selected same-weight deltas.
- Acceptance requires each surface to satisfy both a pass-fraction threshold
  and a mean-delta threshold. The default is strict: every supplied split must
  clear `+0.005` nDCG@10 and the mean must also be at least `+0.005`.

Sanity check on existing v72a tune/held-out summaries:

| Surface | Mean delta | Min delta | Max delta | Pass fraction |
| --- | ---: | ---: | ---: | ---: |
| Best-vs-best | +0.002593 | -0.001293 | +0.006478 | 0.5 |
| Model-card | +0.004125 | +0.002021 | +0.006228 | 0.5 |
| Selected sparse+ColBERT | +0.005881 | +0.002433 | +0.009329 | 0.5 |

Decision:

- v73 correctly rejects v72a under a multi-split stability rule.
- No new GPU training was launched.
- Before another student training run, either evaluate the existing v72a
  checkpoint on more stride-based diagnostic slices or redesign the
  surface-aware teacher rows so the signal is less split-specific.

## v74 Four-Slice Stability Check

v74 executed the first option from v73: evaluate the existing v72a checkpoint
on four query-disjoint stride-4 MIRACL Arabic diagnostic slices. This launched
no new training and kept all generated summaries, run files, and checkpoints on
the remote host.

Setup:

- Student checkpoint: v72a surface-aware attention-qkv LoRA over BGE-M3.
- Candidate pool: v43 BGE-M3 hybrid r100 MIRACL Arabic dev run.
- Slices: offsets `0`, `1`, `2`, and `3`, stride `4`, 50 queries each.
- Gate: `scripts/check_student_multisplit_gate.py`, required delta `+0.005`,
  pass fraction `1.0`.

Aggregate result:

| Surface | Mean delta | Min delta | Max delta | Pass fraction |
| --- | ---: | ---: | ---: | ---: |
| Best-vs-best | +0.003868 | -0.000101 | +0.012598 | 0.25 |
| Model-card | +0.004125 | +0.000486 | +0.008286 | 0.25 |
| Selected sparse+ColBERT | +0.005881 | +0.000908 | +0.012598 | 0.5 |

Per-slice deltas:

| Slice | Best-vs-best | Model-card | Selected sparse+ColBERT |
| --- | ---: | ---: | ---: |
| slice0 | -0.000101 | +0.003557 | +0.003958 |
| slice1 | +0.002339 | +0.004171 | +0.006060 |
| slice2 | +0.000638 | +0.000486 | +0.000908 |
| slice3 | +0.012598 | +0.008286 | +0.012598 |

Decision:

- v74 rejects v72a under the four-slice stability gate.
- The selected sparse+ColBERT surface clears the mean-delta threshold, but it
  fails pass-fraction stability; best-vs-best and model-card also fail both
  mean and pass-fraction criteria.
- Do not scale v72a, run full-dev evaluation, upload the checkpoint, or make a
  standalone-model claim.
- The next standalone-student attempt must change the data/objective design
  enough to move model-card and base-best surfaces consistently across slices.

## v75 BGE-M3 Component-Level Teacher Augmentation

v75 changed the train-data observability layer after v74 rejected v72a as
split-specific. No model training was launched.

Fresh source check:

- `BAAI/bge-m3` exposes dense, sparse, and multi-vector retrieval modes and
  documents hybrid scoring: https://huggingface.co/BAAI/bge-m3
- FlagEmbedding's embedder finetuning path consumes teacher scores through
  `pos_scores` and `neg_scores`, including the M3 unified KD path:
  https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
- The BGE-M3 technical report frames the model around multi-function retrieval
  and self-knowledge distillation: https://arxiv.org/abs/2402.03216
- MIRACL remains a high-signal Arabic retrieval benchmark, but dev-set student
  claims need query-disjoint validation: https://github.com/project-miracl/miracl

Implemented:

- Added `scripts/augment_teacher_rows_bge_m3_components.py`.
- Added `tests/test_augment_teacher_rows_bge_m3_components.py`.
- The new tool scores each query-positive and query-negative pair with
  BGE-M3 `dense`, `sparse`, and `colbert` components, then materializes named
  fusion surfaces from the same weight grid used by the diagnostic sweeps.
- It records only a small summary in git; the augmented teacher JSONL remains
  remote-only.

Remote run:

| Item | Value |
| --- | ---: |
| Input rows | 3,495 |
| Written rows | 3,495 |
| Pairs scored | 32,909 |
| Runtime | 153.35s |
| Throughput | 214.61 pairs/s |
| Summary size | 76K |
| Augmented JSONL size | 58M |

Surface diagnostics from the augmented train rows:

| Surface | Pos above max neg | Rows with max neg >= pos | Mean best margin | P25 | P50 | Entropy at T=0.2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model-card | 0.749070 | 877 | 0.035059 | -0.000308 | 0.036850 | 0.985360 |
| colbert-heavy | 0.746781 | 885 | 0.035562 | -0.000463 | 0.037637 | 0.984829 |
| sparse+ColBERT | 0.731330 | 939 | 0.031279 | -0.003066 | 0.032977 | 0.986386 |
| sparse-heavy | 0.720172 | 978 | 0.027883 | -0.005695 | 0.030147 | 0.987658 |
| training-default | 0.749642 | 875 | 0.083093 | -0.000257 | 0.087411 | 0.911706 |

Interpretation:

- v72 selected by a single precomputed BGE-M3 hybrid aggregate. v75 exposes the
  actual component and named-surface scores, so the next selector can target
  the surfaces that the v73/v74 gate actually cares about.
- The train rows have hundreds of useful surface-specific hard cases. For
  example, model-card has 877 rows where at least one negative scores at or
  above the positive, and sparse+ColBERT has 939.
- This is not a standalone-model result and not a SOTA claim. It is the data
  substrate needed for a more defensible student-training attempt.

Decision:

- Do not launch training from v75 directly.
- Next implement a component-aware multi-surface selector over the augmented
  rows. It should balance model-card, colbert-heavy, sparse+ColBERT, and
  training-default false positives, produce query-disjoint tune/held-out rows,
  and only then run another small LoRA/full-encoder smoke.
- Any future checkpoint still has to pass the v73/v74 multi-split gate before
  full-dev evaluation, scaling, upload, publication, or SOTA wording.

## v76 Multi-Surface Teacher Selector

v76 implements the next data-design step from v75. No GPU training was launched.

Fresh source check was consistent with v75:

- `BAAI/bge-m3` exposes dense, sparse, and multi-vector retrieval modes:
  https://huggingface.co/BAAI/bge-m3
- FlagEmbedding's M3 finetuning path consumes `pos_scores` and `neg_scores`:
  https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
- The BGE-M3 technical report emphasizes multi-function retrieval and
  self-knowledge distillation: https://arxiv.org/abs/2402.03216
- MIRACL Arabic requires query-disjoint validation discipline before claims:
  https://github.com/project-miracl/miracl

Implemented:

- Added `scripts/prepare_multisurface_teacher_rows.py`.
- Added `tests/test_prepare_multisurface_teacher_rows.py`.
- The selector reads v75 component-augmented rows, computes row-minmax teacher
  targets from reranker scores, existing v69 blended scores, and the average of
  target BGE-M3 fusion surfaces.
- It selects negatives that are near false positives on `model_card`,
  `colbert_heavy`, `sparse_colbert`, and `training_default` while the teacher
  target still ranks the positive higher. Outputs remain standard
  `pos_scores`/`neg_scores` JSONL for the official FlagEmbedding M3 path.

Remote selector sweep:

| Variant | Surface margin | Teacher separation | Candidates | Train | Held-out | Train distinct surfaces | Held-out distinct surfaces | Train mix rows | Held-out mix rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v76 strict | 0.00 | 0.05 | 298 | 250 | 48 | 2.856 | 2.646 | 0.092 | 0.104 |
| v76b relaxed | 0.02 | 0.03 | 720 | 512 | 96 | 3.279 | 3.302 | 0.125 | 0.167 |
| v76c near-FP | 0.05 | 0.02 | 1,492 | 512 | 96 | 3.881 | 3.844 | 0.312 | 0.344 |

v76c selected surface counts:

| Split | model-card | colbert-heavy | sparse+ColBERT | training-default |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,685 | 1,639 | 1,797 | 771 |
| Held-out | 302 | 291 | 332 | 145 |

v76c target distribution:

| Split | Mean target best margin | Entropy at T=0.2 | Rows | Negative pairs |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.147215 | 0.622076 | 512 | 4,096 |
| Held-out | 0.123702 | 0.623158 | 96 | 768 |

Interpretation:

- Strict surface false positives are too narrow for a useful next training
  smoke. v76 only finds 298 candidate rows with margin `<= 0.00`.
- v76c is the best next data source because a 0.05 near-false-positive margin
  produces full query-disjoint train/held-out splits and much better surface
  coverage while preserving positive-above-negative teacher targets.
- `training_default` remains the scarcest target surface, so the next smoke
  still needs strict v73/v74 diagnostics before scaling.

Decision:

- v76 is selector/data plumbing, not a model result.
- v76c is the preferred input for the next tiny v77 smoke.
- The next run may be attention-qkv LoRA or a very small full-encoder smoke
  over v76c rows, but it must immediately run the v73/v74 multi-split gate.
- Do not full-dev evaluate, upload, publish, or claim SOTA unless that future
  checkpoint robustly improves model-card and base-best surfaces.

## v77 Multi-Surface LoRA Smoke

v77 ran the planned tiny standalone-student smoke over the v76c multi-surface
teacher rows, then evaluated it with the four-slice v73/v74 diagnostic. This was
a GPU training run, but still a smoke test rather than a publishable model.

Fresh source check was unchanged from v76:

- `BAAI/bge-m3` exposes dense, sparse, and multi-vector retrieval modes:
  https://huggingface.co/BAAI/bge-m3
- FlagEmbedding's M3 finetuning path consumes `pos_scores` and `neg_scores`:
  https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
- The BGE-M3 technical report emphasizes unified retrieval modes and
  distillation: https://arxiv.org/abs/2402.03216
- MIRACL Arabic is the current high-signal Arabic retrieval benchmark target:
  https://github.com/project-miracl/miracl

Training setup:

- Base model: `BAAI/bge-m3`.
- Trainer: official FlagEmbedding M3 no-DDP wrapper.
- Teacher rows: `remote_outputs/student-distill-v76c-multisurface-teacher/teacher_train_multisurface.jsonl`.
- LoRA: attention q/k/v, rank `16`, alpha `32`, dropout `0.10`, `72`
  selected modules, `2,359,296` trainable parameters.
- Train rows: `512`; group size: `9`; lr: `1e-5`; temperature: `0.05`;
  gradient accumulation: `8`.
- Runtime: `105.971` seconds; train loss: `1.212927`; no OOM.

Gate result:

| Criterion | Mean delta | Min delta | Max delta | Pass fraction |
| --- | ---: | ---: | ---: | ---: |
| Best-vs-best | `+0.000728` | `-0.000535` | `+0.002848` | `0.00` |
| Model-card | `+0.003081` | `+0.000486` | `+0.005714` | `0.25` |
| Selected sparse+ColBERT | `+0.004748` | `+0.000104` | `+0.014940` | `0.25` |

Per-slice deltas:

| Slice | Best-vs-best | Model-card | Selected sparse+ColBERT |
| --- | ---: | ---: | ---: |
| slice0 | `+0.000127` | `+0.001447` | `+0.003478` |
| slice1 | `+0.002848` | `+0.004680` | `+0.014940` |
| slice2 | `-0.000535` | `+0.000486` | `+0.000104` |
| slice3 | `+0.000471` | `+0.005714` | `+0.000471` |

Interpretation:

- v77 confirms that the v76c rows can move sparse+ColBERT and model-card
  surfaces locally, especially slice1 and slice3.
- It still fails every predeclared multi-split gate criterion. Best-vs-best
  never clears `+0.005`; model-card clears only one slice; selected
  sparse+ColBERT clears only one slice and misses the mean threshold.
- This is not a standalone-model result, not a full-dev result, and not a SOTA
  claim.

Decision:

- Do not scale, full-dev evaluate, upload, publish, or describe v77 as a
  standalone model.
- Do not launch a nearby larger v77 LoRA run.
- The next step should be per-query/per-surface failure analysis and a larger
  data/objective change that directly targets model-card and base-best stability
  across all stride slices.

## v78 v77 Query-Level Failure Analysis

v78 did not train a new model. It added diagnostic plumbing and reran the v77
checkpoint over the four stride slices to persist query-level metrics.

Fresh source check remained aligned with the previous design:

- `BAAI/bge-m3` exposes dense, sparse, and multi-vector retrieval modes:
  https://huggingface.co/BAAI/bge-m3
- FlagEmbedding's M3 finetuning path consumes `pos_scores` and `neg_scores`:
  https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
- The BGE-M3 technical report emphasizes unified retrieval and distillation:
  https://arxiv.org/abs/2402.03216
- MIRACL Arabic remains the high-signal same-task diagnostic benchmark:
  https://github.com/project-miracl/miracl

Implemented:

- `scripts/sweep_bge_m3_hybrid_weights.py` now supports `--write-per-query`,
  writing `per_query_metrics.jsonl` with query-level base/student metrics and
  top doc ids for each weight surface.
- `scripts/analyze_student_surface_failures.py` aggregates those JSONL files
  across split labels, reports pass/regression rates, and surfaces worst and
  best focus queries.

Remote no-training analysis:

- Student: v77 attention-qkv LoRA checkpoint.
- Candidate pool: v43 BGE-M3 hybrid r100 MIRACL Arabic dev run.
- Slices: four stride-4 splits, 50 queries each.
- Outputs remain remote-only under
  `remote_outputs/v78-v77-per-query-failure-analysis`.
- Remote output sizes: failure analysis `70K`, gate summary `10K`, per-slice
  per-query JSONL files about `342K` to `350K`.

Aggregate query-level findings:

| Surface | Queries | Mean delta | Min delta | Max delta | Pass count | Pass fraction | Regression count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model-card | 200 | `+0.003081` | `-0.130930` | `+0.184576` | 22 | `0.110` | 15 |
| sparse+ColBERT | 200 | `+0.004748` | `-0.369070` | `+0.500000` | 27 | `0.135` | 11 |
| training-default | 200 | `+0.000159` | `-0.226294` | `+0.141267` | 10 | `0.050` | 7 |
| dense+sparse | 200 | `-0.002930` | n/a | n/a | 20 | `0.100` | 16 |
| dense-heavy | 200 | `-0.002624` | n/a | n/a | 10 | `0.050` | 12 |

Split-level focus:

| Slice | model-card mean | model-card pass frac | sparse+ColBERT mean | sparse+ColBERT pass frac | training-default mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| slice0 | `+0.001447` | `0.08` | `+0.003478` | `0.12` | `+0.000318` |
| slice1 | `+0.004680` | `0.14` | `+0.014940` | `0.08` | `-0.002554` |
| slice2 | `+0.000486` | `0.08` | `+0.000104` | `0.12` | `-0.000113` |
| slice3 | `+0.005714` | `0.14` | `+0.000471` | `0.22` | `+0.002984` |

Worst focus queries:

| Surface | Query | Delta | Base | Student |
| --- | ---: | ---: | ---: | ---: |
| model-card | 10527 | `-0.130930` | `0.630930` | `0.500000` |
| model-card | 10590 | `-0.087472` | `0.777323` | `0.689851` |
| model-card | 10717 | `-0.087472` | `0.732296` | `0.644824` |
| sparse+ColBERT | 10899 | `-0.369070` | `1.000000` | `0.630930` |
| sparse+ColBERT | 10444 | `-0.173197` | `0.469279` | `0.296082` |
| sparse+ColBERT | 0 | `-0.083401` | `0.524981` | `0.441580` |

Interpretation:

- v77's mean gains are driven by relatively few query-level wins. Most queries
  do not clear `+0.005`, and several regress sharply.
- The sparse+ColBERT signal is real but unstable: it has the largest positive
  outlier (`+0.500000`) and the worst negative outlier (`-0.369070`).
- The model-card surface is closer to the actual public BGE-M3 usage, but only
  `11%` of query comparisons pass the gate threshold.

Decision:

- v78 confirms that scaling v77 would be the wrong next move.
- The next standalone-student attempt should be a v79 data/objective redesign
  that targets query-level model-card and base-best stability directly.
- Do not start another GPU run until the v79 design explains how it will raise
  pass fraction across all four stride slices, not only improve mean delta.

## v79 Failure-Aware Teacher Selector

v79 converts the v78 query-level failure analysis into a train-split data design
without using dev queries as training examples. No model was trained.

Fresh source check remained consistent:

- `BAAI/bge-m3` exposes dense, sparse, and multi-vector retrieval modes:
  https://huggingface.co/BAAI/bge-m3
- FlagEmbedding's M3 finetuning path consumes `pos_scores` and `neg_scores`:
  https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
- The BGE-M3 technical report emphasizes unified retrieval and distillation:
  https://arxiv.org/abs/2402.03216
- MIRACL Arabic remains the same-task diagnostic benchmark, but dev query ids
  from v78 must not be copied into train rows:
  https://github.com/project-miracl/miracl

Implemented:

- `scripts/prepare_multisurface_teacher_rows.py` now supports
  `--surface-average-source-weights`, so the BGE-M3 surface-average teacher
  target can weight model-card/training-default more heavily than sparse-only
  local gains.
- `scripts/prepare_failure_aware_teacher_rows.py` reads v78
  `failure_analysis.json`, computes surface risk, allocates negative-selection
  quotas, and delegates row construction to the v76 multisurface selector.

Failure risk formula:

```text
risk = (1 - pass_fraction)
       + regression_fraction
       + max(0, required_delta - mean_delta) / required_delta
weighted_risk = risk * surface_priority
```

The first default v79 plan produced quotas:

```text
model_card=2, training_default=3, sparse_colbert=1, colbert_heavy=2
```

and weighted surface-average target weights:

```text
model_card=2.157936
training_default=2.734607
sparse_colbert=0.824784
colbert_heavy=1.620022
```

Remote no-training selector sweep:

| Variant | Candidate rows | Train / held-out | Query overlap | Surface margin | Teacher sep | Train mix fraction | Train target margin mean | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v79 default | 1,327 | 512 / 96 | 0 | 0.05 | 0.02 | 0.133 | 0.1420 | Usable, but training-default coverage did not improve over v76 |
| v79b three-surface relaxed | 1,983 | 512 / 96 | 0 | 0.08 | 0.01 | 0.156 | 0.1234 | Better model-card/training-default coverage, drops colbert-heavy |
| v79c dense guard | 1,406 | 512 / 96 | 0 | 0.05 | 0.02 | 0.320 | 0.1407 | Adds dense-surface guard, less focused on model-card/training-default |
| v79d relaxed four-surface | 2,311 | 512 / 96 | 0 | 0.10 | 0.01 | 0.342 | 0.1185 | Preferred next tiny-smoke input |

v79d train selected surface counts:

| Surface | v76c count | v79d count |
| --- | ---: | ---: |
| model-card | 1,685 | 2,825 |
| training-default | 771 | 1,234 |
| sparse+ColBERT | 1,797 | 2,949 |
| colbert-heavy | 1,639 | 2,786 |

v79d held-out selected surface counts:

| Surface | Count |
| --- | ---: |
| model-card | 546 |
| training-default | 234 |
| sparse+ColBERT | 577 |
| colbert-heavy | 534 |

Interpretation:

- The default failure plan correctly shifts quota and target weights toward
  training-default/model-card, but strict margins still limit available rows.
- v79d is the best no-training input because relaxed surface margin and teacher
  separation materially increase model-card and training-default coverage while
  preserving the four-surface setting and query-disjoint train/held-out splits.
- v79d is not a model result. It only prepares a better tiny-smoke input.

Decision:

- Do not claim any standalone-model progress from v79 alone.
- v79d can be used for at most one tiny v80 smoke after the next fresh state and
  research check.
- Any v80 checkpoint must immediately run the v73/v74 four-slice gate plus v78
  query-level analysis. Reject unless model-card and best-vs-best pass fractions
  improve materially, not just mean sparse+ColBERT delta.

## v80 Failure-Aware LoRA Smoke

v80 ran the single allowed tiny GPU smoke over the preferred v79d
failure-aware teacher rows. It intentionally kept the v77 LoRA setup nearly
fixed so the main changed factor was the data selector:

- Teacher rows:
  `remote_outputs/student-distill-v79d-failure-aware-teacher/teacher_train_failure_aware.jsonl`.
- Trainer: official FlagEmbedding BGE-M3 no-DDP path through
  `scripts/run_flagembedding_m3_no_ddp.py`.
- Base: `BAAI/bge-m3`.
- LoRA: attention q/k/v, rank `16`, alpha `32`, dropout `0.10`, `72`
  selected modules, `2,359,296` trainable parameters.
- KD/loss: `m3_kd_loss`, `unified_finetuning=True`, `fix_encoder=False`,
  temperature `0.05`.
- Training: `512` rows, group size `9`, batch size `1`, gradient accumulation
  `8`, learning rate `1e-5`, one epoch.
- Runtime: `107.6007s`, train loss `1.203204`.
- Evaluation: four stride-4 MIRACL Arabic diagnostic slices over the v43
  BGE-M3 hybrid r100 candidate pool, with per-query output enabled.

Fresh source check remained consistent:

- `BAAI/bge-m3` exposes dense, sparse, and multi-vector retrieval modes:
  https://huggingface.co/BAAI/bge-m3
- FlagEmbedding's M3 finetuning path consumes `pos_scores` and `neg_scores`:
  https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
- The BGE-M3 technical report emphasizes unified retrieval and distillation:
  https://arxiv.org/abs/2402.03216
- MIRACL Arabic remains the same-task diagnostic benchmark:
  https://github.com/project-miracl/miracl

The four-slice gate failed:

| Surface | Mean delta | Min delta | Max delta | Passing slices |
| --- | ---: | ---: | ---: | ---: |
| best-vs-best | +0.001040 | +0.000090 | +0.001757 | 0 / 4 |
| model-card | +0.003157 | +0.001581 | +0.005236 | 1 / 4 |
| selected same-weight sparse+ColBERT | +0.004704 | +0.000952 | +0.013335 | 1 / 4 |

Per-slice deltas:

| Slice | best-vs-best | model-card | selected sparse+ColBERT |
| --- | ---: | ---: | ---: |
| slice0 | +0.000090 | +0.001581 | +0.002131 |
| slice1 | +0.001359 | +0.002708 | +0.013335 |
| slice2 | +0.001757 | +0.003104 | +0.002396 |
| slice3 | +0.000952 | +0.005236 | +0.000952 |

Query-level aggregate compared with v77:

| Surface | v77 mean | v80 mean | v77 pass count | v80 pass count | v77 regressions | v80 regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model-card | +0.003081 | +0.003157 | 22 / 200 | 18 / 200 | 15 | 11 |
| sparse+ColBERT | +0.004748 | +0.004704 | 27 / 200 | 23 / 200 | 11 | 12 |
| training-default | +0.000159 | -0.000342 | 10 / 200 | 8 / 200 | 7 | 8 |
| colbert-heavy | -0.000850 | +0.000151 | 11 / 200 | 11 / 200 | 9 | 9 |

Interpretation:

- v80 is an engineering success but a model failure.
- The failure-aware v79d rows slightly improved best-vs-best mean versus v77
  (`+0.000728` to `+0.001040`), but the improvement is far below the `+0.005`
  gate and has `0/4` passing slices.
- Model-card mean was effectively unchanged and query-level model-card pass
  count got worse (`22/200` to `18/200`).
- The selected surface remained sparse+ColBERT, so the apparent movement is
  still concentrated on the weaker diagnostic surface rather than the
  model-card/base-best surfaces that matter.

Decision:

- Reject v80 for scaling, full-dev evaluation, upload, publication, or any
  standalone-model claim.
- Do not repeat nearby v80 runs by simply increasing row count or changing a
  small LoRA hyperparameter.
- The next standalone attempt must change the objective or supervision shape:
  directly optimize model-card/base-best anti-regression behavior, build
  listwise rows from base-best failures, or use a broader mixed teacher that
  directly targets the BGE-M3 public fusion surface.

## v81 Anti-Regression Teacher Selector

v81 changes the supervision shape before any new GPU training. v80 proved that
surface-risk row reweighting alone was not enough: model-card query-level pass
count dropped from `22/200` to `18/200`, and the selected gain surface remained
sparse+ColBERT. The v81 design therefore adds a direct anti-regression target
source to the existing multisurface teacher-row builder.

Fresh source check remained consistent:

- `BAAI/bge-m3` exposes dense, sparse, and multi-vector retrieval modes:
  https://huggingface.co/BAAI/bge-m3
- FlagEmbedding's M3 finetuning path consumes `pos_scores` and `neg_scores`:
  https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
- The BGE-M3 technical report emphasizes unified retrieval and distillation:
  https://arxiv.org/abs/2402.03216
- MIRACL Arabic remains the same-task diagnostic benchmark:
  https://github.com/project-miracl/miracl

Implemented:

- `scripts/prepare_multisurface_teacher_rows.py` now supports optional
  `--anti-regression-surfaces`, `--anti-regression-weight`,
  `--anti-regression-margin`, and `--anti-regression-surface-hard-margin`.
- The new target source gives positives score `1.0`; negatives that are false
  positives on the configured guard surfaces get `1.0 - margin`; other
  negatives get `0.0`; this source is then blended through the same row-level
  min-max target path used by existing teacher rows.
- Defaults keep this source off, so prior experiments remain reproducible.
- `tests/test_prepare_multisurface_teacher_rows.py` now covers surface-list
  validation, anti-regression target-score behavior, and summary persistence.

Remote no-GPU selector sweep from v75 component rows:

| Variant | Candidate rows | Train / held-out | Query overlap | Anti weight | Anti margin | Train target best-margin mean | T=0.2 pos prob | T=0.2 entropy | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v81a | 2,245 | 512 / 96 | 0 | 0.50 | 0.20 | 0.1193 | 0.4173 | 0.7191 | Preferred |
| v81b | 2,153 | 512 / 96 | 0 | 0.70 | 0.25 | 0.1487 | 0.4419 | 0.7076 | Stronger margin, lower training-default coverage |
| v81c | 2,155 | 512 / 96 | 0 | 0.60 | 0.20 | 0.1477 | 0.4347 | 0.7204 | Broad coverage, still sparse+ColBERT-heavy |

v81a selected train surface counts:

| Surface | Count |
| --- | ---: |
| model-card | 2,800 |
| training-default | 1,475 |
| sparse+ColBERT | 2,993 |
| colbert-heavy | 2,763 |

v81a selected primary train surfaces:

| Surface | Count |
| --- | ---: |
| sparse+ColBERT | 1,683 |
| model-card | 557 |
| training-default | 544 |
| colbert-heavy | 275 |

Interpretation:

- v81a is the best next data input because it materially increases explicit
  model-card/training-default supervision while retaining enough broad
  multi-surface coverage.
- The target distribution is not excessively sharp: at temperature `0.2`,
  mean positive probability is `0.4173` and mean normalized entropy is
  `0.7191`.
- v81 is still only data design. It is not a model score and does not justify
  any standalone checkpoint claim.

Decision:

- Do not launch a large training run from v81.
- After a fresh state/research check, v81a can support at most one tiny v82
  smoke with the same strict four-slice gate and query-level analysis.
- Reject v82 unless model-card and best-vs-best pass fractions improve
  materially; do not accept sparse+ColBERT-only gains.

## v82 Anti-Regression LoRA Smoke

v82 ran the single allowed tiny GPU smoke over the preferred v81a
anti-regression teacher rows. It kept the v77/v80 LoRA setup fixed so the main
changed factor was the explicit model-card/training-default anti-regression
target:

- Teacher rows:
  `remote_outputs/student-distill-v81a-antiregression-teacher/teacher_train_antiregression.jsonl`.
- Trainer: official FlagEmbedding BGE-M3 no-DDP path through
  `scripts/run_flagembedding_m3_no_ddp.py`.
- Base: `BAAI/bge-m3`.
- LoRA: attention q/k/v, rank `16`, alpha `32`, dropout `0.10`, `72`
  selected modules, `2,359,296` trainable parameters.
- KD/loss: `m3_kd_loss`, `unified_finetuning=True`, `fix_encoder=False`,
  temperature `0.05`.
- Training: `512` rows, group size `9`, batch size `1`, gradient accumulation
  `8`, learning rate `1e-5`, one epoch.
- Runtime: `106.1364s`, train loss `1.296099`.
- Evaluation: four stride-4 MIRACL Arabic diagnostic slices over the v43
  BGE-M3 hybrid r100 candidate pool, with per-query output enabled.

Fresh source check remained consistent:

- `BAAI/bge-m3` exposes dense, sparse, and multi-vector retrieval modes:
  https://huggingface.co/BAAI/bge-m3
- FlagEmbedding's M3 finetuning path consumes `pos_scores` and `neg_scores`:
  https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
- The BGE-M3 technical report emphasizes unified retrieval and distillation:
  https://arxiv.org/abs/2402.03216
- MIRACL Arabic remains the same-task diagnostic benchmark:
  https://github.com/project-miracl/miracl

The four-slice gate failed:

| Surface | Mean delta | Min delta | Max delta | Passing slices |
| --- | ---: | ---: | ---: | ---: |
| best-vs-best | +0.001203 | +0.000328 | +0.003016 | 0 / 4 |
| model-card | +0.004190 | +0.001727 | +0.007083 | 1 / 4 |
| selected same-weight model-card | +0.004190 | +0.001727 | +0.007083 | 1 / 4 |

Per-slice deltas:

| Slice | best-vs-best | model-card | selected same-weight |
| --- | ---: | ---: | ---: |
| slice0 | +0.000328 | +0.001727 | +0.001727 |
| slice1 | +0.003016 | +0.004847 | +0.004847 |
| slice2 | +0.000576 | +0.003104 | +0.003104 |
| slice3 | +0.000893 | +0.007083 | +0.007083 |

Query-level aggregate compared with v77/v80:

| Surface | v77 mean | v80 mean | v82 mean | v77 pass count | v80 pass count | v82 pass count | v77 regressions | v80 regressions | v82 regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model-card | +0.003081 | +0.003157 | +0.004190 | 22 / 200 | 18 / 200 | 22 / 200 | 15 | 11 | 10 |
| sparse+ColBERT | +0.004748 | +0.004704 | +0.002842 | 27 / 200 | 23 / 200 | 22 / 200 | 11 | 12 | 12 |
| training-default | +0.000159 | -0.000342 | +0.001581 | 10 / 200 | 8 / 200 | 11 / 200 | 7 | 8 | 7 |
| colbert-heavy | -0.000850 | +0.000151 | +0.000304 | 11 / 200 | 11 / 200 | 15 / 200 | 9 | 9 | 12 |

Interpretation:

- v82 is an engineering success but still a model failure under the
  predeclared standalone-student gate.
- The useful signal is real: unlike v77/v80, the selected same-weight surface
  is now model-card rather than sparse+ColBERT, and model-card mean delta rose
  from v80's `+0.003157` to `+0.004190`.
- The signal is still too weak to scale: best-vs-best mean delta is only
  `+0.001203`, model-card is still below `+0.005`, and pass fractions are
  `0/4` and `1/4`.

Decision:

- Reject v82 for scaling, full-dev evaluation, upload, publication, or any
  standalone-model claim.
- Do not simply enlarge v82. The next attempt should keep the anti-regression
  idea but change the objective more directly: explicit pair/list constraints
  against base model-card winners, stronger model-card positive separation,
  or a validation-aware loss outside the stock M3 KD objective.
