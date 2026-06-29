# Future Standalone Retriever Model Card Draft

This is a placeholder model card for a future standalone Arabic retriever
checkpoint. It is not a model release. Do not publish or upload a checkpoint
until a trained student clears the repo's diagnostic gate and has a completed
benchmark table.

## Intended Use

Arabic semantic search, retrieval, clustering, classification features, and
sentence similarity on benchmark-like Arabic text.

The intended first release target is a standalone retriever that can be used
without a cross-encoder reranker. Retrieval-system results that depend on
reranking or score blending should be reported separately.

## Benchmark Requirements Before Release

The first public model claim must include:

- benchmark suite and task list;
- exact split, revision, and metric for each task;
- per-task metric table;
- aggregate score;
- compared public baselines;
- model size and embedding dimension;
- training-data summary;
- limitations and failure modes.

Minimum release gate for the current MIRACL Arabic loop:

- robust gain over frozen BGE-M3 on query-disjoint diagnostic slices;
- improvement on model-card and best-vs-best fusion surfaces, not only
  sparse+ColBERT ablations;
- per-query regression analysis;
- full-dev evaluation only after the diagnostic gate passes.

## Current Status

No standalone checkpoint is ready for publication. v58-v82 trained or evaluated
student variants, but all failed the strict diagnostic gate. The strongest
completed result remains a retrieval system, not a standalone model.

## Safety And Limitations

Embedding similarity is not semantic truth. Arabic dialect, code-switching,
religious/political content, named entities, and domain-specific language can
shift model behavior. Do not use an embedding model as the sole input for
high-stakes decisions.
