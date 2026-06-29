# Evaluations

This directory contains isolated evaluation and evidence subprojects. They are
kept out of the main `fast_rag` application package so benchmark code can evolve
without changing the runtime web-search app.

## Subprojects

- [`claim_bench/`](claim_bench/) contains the migrated `llm-claim-bench`
  harness for auditable HumanEval/GSM8K inference runs, checked-in evidence
  summaries, and single-GPU L20 benchmark reproduction.
- [`retrieval/`](retrieval/) contains migrated retrieval benchmark snapshots for
  Chinese finance reranking, CoREB code retrieval, and Arabic
  retrieval/embedding experiments.

## Local Checks

Run the claim benchmark checks from the repository root:

```bash
PYTHONPYCACHEPREFIX=/tmp/signal-rag-claimbench-pycache \
  python -m compileall -q evals/claim_bench/src evals/claim_bench/tests
```

Retrieval subproject checks are listed in
[`retrieval/README.md`](retrieval/README.md).
