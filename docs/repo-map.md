# SignalRAG Repository Map

SignalRAG keeps the production-style web-search RAG app separate from migrated
research subprojects. This makes the public repo readable without flattening
every experiment into the main application package.

## Top-Level Areas

| Path | Role | Notes |
| --- | --- | --- |
| `fast_rag/` | Main application package | Search planning, provider routing, extraction, ranking, answer generation, and citation checks. |
| `tests/` | Main app tests | Tests for `fast_rag`, not for migrated subprojects. |
| `benchmark_results/` | Main app benchmark snapshots | JSON outputs from SignalRAG web-search benchmark runs. |
| `evals/` | Evaluation subprojects | Isolated benchmark/evidence projects with their own local assumptions. |
| `tools/` | Auxiliary retrieval tools | CodeGraph lives here because it is useful for retrieval research but not a runtime dependency. |
| `extensions/` | Browser integration | Chromium side panel, omnibox, and search-provider manifests. |
| `scripts/` | Top-level helpers | Local setup and extension helper scripts. |

## Evaluation Subprojects

| Path | Scope |
| --- | --- |
| `evals/claim_bench/` | Claim-level HumanEval/GSM8K inference evidence and single-GPU L20 benchmark reproduction. |
| `evals/retrieval/finmteb_zh/` | Chinese FinanceMTEB reranking snapshot and Qwen3 reranker experiments. |
| `evals/retrieval/coreb/` | Lightweight CoREB code-retrieval harness. Heavy data/report artifacts remain in the archived source repo. |
| `evals/retrieval/arabic/` | Arabic retrieval and embedding benchmark lab for MIRACL/MTEB-style experiments. |

## Dependency Boundary

The root `pyproject.toml` and `requirements.txt` describe the main
`fast_rag` application. Migrated subprojects keep their own `pyproject.toml`,
requirements, configs, and README files inside their subdirectories. Run their
checks from the commands listed in [`evals/README.md`](../evals/README.md) and
[`evals/retrieval/README.md`](../evals/retrieval/README.md).

## What Should Go Where

- New web-search, citation, ranking, provider, and UI code should go under
  `fast_rag/`, `tests/`, `extensions/`, or top-level `scripts/`.
- New benchmark or evidence harnesses should go under `evals/`.
- New codebase-structure retrieval utilities should go under `tools/codegraph/`
  unless they become part of the main app.
- Large generated outputs should stay out of the repo unless they are concise
  evidence snapshots needed for public reproducibility.
