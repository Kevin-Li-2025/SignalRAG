# SignalRAG

A fast web-search RAG workbench inspired by ChatGPT Search:

- rewrites the user question into targeted search queries
- searches multiple providers concurrently
- fetches and extracts source pages in parallel
- uses a lightweight query planner before answer generation
- exposes search controls for domains, recency, locale, and citation verifier choice
- evaluates retrieval quality with a CRAG-style corrective pass
- runs multi-step retrieval in Deep Research mode
- integrates with Chromium through a local search URL and unpacked extension
- fuses multi-query search results with reciprocal rank fusion before fetching
- ranks passages with source-aware contextual BM25 signals
- packs answer context with query-aware compression and long-context reordering
- reranks passages with hybrid lexical scoring and source-quality signals
- generates a cited answer from retrieved evidence and returns claim-level citation checks
- falls back to extractive answers when no LLM API key is configured

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m fast_rag.app
```

Open http://127.0.0.1:8000.

## Chromium Integration

SignalRAG exposes a search-engine-compatible URL:

```text
http://127.0.0.1:8000/engine?q=%s&mode=pro
```

It also includes a Manifest V3 extension in
`extensions/signalrag-chromium` with:

- `sr` omnibox keyword search.
- selected-text context menu search.
- browser side panel search.
- extension options for the local API URL and default mode.

Load it from `chrome://extensions` with Developer mode and "Load unpacked".
See `extensions/signalrag-chromium/README.md`.

## Optional Accuracy Upgrades

Set these environment variables before starting the server:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_PLANNER_MODEL="deepseek-v4-flash"
export DEEPSEEK_VERIFIER_MODEL="deepseek-v4-flash"
export BRAVE_API_KEY="..."
```

Without API keys, the app still works using DuckDuckGo HTML search and an extractive cited answer.

If both DeepSeek and OpenAI keys are present, DeepSeek is used first by default. Override with:

```bash
export LLM_PROVIDER="openai"   # or deepseek / auto
```

## API

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"how does ChatGPT search work",
    "mode":"pro",
    "lens":"official",
    "max_results":10,
    "include_domains":["openai.com","help.openai.com"],
    "exclude_domains":["medium.com"],
    "recency":"year",
    "country":"us",
    "language":"en",
    "citation_verifier":"auto"
  }'
```

Modes:

- `fast`: low latency, fewer pages, shorter timeouts.
- `pro`: balanced mode for fresher, comparative, or multi-hop questions.
- `deep`: Deep Research mode. It builds several focused research steps, runs them
  in parallel, dedupes the evidence, and returns `research_trace`.

The API response includes `query_plan`, which reports the planner's inferred intent,
freshness need, search depth, and DeepSeek reasoning effort:

- `none`: thinking disabled for simple lookups.
- `high`: thinking enabled for comparisons, recommendations, API/code guidance, multi-hop synthesis, or uncertainty.
- `max`: thinking enabled with max effort only for deep research, long-horizon tasks, formal proof, or many constraints.

Deep Research mode always escalates final synthesis to `reasoning_effort: max`
and enables DeepSeek thinking mode. This matches DeepSeek's current OpenAI-format
API controls: `{"thinking":{"type":"enabled"}}` plus `reasoning_effort` of
`high` or `max`.

The request can include:

- `lens`: `web`, `official`, `academic`, `forums`, `news`, `pdf`, or `finance`.
  Lenses add intent-specific query rewrites while still respecting explicit
  domain, recency, country, and language controls.
- `include_domains` / `exclude_domains`: allowlist or denylist domains.
- `recency`: `any`, `day`, `week`, `month`, or `year`.
- `country` / `language`: two-letter locale hints for supported providers.
- `citation_verifier`: `auto`, `lexical`, or `deepseek`.

The response includes:

- `crag`: retrieval quality before and, if needed, after a corrective search.
- `research_trace`: the per-step trace used by Deep Research mode.
- `context_packing`: answer-context compression stats, including strategy,
  packed evidence count, budget, packed characters, and compression ratio.
- `claim_citations`: per-claim citation trace. With DeepSeek configured, `auto`
  uses a judge model for supported/weak/contradicted/insufficient decisions;
  otherwise it falls back to the fast lexical verifier.

## Further Improvement Roadmap

This roadmap is based on current RAG research and search-product patterns:

1. **Source lenses and search controls**: advanced search products expose
   domain/date/location/source controls and reusable lenses. SignalRAG now
   supports first-pass source lenses for web, official, academic, forums, news,
   PDFs, and finance. Next step: make lenses editable and persist custom lens
   presets in the UI.
   - Reference: [Kagi Lenses](https://help.kagi.com/kagi/features/lenses.html)
   - Reference: [Perplexity Search Filters](https://docs.perplexity.ai/docs/agent-api/filters)
   - Reference: [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)

2. **Contextual retrieval**: enrich extracted passages with source title,
   domain, and snippet context before ranking. SignalRAG now applies
   source-aware contextual BM25 signals before hybrid scoring. Next step:
   persist section/date metadata and add optional embeddings plus a cross-encoder
   reranker. Anthropic reports that contextual BM25 + contextual embeddings +
   reranking substantially reduces retrieval misses.
   - Reference: [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

3. **Stronger reranking**: keep the current fast BM25-style scorer for first
   pass, then add an optional cross-encoder or LLM reranker for Pro/Deep modes.
   Use this only after broad recall, so latency stays controlled.

4. **Query decomposition and RAG-Fusion**: generate multiple targeted queries
   for complex questions, then fuse ranked results with reciprocal rank fusion.
   SignalRAG now applies RRF before page fetching, so URLs that appear across
   several query rewrites are prioritized before extraction and passage ranking.
   Next step: add LLM-generated subquestions for high-complexity queries and
   tune fusion weights by lens/provider.
   - Reference: [LlamaIndex Query Transformations](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/query_transformations/)
   - Reference: [RAG-Fusion](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4775462)

5. **Self-reflection and correction loop**: extend the current CRAG pass so the
   model can decide whether retrieval is needed, whether evidence is sufficient,
   and whether the draft answer needs another retrieval pass.
   - Reference: [Self-RAG](https://research.ibm.com/publications/self-rag-learning-to-retrieve-generate-and-critique-through-self-reflection)
   - Reference: [Corrective RAG](https://huggingface.co/papers/2401.15884)

6. **Citation and faithfulness evaluation**: keep claim-level citation checks,
   then add offline regression metrics for context relevance, groundedness,
   answer relevance, faithfulness, and contextual recall.
   - Reference: [TruLens RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/)
   - Reference: [DeepEval RAG Evaluation](https://deepeval.com/docs/getting-started-rag)

7. **Context packing**: avoid dumping too much evidence into the final prompt.
   SignalRAG now uses query-aware extractive compression, adds source context to
   every packed passage, and reorders evidence in a "sandwich" pattern so strong
   sources appear near the beginning and end of the model context. This follows
   the same practical lesson as LongLLMLingua and lost-in-the-middle research:
   key information density and position matter, even with long-context models.
   Next step: add optional LLMLingua-style small-model compression for very large
   reports.
   - Reference: [Lost in the Middle](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long)
   - Reference: [LongLLMLingua](https://www.microsoft.com/en-us/research/project/llmlingua/longllmlingua/)

8. **Deep Research UX and reasoning**: expose a visible plan, progress trace,
   source controls, exportable report, table of contents, and source list for
   review. SignalRAG now runs deeper research steps, including countercheck and
   synthesis, and forces max-effort DeepSeek thinking for final synthesis.
   - Reference: [ChatGPT Deep Research](https://help.openai.com/en/articles/10500283-deep-research-faq/)
   - Reference: [Perplexity Sonar Deep Research](https://docs.perplexity.ai/docs/sonar/models/sonar-deep-research)
   - Reference: [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)

## Example Runs

These examples were run locally on 2026-05-12 with DeepSeek enabled,
DuckDuckGo search, and no Brave API key. They are the best demo cases because
they use official sources, produce inline citations, and exercise different
parts of the retrieval stack.

| Use case | Query | Mode | Observed result |
| --- | --- | --- | --- |
| Fast official API lookup | `DeepSeek API chat completion base URL model name and first API call` | `fast` | 2 official DeepSeek citations, 3 supported claims, ~4.0s |
| Product search explanation | `How does ChatGPT search work and how does it cite sources?` | `pro` | 2 official OpenAI citations, 3 supported claims, ~9.2s |
| API docs with source controls | `OpenAI web search API citations and domain filtering` | `pro` | OpenAI developer citation, planner chose `high` reasoning, ~12.6s |
| Deep Research trace | `Explain ChatGPT search for Enterprise and Edu data sharing and source citations.` | `deep` | 2 official OpenAI citations, 3 research steps, 5 supported claims, ~19.2s |

### Fast official API lookup

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"DeepSeek API chat completion base URL model name and first API call",
    "mode":"fast",
    "max_results":8,
    "include_domains":["api-docs.deepseek.com"],
    "recency":"year",
    "country":"us",
    "language":"en",
    "citation_verifier":"auto"
  }'
```

Why this is a strong demo: it shows the lightweight planner choosing
`reasoning_effort: none`, keeps latency low, and returns only official
DeepSeek API documentation as citations.

### ChatGPT Search citation behavior

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"How does ChatGPT search work and how does it cite sources?",
    "mode":"pro",
    "max_results":10,
    "include_domains":["openai.com","help.openai.com"],
    "recency":"year",
    "country":"us",
    "language":"en",
    "citation_verifier":"auto"
  }'
```

Why this is a strong demo: it exercises official-source prioritization,
answer citations, and claim-level verification against OpenAI Help Center and
OpenAI announcement pages.

### API docs with source controls

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"OpenAI web search API citations and domain filtering",
    "mode":"pro",
    "max_results":10,
    "include_domains":["developers.openai.com","platform.openai.com","openai.com"],
    "recency":"year",
    "country":"us",
    "language":"en",
    "citation_verifier":"auto"
  }'
```

Why this is a strong demo: it shows include-domain controls, freshness-aware
planning for API documentation, and citation grounding from developer docs.

### Deep Research trace

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"Explain ChatGPT search for Enterprise and Edu data sharing and source citations.",
    "mode":"deep",
    "max_results":12,
    "include_domains":["help.openai.com","openai.com"],
    "recency":"year",
    "country":"us",
    "language":"en",
    "citation_verifier":"auto"
  }'
```

Why this is a strong demo: it runs Deep Research mode, returns a multi-step
`research_trace`, and verifies claims across workspace policy and citation
behavior sources.

## Recall Evaluation

```bash
python -m fast_rag.eval --mode fast --top-k 5
python -m fast_rag.eval --mode pro --top-k 8
python -m fast_rag.eval --mode deep --top-k 10
```

The evaluator reports recall@k, hit rate, MRR, and latency over a small set of known-answer web-search cases.

## End-to-End Benchmark

For RAG systems, do not rely on a single benchmark number. A useful eval suite
should cover both retrieval and generation:

- Retrieval: recall@k, hit rate, MRR/nDCG, source diversity, latency.
- Generation: answer relevance, faithfulness/groundedness, citation coverage,
  supported-claim rate, contradiction rate, fallback rate, and cost/latency.
- Dataset size: use 20-50 golden queries for early development, 100-300 for a
  release gate, and 500+ mixed production traces once real usage exists. Keep
  separate slices for factual lookup, API docs, fresh/news, comparison,
  multi-hop, Deep Research, and adversarial/no-answer cases.

SignalRAG includes a small end-to-end benchmark runner:

```bash
python -m fast_rag.benchmark \
  --api-base http://127.0.0.1:8000 \
  --timeout 220 \
  --output benchmark_results/signalrag-benchmark-2026-05-12.json
```

Latest local run with DeepSeek enabled, DuckDuckGo search, and no Brave API key:

| Metric | Result |
| --- | ---: |
| Cases | 6 |
| Expected source recall | 0.8889 |
| Used source recall | 0.7222 |
| Answer term coverage | 0.9028 |
| Citation coverage | 0.5868 |
| Supported claim rate | 0.5471 |
| Review claim rate | 0.4465 |
| CRAG sufficient rate | 1.0000 |
| Fallback rate | 0.0000 |
| Average latency | 39.8s |
| P95 latency | 75.4s |

Interpretation: retrieval is already strong for official-source and API-doc
queries, and Deep Research now completes without extractive fallback. The main
quality gap is citation verifier strictness: many claims are marked for review
even when the answer has useful citations. The main latency gap is Deep Research
max-reasoning, which currently dominates p95 latency.

## Design Notes

Accuracy comes from grounding every answer in retrieved passages, checking retrieval quality before generation, fusing multi-query search results with RRF, ranking with contextual BM25 signals, packing answer context with query-aware compression, and returning only used citations by default. Speed comes from short timeouts, request concurrency, page caching, early reranking, compression before generation, and the planner choosing the cheapest mode that fits the query. For production, use a paid search API such as Brave or Tavily, add an embedding or cross-encoder reranker, and persist traces for evaluation.
