from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from .cache import PageCache
from .extract import clean_text
from .models import Document
from .search import SearchFilters, dedupe_documents, retrieve_documents


@dataclass(frozen=True)
class ResearchStep:
    label: str
    query: str
    purpose: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_research_steps(query: str, query_plan: dict | None = None) -> list[ResearchStep]:
    plan = query_plan or {}
    intent = str(plan.get("intent") or "")
    needs_freshness = bool(plan.get("needs_freshness"))
    is_deep = plan.get("reasoning_effort") == "max" or plan.get("search_depth") == "deep"
    base = clean_text(query)
    steps = [
        ResearchStep("official", f"{base} official source documentation", "ground the answer in primary sources"),
        ResearchStep("evidence", f"{base} evidence details citations", "collect concrete supporting details"),
    ]
    if needs_freshness:
        steps.append(ResearchStep("freshness", f"{base} latest 2026", "check recent or time-sensitive changes"))
    if is_deep or intent in {"comparison", "recommendation", "analysis", "api_or_code"}:
        steps.append(ResearchStep("countercheck", f"{base} limitations risks comparison", "find caveats and conflicting evidence"))
    if len(steps) < 3:
        steps.append(ResearchStep("context", f"{base} explained", "fill context that primary sources may omit"))
    if is_deep:
        steps.append(ResearchStep("synthesis", f"{base} expert analysis key findings", "find synthesis-oriented sources and missing angles"))
    return steps[:5]


async def run_deep_research(
    query: str,
    query_plan: dict,
    filters: SearchFilters,
    max_results: int,
    cache: PageCache,
) -> tuple[list[Document], dict, list[dict]]:
    steps = build_research_steps(query, query_plan)

    async def run_step(step: ResearchStep) -> tuple[ResearchStep, list[Document], dict]:
        docs, meta = await retrieve_documents(
            step.query,
            "pro",
            max(6, min(max_results, 10)),
            cache,
            filters=filters,
            page_limit_override=7,
        )
        return step, docs, meta

    batches = await asyncio.gather(*(run_step(step) for step in steps), return_exceptions=True)
    documents: list[Document] = []
    trace: list[dict] = []
    raw_results = 0
    elapsed = 0
    queries: list[str] = []
    for batch in batches:
        if isinstance(batch, Exception):
            continue
        step, docs, meta = batch
        documents.extend(docs)
        raw_results += int(meta.get("raw_results", 0))
        elapsed = max(elapsed, int(meta.get("elapsed_ms", 0)))
        queries.extend(meta.get("queries", []))
        trace.append(
            {
                **step.to_dict(),
                "documents": len(docs),
                "raw_results": meta.get("raw_results", 0),
                "top_urls": [doc.url for doc in docs[:4]],
                "elapsed_ms": meta.get("elapsed_ms", 0),
            }
        )

    deduped = dedupe_documents(documents, filters)
    return deduped, {
        "queries": _dedupe(queries),
        "filters": filters.to_dict(),
        "raw_results": raw_results,
        "deduped_results": len(deduped),
        "documents": len(deduped),
        "elapsed_ms": elapsed,
        "research_steps": len(trace),
    }, trace


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        lowered = item.lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            deduped.append(item)
    return deduped
