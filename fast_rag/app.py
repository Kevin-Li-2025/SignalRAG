from __future__ import annotations

import copy
import json
from collections import OrderedDict
from pathlib import Path
from time import perf_counter, time
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer import generate_answer
from .cache import PageCache
from .citations import (
    normalize_answer_citations,
    select_answer_evidence,
    serialize_evidence,
    verify_claim_citations,
    verify_claim_citations_with_judge,
)
from .config import settings
from .context import pack_answer_context
from .crag import assess_retrieval, should_correct
from .planner import plan_query
from .rank import rank_evidence
from .research import run_deep_research
from .search import dedupe_documents, normalize_search_filters, retrieve_documents


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title=settings.app_name)
cache = PageCache(settings.cache_path)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
RESPONSE_CACHE_TTL_SECONDS = 600
RESPONSE_CACHE_MAX_ITEMS = 96
response_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    mode: Literal["fast", "pro", "deep"] = "fast"
    max_results: int = Field(default=8, ge=3, le=20)
    lens: Literal["web", "official", "academic", "forums", "news", "pdf", "finance"] = "web"
    include_domains: list[str] = Field(default_factory=list, max_length=20)
    exclude_domains: list[str] = Field(default_factory=list, max_length=20)
    recency: Literal["any", "day", "week", "month", "year"] = "any"
    country: str | None = Field(default=None, max_length=24)
    language: str | None = Field(default=None, max_length=12)
    citation_verifier: Literal["auto", "lexical", "deepseek"] = "auto"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/engine")
async def engine() -> FileResponse:
    return FileResponse(STATIC_DIR / "engine.html")


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "openai": bool(settings.openai_api_key),
        "deepseek": bool(settings.deepseek_api_key),
        "provider": _active_provider(),
        "brave": bool(settings.brave_api_key),
    }


def _active_provider() -> str:
    if settings.llm_provider == "deepseek":
        return "deepseek" if settings.deepseek_api_key else "extractive"
    if settings.llm_provider == "openai":
        return "openai" if settings.openai_api_key else "extractive"
    if settings.deepseek_api_key:
        return "deepseek"
    if settings.openai_api_key:
        return "openai"
    return "extractive"


def _resolve_mode(requested_mode: str, planned_mode: str) -> str:
    rank = {"fast": 0, "pro": 1, "deep": 2}
    requested = requested_mode if requested_mode in rank else "fast"
    planned = planned_mode if planned_mode in rank else "fast"
    return requested if rank[requested] >= rank[planned] else planned


@app.post("/api/search")
async def search(request: SearchRequest) -> dict:
    started = perf_counter()
    cache_key = _response_cache_key(request)
    cached = _get_cached_response(cache_key)
    if cached:
        return cached

    filters = normalize_search_filters(
        request.include_domains,
        request.exclude_domains,
        request.recency,
        request.country,
        request.language,
        request.lens,
    )
    query_plan = await plan_query(request.query, request.mode)
    effective_mode = _resolve_mode(request.mode, query_plan.search_depth)
    query_plan_dict = query_plan.to_dict()
    if effective_mode == "deep" and query_plan_dict.get("reasoning_effort") == "none":
        query_plan_dict["reasoning_effort"] = "high"
    research_trace: list[dict] = []
    if effective_mode == "deep":
        docs, retrieve_meta, research_trace = await run_deep_research(
            request.query,
            query_plan_dict,
            filters,
            request.max_results,
            cache,
        )
    else:
        docs, retrieve_meta = await retrieve_documents(
            request.query,
            effective_mode,
            request.max_results,
            cache,
            filters=filters,
        )
    evidence_limit = 16 if effective_mode == "deep" else 9 if effective_mode == "pro" else 7
    evidence = rank_evidence(request.query, docs, limit=evidence_limit)
    crag_before = assess_retrieval(request.query, docs, evidence, filters)
    crag_after = None
    crag_corrected = False
    if effective_mode != "deep" and should_correct(crag_before):
        correction_mode = "pro" if effective_mode == "fast" else "deep"
        corrective_docs, correction_meta = await retrieve_documents(
            request.query,
            correction_mode,
            min(20, request.max_results + 4),
            cache,
            filters=filters,
            extra_queries=crag_before.corrective_queries,
            page_limit_override=8,
        )
        docs = dedupe_documents([*docs, *corrective_docs], filters)
        evidence = rank_evidence(request.query, docs, limit=evidence_limit)
        crag_after = assess_retrieval(request.query, docs, evidence, filters)
        crag_corrected = True
        retrieve_meta = {
            **retrieve_meta,
            "correction": correction_meta,
            "documents": len(docs),
        }
    selected_evidence = select_answer_evidence(evidence)
    packed_context = pack_answer_context(request.query, selected_evidence, effective_mode)
    answer_evidence = packed_context.evidence
    answer, answer_mode = await generate_answer(
        request.query,
        answer_evidence,
        effective_mode,
        query_plan_dict,
    )
    answer, used_citations, used_citation_ids = normalize_answer_citations(answer, answer_evidence)
    claim_citations = verify_claim_citations(answer, answer_evidence)
    if request.citation_verifier != "lexical" and (
        request.citation_verifier == "deepseek" or effective_mode in {"pro", "deep"}
    ):
        claim_citations = await verify_claim_citations_with_judge(
            request.query,
            claim_citations,
            answer_evidence,
        )
    candidate_citations = [serialize_evidence(item) for item in evidence]
    elapsed_ms = round((perf_counter() - started) * 1000)
    response = {
        "query": request.query,
        "mode": effective_mode,
        "requested_mode": request.mode,
        "query_plan": query_plan_dict,
        "answer": answer,
        "answer_mode": answer_mode,
        "citations": used_citations,
        "used_citations": used_citations,
        "used_citation_ids": used_citation_ids,
        "claim_citations": claim_citations,
        "candidate_citations": candidate_citations,
        "crag": {
            "corrected": crag_corrected,
            "before": crag_before.to_dict(),
            "after": crag_after.to_dict() if crag_after else None,
        },
        "research_trace": research_trace,
        "meta": {
            **retrieve_meta,
            "requested_mode": request.mode,
            "effective_mode": effective_mode,
            "filters": filters.to_dict(),
            "ranked_evidence": len(evidence),
            "selected_evidence": len(selected_evidence),
            "answer_evidence": len(answer_evidence),
            "context_packing": packed_context.meta,
            "used_citations": len(used_citations),
            "verified_claims": len(claim_citations),
            "citation_verifier": claim_citations[0].get("verifier") if claim_citations else "none",
            "crag_status": (crag_after or crag_before).status,
            "crag_corrected": crag_corrected,
            "research_steps": len(research_trace),
            "cache_hit": False,
            "elapsed_ms": elapsed_ms,
        },
    }
    _store_cached_response(cache_key, response)
    return response


def _response_cache_key(request: SearchRequest) -> str:
    return json.dumps(request.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


def _get_cached_response(key: str) -> dict | None:
    cached = response_cache.get(key)
    if not cached:
        return None
    stored_at, response = cached
    if time() - stored_at > RESPONSE_CACHE_TTL_SECONDS:
        response_cache.pop(key, None)
        return None
    response_cache.move_to_end(key)
    copied = copy.deepcopy(response)
    copied.setdefault("meta", {})["cache_hit"] = True
    copied["meta"]["elapsed_ms"] = 0
    return copied


def _store_cached_response(key: str, response: dict) -> None:
    cached = copy.deepcopy(response)
    cached.setdefault("meta", {})["cache_hit"] = False
    response_cache[key] = (time(), cached)
    response_cache.move_to_end(key)
    while len(response_cache) > RESPONSE_CACHE_MAX_ITEMS:
        response_cache.popitem(last=False)


def main() -> None:
    import uvicorn

    uvicorn.run("fast_rag.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
