from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from time import perf_counter, time
from typing import Literal

import httpx

from .cache import normalize_query
from .config import settings


ReasoningEffort = Literal["none", "high", "max"]
SearchDepth = Literal["fast", "pro", "deep"]


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    answer_style: str
    needs_freshness: bool
    search_depth: SearchDepth
    reasoning_effort: ReasoningEffort
    confidence: float
    rationale: str
    planner: str = "heuristic"
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


FRESHNESS_RE = re.compile(
    r"\b(latest|today|current|now|recent|new|202[5-9])\b|最新|今天|当前|最近|今年",
    re.IGNORECASE,
)
COMPLEX_RE = re.compile(
    r"\b(compare|versus|vs\.?|trade[- ]?off|why|how|architecture|debug|root cause|evaluate|recall|precision|benchmark|policy|legal|medical|financial)\b"
    r"|比较|权衡|为什么|怎么|架构|调试|诊断|根因|评估|召回|准确|政策|法律|医疗|金融",
    re.IGNORECASE,
)
MAX_RE = re.compile(
    r"\b(deep research|comprehensive|multi[- ]?step|long[- ]?horizon|formal proof|threat model)\b"
    r"|深入研究|全面|多步|复杂|系统设计|威胁模型|端到端",
    re.IGNORECASE,
)
API_RE = re.compile(r"\b(api|sdk|endpoint|json|curl|python|javascript|typescript)\b|接口|代码|报错|实现", re.IGNORECASE)
RECOMMEND_RE = re.compile(r"\b(best|recommend|choose|which|should I)\b|推荐|选择|哪个好|应该", re.IGNORECASE)
COMPARE_RE = re.compile(r"\b(compare|versus|vs\.?|difference|trade[- ]?off)\b|比较|区别|权衡", re.IGNORECASE)
PLAN_CACHE_TTL_SECONDS = 600
PLAN_CACHE_MAX_ITEMS = 256
_PLAN_CACHE: OrderedDict[tuple[str, str, bool], tuple[float, QueryPlan]] = OrderedDict()


def heuristic_query_plan(query: str, requested_mode: str = "fast") -> QueryPlan:
    query = query.strip()
    needs_freshness = bool(FRESHNESS_RE.search(query))
    is_complex = bool(COMPLEX_RE.search(query))
    is_max = bool(MAX_RE.search(query))

    if COMPARE_RE.search(query):
        intent = "comparison"
        answer_style = "tradeoff_summary"
    elif RECOMMEND_RE.search(query):
        intent = "recommendation"
        answer_style = "ranked_recommendation"
    elif API_RE.search(query):
        intent = "api_or_code"
        answer_style = "implementation_guidance"
    elif is_complex:
        intent = "analysis"
        answer_style = "structured_explanation"
    else:
        intent = "lookup"
        answer_style = "concise_answer"

    if is_max:
        reasoning_effort: ReasoningEffort = "max"
    elif requested_mode == "deep":
        reasoning_effort = "high"
    elif is_complex or intent in {"comparison", "recommendation", "api_or_code"}:
        reasoning_effort = "high"
    else:
        reasoning_effort = "none"

    if requested_mode == "deep" or is_max:
        search_depth: SearchDepth = "deep"
    elif requested_mode == "pro" or is_complex or needs_freshness:
        search_depth = "pro"
    else:
        search_depth = "fast"
    return QueryPlan(
        intent=intent,
        answer_style=answer_style,
        needs_freshness=needs_freshness,
        search_depth=search_depth,
        reasoning_effort=reasoning_effort,
        confidence=0.62,
        rationale="heuristic rules based on freshness, complexity, and task type",
    )


async def plan_query(query: str, requested_mode: str = "fast") -> QueryPlan:
    started = perf_counter()
    fallback = heuristic_query_plan(query, requested_mode)
    cache_key = _plan_cache_key(query, requested_mode)
    cached = _get_cached_plan(cache_key)
    if cached:
        return cached
    if not settings.deepseek_api_key:
        _store_plan(cache_key, fallback)
        return fallback

    payload = {
        "model": os.getenv("DEEPSEEK_PLANNER_MODEL", settings.deepseek_model),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a tiny query planning model for a web-search RAG system. "
                    "Classify what the user wants and choose the minimum search depth and DeepSeek reasoning effort needed. "
                    "Return only valid JSON. Use reasoning_effort 'none' for simple factual lookup, "
                    "'high' for multi-hop synthesis, comparisons, recommendations, API/code guidance, or uncertainty, "
                    "and 'max' only for complex long-horizon analysis, formal proof, deep research, or many constraints. "
                    "Use search_depth 'fast' for simple lookup, 'pro' for fresh, comparative, or multi-hop questions, "
                    "and 'deep' only for deep research or max-effort tasks. "
                    "DeepSeek supports only high/max when thinking is enabled; 'none' means thinking disabled."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Requested UI mode: {requested_mode}\n"
                    f"Query: {query}\n\n"
                    "Return JSON with keys: intent, answer_style, needs_freshness, search_depth, "
                    "reasoning_effort, confidence, rationale."
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 320,
    }
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            response = await client.post(
                f"{settings.deepseek_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        plan = _coerce_plan(json.loads(content), fallback)
        planned = QueryPlan(
            **{**plan.to_dict(), "planner": "deepseek", "elapsed_ms": round((perf_counter() - started) * 1000)}
        )
        _store_plan(cache_key, planned)
        return planned
    except Exception:
        planned = QueryPlan(
            **{**fallback.to_dict(), "planner": "heuristic_fallback", "elapsed_ms": round((perf_counter() - started) * 1000)}
        )
        _store_plan(cache_key, planned)
        return planned


def _plan_cache_key(query: str, requested_mode: str) -> tuple[str, str, bool]:
    return (normalize_query(query), requested_mode, bool(settings.deepseek_api_key))


def _get_cached_plan(key: tuple[str, str, bool]) -> QueryPlan | None:
    cached = _PLAN_CACHE.get(key)
    if not cached:
        return None
    stored_at, plan = cached
    if time() - stored_at > PLAN_CACHE_TTL_SECONDS:
        _PLAN_CACHE.pop(key, None)
        return None
    _PLAN_CACHE.move_to_end(key)
    planner = plan.planner if plan.planner.endswith("_cache") else f"{plan.planner}_cache"
    return QueryPlan(**{**plan.to_dict(), "planner": planner, "elapsed_ms": 0})


def _store_plan(key: tuple[str, str, bool], plan: QueryPlan) -> None:
    _PLAN_CACHE[key] = (time(), plan)
    _PLAN_CACHE.move_to_end(key)
    while len(_PLAN_CACHE) > PLAN_CACHE_MAX_ITEMS:
        _PLAN_CACHE.popitem(last=False)


def _coerce_plan(raw: dict, fallback: QueryPlan) -> QueryPlan:
    intent = str(raw.get("intent") or fallback.intent)[:48]
    answer_style = str(raw.get("answer_style") or fallback.answer_style)[:48]
    needs_freshness = bool(raw.get("needs_freshness", fallback.needs_freshness))
    search_depth = str(raw.get("search_depth") or fallback.search_depth).lower()
    if search_depth not in {"fast", "pro", "deep"}:
        search_depth = fallback.search_depth
    effort = str(raw.get("reasoning_effort") or fallback.reasoning_effort).lower()
    if effort in {"disabled", "off", "false", "no", "low"}:
        effort = "none"
    if effort in {"medium"}:
        effort = "high"
    if effort in {"xhigh", "maximum"}:
        effort = "max"
    if effort not in {"none", "high", "max"}:
        effort = fallback.reasoning_effort
    confidence = raw.get("confidence", fallback.confidence)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = fallback.confidence
    rationale = str(raw.get("rationale") or fallback.rationale)[:240]
    return QueryPlan(
        intent=intent,
        answer_style=answer_style,
        needs_freshness=needs_freshness,
        search_depth=search_depth,  # type: ignore[arg-type]
        reasoning_effort=effort,  # type: ignore[arg-type]
        confidence=confidence,
        rationale=rationale,
    )
