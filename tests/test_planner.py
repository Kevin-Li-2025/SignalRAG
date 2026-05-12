from fast_rag.planner import _coerce_plan, heuristic_query_plan


def test_heuristic_query_plan_simple_lookup_uses_no_reasoning() -> None:
    plan = heuristic_query_plan("What is ChatGPT search?", "fast")
    assert plan.intent == "lookup"
    assert plan.reasoning_effort == "none"
    assert plan.search_depth == "fast"


def test_heuristic_query_plan_api_eval_uses_high_reasoning() -> None:
    plan = heuristic_query_plan("分析一下 API recall 和 latency 怎么评估", "fast")
    assert plan.intent in {"api_or_code", "analysis"}
    assert plan.reasoning_effort == "high"
    assert plan.search_depth == "pro"


def test_heuristic_query_plan_deep_research_uses_max_reasoning() -> None:
    plan = heuristic_query_plan("做一个全面复杂的多步系统设计分析", "fast")
    assert plan.reasoning_effort == "max"
    assert plan.search_depth == "deep"


def test_coerce_plan_maps_medium_to_high_for_deepseek() -> None:
    fallback = heuristic_query_plan("compare two APIs", "fast")
    plan = _coerce_plan(
        {
            "intent": "comparison",
            "answer_style": "tradeoff_summary",
            "needs_freshness": True,
            "search_depth": "deep",
            "reasoning_effort": "medium",
            "confidence": 0.9,
            "rationale": "comparison",
        },
        fallback,
    )
    assert plan.reasoning_effort == "high"
    assert plan.search_depth == "deep"


def test_coerce_plan_accepts_pro_search_depth() -> None:
    fallback = heuristic_query_plan("latest API docs", "fast")
    plan = _coerce_plan(
        {
            "intent": "lookup",
            "answer_style": "concise",
            "needs_freshness": True,
            "search_depth": "pro",
            "reasoning_effort": "none",
            "confidence": 0.8,
            "rationale": "fresh lookup",
        },
        fallback,
    )
    assert plan.search_depth == "pro"
