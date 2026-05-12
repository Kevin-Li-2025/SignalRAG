from fast_rag.research import build_research_steps


def test_build_research_steps_includes_freshness_and_countercheck() -> None:
    steps = build_research_steps(
        "compare current RAG APIs",
        {"intent": "comparison", "needs_freshness": True},
    )
    labels = [step.label for step in steps]
    assert "official" in labels
    assert "freshness" in labels
    assert "countercheck" in labels
    assert len(steps) <= 5


def test_build_research_steps_has_context_for_simple_queries() -> None:
    steps = build_research_steps("ChatGPT search", {"intent": "lookup", "needs_freshness": False})
    assert len(steps) >= 3
    assert steps[-1].label == "context"


def test_build_research_steps_deep_adds_countercheck_and_synthesis() -> None:
    steps = build_research_steps(
        "build a frontier AI search engine",
        {"intent": "feature explanation", "needs_freshness": False, "reasoning_effort": "max"},
    )
    labels = [step.label for step in steps]
    assert "countercheck" in labels
    assert labels[-1] == "synthesis"
