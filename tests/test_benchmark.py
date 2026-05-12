from fast_rag.benchmark import REALISTIC_CASES, SUITES, BenchmarkCase, evaluate_response


def test_realistic_suite_has_50_realistic_queries_without_domain_allowlists() -> None:
    assert len(REALISTIC_CASES) == 50
    assert SUITES["realistic"] is REALISTIC_CASES
    assert all(case.include_domains == () for case in REALISTIC_CASES)
    assert any(case.mode == "deep" for case in REALISTIC_CASES)


def test_benchmark_payload_preserves_search_controls() -> None:
    case = BenchmarkCase(
        name="fresh",
        query="latest docs",
        mode="pro",
        lens="news",
        recency="week",
        country="gb",
        language="en",
    )
    payload = case.payload()
    assert payload["recency"] == "week"
    assert payload["country"] == "gb"
    assert payload["language"] == "en"
    assert payload["lens"] == "news"


def test_evaluate_response_omits_source_recall_when_no_gold_source() -> None:
    case = BenchmarkCase(name="observed", query="what is x", expected_terms=("answer",))
    result = evaluate_response(
        case,
        {
            "answer": "answer",
            "meta": {},
            "used_citations": [],
            "candidate_citations": [],
            "claim_citations": [],
        },
        wall_ms=10,
    )
    assert result["expected_source_recall"] is None
    assert result["used_source_recall"] is None
