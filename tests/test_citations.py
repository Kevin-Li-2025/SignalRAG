from fast_rag.citations import (
    _merge_judgements,
    normalize_answer_citations,
    select_answer_evidence,
    verify_claim_citations,
)
from fast_rag.models import Evidence


def test_normalize_answer_citations_returns_only_used_sources() -> None:
    evidence = [
        Evidence(id=1, title="One", url="https://one.example", passage="A", score=1),
        Evidence(id=2, title="Two", url="https://two.example", passage="B", score=1),
        Evidence(id=3, title="Three", url="https://three.example", passage="C", score=1),
    ]
    answer, citations, ids = normalize_answer_citations("Alpha [2], beta [9], gamma [1, 3].", evidence)
    assert answer == "Alpha [2], beta , gamma [1][3]."
    assert ids == [2, 1, 3]
    assert [item["url"] for item in citations] == [
        "https://two.example",
        "https://one.example",
        "https://three.example",
    ]


def test_normalize_answer_citations_dedupes_used_ids() -> None:
    evidence = [
        Evidence(id=1, title="One", url="https://one.example", passage="A", score=1),
    ]
    _, citations, ids = normalize_answer_citations("Alpha [1]. Beta [1].", evidence)
    assert ids == [1]
    assert len(citations) == 1


def test_normalize_answer_citations_ignores_sources_footer() -> None:
    evidence = [
        Evidence(id=1, title="One", url="https://one.example", passage="A", score=1),
        Evidence(id=2, title="Two", url="https://two.example", passage="B", score=1),
    ]
    answer, citations, ids = normalize_answer_citations(
        "Alpha [1].\n\nSources checked: [1], [2]",
        evidence,
    )
    assert answer == "Alpha [1]."
    assert ids == [1]
    assert [item["id"] for item in citations] == [1]


def test_select_answer_evidence_keeps_best_passage_per_url() -> None:
    evidence = [
        Evidence(id=1, title="One", url="https://one.example", passage="A", score=2),
        Evidence(id=2, title="One again", url="https://one.example/", passage="B", score=1),
        Evidence(id=3, title="Two", url="https://two.example", passage="C", score=1),
    ]
    selected = select_answer_evidence(evidence)
    assert [item.id for item in selected] == [1, 3]


def test_verify_claim_citations_marks_supported_claims() -> None:
    evidence = [
        Evidence(
            id=1,
            title="ChatGPT Search",
            url="https://help.openai.com/search",
            passage="ChatGPT search rewrites your query into targeted queries and may include inline citations.",
            score=3,
        )
    ]
    claims = verify_claim_citations(
        "ChatGPT search rewrites your query into targeted queries. [1]",
        evidence,
    )
    assert claims[0]["status"] == "supported"
    assert claims[0]["citation_ids"] == [1]
    assert claims[0]["citations"][0]["url"] == "https://help.openai.com/search"


def test_verify_claim_citations_inherits_line_citation_for_split_claims() -> None:
    evidence = [
        Evidence(
            id=1,
            title="DeepSeek Thinking Mode",
            url="https://api-docs.deepseek.com/guides/thinking_mode",
            passage="DeepSeek thinking mode supports high and max reasoning effort. Thinking can be disabled for simple tasks.",
            score=3,
        )
    ]
    claims = verify_claim_citations(
        "DeepSeek thinking mode supports high and max reasoning effort. Thinking can be disabled for simple tasks. [1]",
        evidence,
    )
    assert len(claims) == 2
    assert all(claim["citation_ids"] == [1] for claim in claims)
    assert all(claim["status"] == "supported" for claim in claims)


def test_verify_claim_citations_flags_missing_citation() -> None:
    claims = verify_claim_citations(
        "ChatGPT search rewrites a user query into targeted search queries before answering.",
        [],
    )
    assert claims[0]["status"] == "missing_citation"


def test_merge_judgements_overrides_lexical_status() -> None:
    claims = [
        {
            "claim": "A",
            "citation_ids": [1],
            "citations": [],
            "status": "weak",
            "support_score": 0.2,
            "verifier": "lexical",
        }
    ]
    merged = _merge_judgements(
        claims,
        {"claims": [{"index": 0, "status": "supported", "confidence": 0.91, "rationale": "direct"}]},
    )
    assert merged[0]["status"] == "supported"
    assert merged[0]["support_score"] == 0.91
    assert merged[0]["verifier"] == "deepseek"


def test_merge_judgements_can_target_adaptive_judge_subset() -> None:
    claims = [
        {"claim": "A", "citation_ids": [1], "status": "supported", "support_score": 0.9},
        {"claim": "B", "citation_ids": [1], "status": "weak", "support_score": 0.1},
    ]
    merged = _merge_judgements(
        claims,
        {"claims": [{"index": 0, "status": "supported", "confidence": 0.82, "rationale": "direct"}]},
        original_indexes=[1],
    )
    assert merged[0]["support_score"] == 0.9
    assert merged[1]["status"] == "supported"
    assert merged[1]["support_score"] == 0.82
