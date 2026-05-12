from fast_rag.crag import assess_retrieval, should_correct
from fast_rag.models import Document, Evidence


def test_assess_retrieval_marks_sparse_evidence_for_correction() -> None:
    assessment = assess_retrieval(
        "OpenAI web search API citations",
        [],
        [],
    )
    assert assessment.status == "insufficient"
    assert should_correct(assessment)
    assert assessment.corrective_queries


def test_assess_retrieval_accepts_relevant_official_evidence() -> None:
    docs = [
        Document(
            url="https://developers.openai.com/api/docs/guides/tools-web-search",
            title="Web search",
            text="Web search API citations and sources for OpenAI responses.",
            provider="official",
        )
    ]
    evidence = [
        Evidence(
            id=1,
            title="Web search",
            url=docs[0].url,
            passage="OpenAI web search API can return answers with citations and sources.",
            score=8,
            provider="official",
        ),
        Evidence(
            id=2,
            title="Help",
            url="https://help.openai.com/search",
            passage="ChatGPT search provides answers with links to relevant web sources.",
            score=5,
            provider="official",
        ),
        Evidence(
            id=3,
            title="Docs",
            url="https://platform.openai.com/docs",
            passage="Developers can use tools to ground model responses in web data.",
            score=4,
            provider="official",
        ),
    ]
    assessment = assess_retrieval("OpenAI web search API citations sources", docs, evidence)
    assert assessment.confidence >= 0.58
    assert not should_correct(assessment)


def test_assess_retrieval_counts_trusted_non_official_sources() -> None:
    docs = [
        Document(
            url="https://www.nih.gov/health-information",
            title="Health information",
            text="Government health information about vaccines and evidence quality.",
        )
    ]
    evidence = [
        Evidence(
            id=1,
            title="Health information",
            url=docs[0].url,
            passage="Government health information about vaccines and evidence quality.",
            score=8,
        ),
        Evidence(
            id=2,
            title="CDC",
            url="https://www.cdc.gov/vaccines",
            passage="CDC vaccine information provides evidence-based public health guidance.",
            score=6,
        ),
        Evidence(
            id=3,
            title="WHO",
            url="https://www.who.int/health-topics/vaccines",
            passage="WHO vaccine information covers public health guidance and evidence.",
            score=5,
        ),
    ]
    assessment = assess_retrieval("vaccine evidence public health guidance", docs, evidence)
    assert assessment.metrics["trusted_sources"] == 3
    assert "no_curated_trusted_source" not in assessment.reasons
