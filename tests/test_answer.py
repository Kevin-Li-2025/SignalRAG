from fast_rag.answer import _extractive_answer, repair_missing_citations
from fast_rag.models import Evidence


def test_extractive_answer_prefers_mechanism_sentences() -> None:
    evidence = [
        Evidence(
            id=1,
            title="ChatGPT Search",
            url="https://help.openai.com/example",
            passage=(
                "Learn how ChatGPT search works, including location-based results. "
                "To provide relevant responses, ChatGPT search typically rewrites your query into one or more targeted queries that it sends those providers. "
                "ChatGPT responses that use search may include inline citations."
            ),
            score=5.0,
        )
    ]
    answer = _extractive_answer("ChatGPT search how it works", evidence)
    assert "rewrites your query" in answer
    assert "inline citations" in answer


def test_extractive_answer_can_build_chatgpt_search_pattern() -> None:
    evidence = [
        Evidence(
            id=1,
            title="Introducing ChatGPT search",
            url="https://openai.com/index/introducing-chatgpt-search",
            passage=(
                "The search model is a fine-tuned version of GPT-4o. "
                "ChatGPT search leverages third-party search providers."
            ),
            score=5.0,
            provider="official",
        ),
        Evidence(
            id=2,
            title="ChatGPT Search",
            url="https://help.openai.com/search",
            passage=(
                "ChatGPT search typically rewrites your query into one or more targeted queries. "
                "ChatGPT will choose to search the web based on what you ask, or you can manually choose search. "
                "You can get fast, timely answers with links to relevant web sources."
            ),
            score=4.0,
            provider="official",
        ),
    ]
    answer = _extractive_answer("ChatGPT search how it works", evidence)
    assert "模型层" in answer
    assert "查询改写" in answer
    assert "输出形式" in answer


def test_repair_missing_citations_adds_best_evidence_id() -> None:
    evidence = [
        Evidence(
            id=1,
            title="DeepSeek Thinking Mode",
            url="https://api-docs.deepseek.com/guides/thinking_mode",
            passage="DeepSeek thinking mode supports high and max reasoning effort.",
            score=5.0,
            provider="official",
        ),
        Evidence(
            id=2,
            title="Other",
            url="https://example.com",
            passage="Unrelated setup information.",
            score=1.0,
        ),
    ]
    answer = repair_missing_citations(
        "DeepSeek thinking mode",
        "DeepSeek thinking mode supports high and max reasoning effort.",
        evidence,
    )
    assert answer.endswith("[1]")


def test_repair_missing_citations_preserves_existing_citations() -> None:
    evidence = [Evidence(id=1, title="One", url="https://one.example", passage="Alpha", score=1)]
    assert repair_missing_citations("Alpha", "Alpha [1]", evidence) == "Alpha [1]"
