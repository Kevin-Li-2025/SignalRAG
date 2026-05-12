from fast_rag.answer import _extractive_answer
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
