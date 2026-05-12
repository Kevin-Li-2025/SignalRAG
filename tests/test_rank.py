from fast_rag.models import Document
from fast_rag.rank import rank_evidence, source_quality, source_trust_tier, tokenize


def test_tokenize_supports_english_and_chinese() -> None:
    tokens = tokenize("ChatGPT 搜索 RAG 很快")
    assert "chatgpt" in tokens
    assert "搜" in tokens
    assert "搜索" in tokens


def test_rank_evidence_prefers_exact_relevant_passage() -> None:
    docs = [
        Document(
            url="https://example.com/slow",
            title="Cooking notes",
            text="This page talks about pasta and tomatoes. It is not related to web search systems.",
        ),
        Document(
            url="https://developers.openai.com/api/docs/guides/tools-web-search",
            title="OpenAI Web search",
            text="Web search allows models to access up-to-date information from the internet and provide answers with sourced citations.",
        ),
    ]
    ranked = rank_evidence("web search sourced citations", docs, limit=2)
    assert ranked[0].url.startswith("https://developers.openai.com")


def test_rank_evidence_uses_phrase_match_over_loose_terms() -> None:
    docs = [
        Document(
            url="https://help.openai.com/deep-research",
            title="Deep research in ChatGPT",
            text="ChatGPT can search the web. This article explains how research works in OpenAI products.",
        ),
        Document(
            url="https://help.openai.com/chatgpt-search",
            title="ChatGPT Search",
            text="ChatGPT search can answer questions with timely information from the web and relevant sources.",
            provider="official",
        ),
    ]
    ranked = rank_evidence("ChatGPT search how it works OpenAI", docs, limit=2)
    assert ranked[0].url.endswith("chatgpt-search")


def test_rank_evidence_uses_source_context_for_pronoun_passages() -> None:
    docs = [
        Document(
            url="https://example.com/generic",
            title="Generic API docs",
            text="It supports a mode switch and an effort parameter for requests.",
        ),
        Document(
            url="https://api-docs.deepseek.com/guides/thinking_mode",
            title="DeepSeek Thinking Mode",
            text="It supports a mode switch and an effort parameter for requests.",
            provider="official",
        ),
    ]
    ranked = rank_evidence("DeepSeek thinking mode reasoning effort", docs, limit=2)
    assert ranked[0].url.startswith("https://api-docs.deepseek.com")
    assert ranked[0].signals["contextual_bm25"] == 1.0


def test_rank_evidence_penalizes_passages_missing_required_topic() -> None:
    docs = [
        Document(
            url="https://help.openai.com/chatgpt-search",
            title="ChatGPT Search",
            text="Contact OpenAI Support if you need help with restaurant results in ChatGPT.",
            provider="official",
        ),
        Document(
            url="https://help.openai.com/chatgpt-search",
            title="ChatGPT Search",
            text="ChatGPT search rewrites queries and returns answers with source links.",
            provider="official",
        ),
    ]
    ranked = rank_evidence("ChatGPT search how it works OpenAI", docs, limit=2)
    assert "rewrites queries" in ranked[0].passage


def test_source_quality_boosts_official_domains() -> None:
    assert source_quality("https://developers.openai.com/api/docs") > source_quality("https://example.com")
    assert source_quality("https://api-docs.deepseek.com/api/create-chat-completion") > source_quality("https://example.com")


def test_source_trust_tiers_cover_authoritative_sources() -> None:
    assert source_trust_tier("https://www.cdc.gov/flu") == "government"
    assert source_trust_tier("https://arxiv.org/abs/2307.03172") == "academic"
    assert source_trust_tier("https://www.w3.org/TR/WCAG22/") == "standards"
    assert source_trust_tier("https://docs.python.org/3/library/json.html") == "official_docs"
    assert source_trust_tier("https://react.dev/reference/react/useEffect") == "official_docs"
    assert source_trust_tier("https://support.google.com/chrome/answer/95426") == "official_docs"
    assert source_trust_tier("https://www.reuters.com/world/") == "news_wire"
    assert source_trust_tier("https://www.reddit.com/r/search/") == "low_signal"


def test_source_quality_orders_trust_tiers() -> None:
    assert source_quality("https://www.nih.gov/health-information") > source_quality("https://example.com")
    assert source_quality("https://www.reddit.com/r/AskDocs") < source_quality("https://example.com")


def test_rank_evidence_limits_single_url_dominance() -> None:
    docs = [
        Document(
            url="https://example.com/one",
            title="Web search guide",
            text="Web search citations and ChatGPT search details. " * 80,
        ),
        Document(
            url="https://another.example/two",
            title="Web search citations",
            text="Web search citations from a second source. " * 20,
        ),
    ]
    ranked = rank_evidence("web search citations", docs, limit=4)
    assert any(item.url == "https://another.example/two" for item in ranked)
