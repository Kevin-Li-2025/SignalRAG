from fast_rag.context import pack_answer_context
from fast_rag.models import Evidence


def test_pack_answer_context_adds_source_context_and_compresses() -> None:
    evidence = [
        Evidence(
            id=1,
            title="DeepSeek Thinking Mode",
            url="https://api-docs.deepseek.com/guides/thinking_mode",
            passage=(
                "This unrelated introduction talks about account setup. "
                "DeepSeek thinking mode is enabled with a thinking parameter and reasoning_effort high or max. "
                "Another unrelated sentence explains billing preferences."
            ),
            score=5,
            provider="official",
        )
    ]
    packed = pack_answer_context("DeepSeek thinking reasoning_effort max", evidence, "fast")
    assert packed.meta["strategy"] == "query_aware_contextual_sandwich"
    assert packed.meta["packed_chars"] <= packed.meta["budget_chars"]
    assert packed.evidence[0].id == 1
    assert "Source context: title=DeepSeek Thinking Mode" in packed.evidence[0].passage
    assert "reasoning_effort high or max" in packed.evidence[0].passage


def test_pack_answer_context_reorders_to_keep_strong_items_at_edges() -> None:
    evidence = [
        Evidence(id=1, title="One", url="https://one.example", passage="Relevant first.", score=5),
        Evidence(id=2, title="Two", url="https://two.example", passage="Relevant second.", score=4),
        Evidence(id=3, title="Three", url="https://three.example", passage="Relevant third.", score=3),
        Evidence(id=4, title="Four", url="https://four.example", passage="Relevant fourth.", score=2),
    ]
    packed = pack_answer_context("relevant", evidence, "pro")
    assert [item.id for item in packed.evidence] == [1, 3, 4, 2]
