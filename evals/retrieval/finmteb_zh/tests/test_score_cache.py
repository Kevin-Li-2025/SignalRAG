from types import SimpleNamespace

from finmteb_sota.score_cache import model_cache_tag, score_cache_key


def test_qwen_model_cache_tags_are_short_and_distinct() -> None:
    assert model_cache_tag("Qwen/Qwen3-Reranker-8B") == "qwen3_8b"
    assert model_cache_tag("Qwen/Qwen3-Reranker-4B") == "qwen3_4b"
    assert model_cache_tag("BAAI/bge-reranker-v2-m3") == "bge_reranker_v2_m3"


def test_score_cache_key_includes_cache_tag() -> None:
    task = SimpleNamespace(leaderboard_name="DISCFinLLMReranking")

    key_8b = score_cache_key(task, "train", "instruction", "qwen3_8b")
    key_4b = score_cache_key(task, "train", "instruction", "qwen3_4b")

    assert key_8b != key_4b
    assert key_8b.startswith("DISCFinLLMReranking_train_qwen3_8b_")
    assert key_4b.startswith("DISCFinLLMReranking_train_qwen3_4b_")
