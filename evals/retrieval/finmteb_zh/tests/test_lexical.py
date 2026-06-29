from finmteb_sota.lexical import (
    blend_scores,
    blend_scores_by_group,
    cjk_ngram_recall,
    lexical_feature_values,
    lexical_score,
    title_text,
    zscore,
)


def test_lexical_prefers_overlap() -> None:
    query = "2023 revenue net income"
    good = "The company reported 2023 revenue and net income growth."
    bad = "Unrelated monetary policy commentary."
    assert lexical_score(query, good) > lexical_score(query, bad)


def test_zscore_constant() -> None:
    assert zscore([1.0, 1.0]) == [0.0, 0.0]


def test_blend_scores() -> None:
    scores = blend_scores([1.0, 2.0], [2.0, 1.0], alpha=0.1)
    assert scores[1] > scores[0]


def test_blend_scores_by_group() -> None:
    scores = blend_scores_by_group(
        ["a", "a", "b", "b"],
        [1.0, 2.0, 100.0, 101.0],
        [2.0, 1.0, 100.0, 99.0],
        alpha=0.1,
    )
    assert scores[1] > scores[0]
    assert scores[3] > scores[2]


def test_title_and_ngram_features_focus_on_document_head() -> None:
    doc = "南京、沈阳、大连全面取消限购，一线城市限购政策会松动吗？  正文后续内容"
    query = "南京、沈阳、大连取消限购政策是否会刺激购房需求？"

    assert title_text(doc).startswith("南京、沈阳、大连")
    assert cjk_ngram_recall(query, title_text(doc), 2) > 0.2
    features = lexical_feature_values(query, doc)
    assert features["title_bigram"] > 0.0
    assert set(features) >= {"lexical", "head_lexical", "title_lexical", "title_trigram"}
