from scripts.rerank_teacher_jsonl import build_items, summarize_rows


def test_build_items_tracks_pos_and_neg_indices():
    rows = [
        {
            "query": "q1",
            "pos": ["p1", "p2"],
            "neg": ["n1"],
        }
    ]

    assert build_items(rows) == [
        {"row_index": 0, "kind": "pos", "item_index": 0, "query": "q1", "text": "p1"},
        {"row_index": 0, "kind": "pos", "item_index": 1, "query": "q1", "text": "p2"},
        {"row_index": 0, "kind": "neg", "item_index": 0, "query": "q1", "text": "n1"},
    ]


def test_summarize_rows_reports_teacher_margin_quality():
    rows = [
        {
            "pos_scores": [4.0],
            "neg_scores": [1.0, 2.0],
            "bge_m3_hybrid_pos_scores": [0.8],
            "bge_m3_hybrid_neg_scores": [0.5, 0.4],
            "neg_sources": ["judged_negative", "unjudged_candidate"],
        },
        {
            "pos_scores": [0.5],
            "neg_scores": [0.7],
            "bge_m3_hybrid_pos_scores": [0.6],
            "bge_m3_hybrid_neg_scores": [0.7],
            "neg_sources": ["judged_negative"],
        },
    ]

    summary = summarize_rows(rows)

    assert summary["positive_pairs"] == 2
    assert summary["negative_pairs"] == 3
    assert summary["positive_above_max_negative_rows"] == 1
    assert summary["positive_above_max_negative_rate"] == 0.5
    assert summary["selected_negative_sources"] == {
        "judged_negative": 2,
        "unjudged_candidate": 1,
    }
    assert summary["hybrid_margin_vs_reranker_margin_pearson"] is not None
