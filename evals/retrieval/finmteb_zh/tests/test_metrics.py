from finmteb_sota.metrics import (
    RankedQuery,
    average_precision,
    ndcg_at_k,
    reciprocal_rank,
    reranking_metrics,
)


def test_average_precision_perfect_order() -> None:
    assert average_precision([1, 0, 1], [3.0, 1.0, 2.0]) == 1.0


def test_average_precision_penalizes_late_positive() -> None:
    assert round(average_precision([1, 0, 1], [1.0, 3.0, 2.0]), 6) == 0.583333


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([0, 0, 1], [3.0, 2.0, 1.0]) == 1 / 3


def test_ndcg_at_k() -> None:
    assert ndcg_at_k([1, 0, 1], [3.0, 2.0, 1.0], k=3) < 1.0
    assert ndcg_at_k([1, 0, 1], [3.0, 1.0, 2.0], k=3) == 1.0


def test_reranking_metrics() -> None:
    metrics = reranking_metrics(
        [
            RankedQuery("a", labels=[1, 0], scores=[2.0, 1.0]),
            RankedQuery("b", labels=[0, 1], scores=[2.0, 1.0]),
        ]
    )
    assert metrics["map"] == 0.75
    assert metrics["mrr"] == 0.75
