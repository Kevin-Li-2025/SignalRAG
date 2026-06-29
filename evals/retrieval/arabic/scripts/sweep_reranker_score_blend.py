#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rerank_miracl_candidates import (
    DEFAULT_REVISION,
    load_relevance,
    mean_metrics,
    metrics_for_query,
    sort_key,
    write_trec_run,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune score-level blends over an existing MIRACL candidate-rerank "
            "ranked_by_query.json. The script uses a deterministic tune/holdout "
            "split so a dev-only blend can be rejected if it does not generalize."
        )
    )
    parser.add_argument("--ranked-by-query", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset", default="ar")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--metric-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--run-id", default="score-blend")
    return parser.parse_args()


def load_ranked_by_query(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} must contain a non-empty query mapping")
    return {
        str(query_id): [dict(row) for row in rows]
        for query_id, rows in payload.items()
    }


def normalize_minmax(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high == low:
        return [0.0 for _value in values]
    return [(value - low) / (high - low) for value in values]


def normalize_zscore(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance == 0.0:
        return [0.0 for _value in values]
    std = variance ** 0.5
    return [(value - mean) / std for value in values]


def reciprocal_ranks(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=True)
    return {
        str(row["docid"]): 1.0 / rank
        for rank, row in enumerate(ordered, start=1)
    }


def ranked_candidate_rows(
    rows: list[dict[str, Any]],
    mode: str,
    alpha: float,
    top_k: int,
) -> list[dict[str, Any]]:
    if mode == "reranker":
        scored = [
            {**row, "blend_score": float(row["score"])}
            for row in rows
        ]
        ordered = sorted(scored, key=lambda row: float(row["blend_score"]), reverse=True)
    elif mode == "first_stage":
        scored = [
            {**row, "blend_score": float(row["first_stage_score"])}
            for row in rows
        ]
        ordered = sorted(scored, key=lambda row: float(row["blend_score"]), reverse=True)
    elif mode in {"minmax", "zscore"}:
        reranker_scores = [float(row["score"]) for row in rows]
        first_stage_scores = [float(row["first_stage_score"]) for row in rows]
        if mode == "minmax":
            reranker_norm = normalize_minmax(reranker_scores)
            first_stage_norm = normalize_minmax(first_stage_scores)
        else:
            reranker_norm = normalize_zscore(reranker_scores)
            first_stage_norm = normalize_zscore(first_stage_scores)
        scored = [
            {
                **row,
                "blend_score": alpha * reranker_value + (1.0 - alpha) * first_value,
            }
            for row, reranker_value, first_value in zip(
                rows,
                reranker_norm,
                first_stage_norm,
                strict=True,
            )
        ]
        ordered = sorted(scored, key=lambda row: float(row["blend_score"]), reverse=True)
    elif mode == "rrf":
        reranker_rr = reciprocal_ranks(rows, "score")
        first_stage_rr = reciprocal_ranks(rows, "first_stage_score")
        scored = [
            {
                **row,
                "blend_score": alpha * reranker_rr[str(row["docid"])]
                + (1.0 - alpha) * first_stage_rr[str(row["docid"])],
            }
            for row in rows
        ]
        ordered = sorted(scored, key=lambda row: float(row["blend_score"]), reverse=True)
    else:
        raise ValueError(f"Unsupported blend mode: {mode}")
    return [
        {
            "docid": str(row["docid"]),
            "rank": rank,
            "score": float(row["blend_score"]),
            "reranker_score": float(row["score"]),
            "first_stage_score": float(row["first_stage_score"]),
        }
        for rank, row in enumerate(ordered[:top_k], start=1)
    ]


def candidate_doc_ids(
    rows: list[dict[str, Any]],
    mode: str,
    alpha: float,
    top_k: int,
) -> list[str]:
    return [
        str(row["docid"])
        for row in ranked_candidate_rows(rows, mode, alpha, top_k)
    ]


def evaluate(
    ranked_by_query: dict[str, list[dict[str, Any]]],
    query_ids: list[str],
    qrels: dict[str, set[str]],
    mode: str,
    alpha: float,
    top_k: int,
    metric_k: int,
) -> dict[str, float]:
    rows = [
        metrics_for_query(
            candidate_doc_ids(ranked_by_query[query_id], mode, alpha, top_k),
            qrels[query_id],
            metric_k,
        )
        for query_id in query_ids
    ]
    metrics = mean_metrics(rows)
    metrics["main_score"] = metrics["ndcg_at_10"]
    return metrics


def alpha_grid(step: float) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("--alpha-step must be in (0, 1]")
    values: list[float] = []
    current = 0.0
    while current < 1.0:
        values.append(round(current, 10))
        current += step
    values.append(1.0)
    return sorted(set(values))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked_by_query = load_ranked_by_query(Path(args.ranked_by_query))
    query_ids = sorted(ranked_by_query, key=sort_key)
    _query_text_by_id, qrels = load_relevance(args.subset, args.split, args.revision, query_ids)
    query_ids = [query_id for query_id in query_ids if qrels.get(query_id)]
    tune_ids = query_ids[::2]
    holdout_ids = query_ids[1::2]
    if not tune_ids or not holdout_ids:
        raise ValueError("Need at least two queries for tune/holdout split")

    baselines = {
        name: {
            "tune": evaluate(ranked_by_query, tune_ids, qrels, name, 1.0, args.top_k, args.metric_k),
            "holdout": evaluate(
                ranked_by_query,
                holdout_ids,
                qrels,
                name,
                1.0,
                args.top_k,
                args.metric_k,
            ),
            "full": evaluate(ranked_by_query, query_ids, qrels, name, 1.0, args.top_k, args.metric_k),
        }
        for name in ("reranker", "first_stage")
    }

    grid: list[dict[str, Any]] = []
    for mode in ("minmax", "zscore", "rrf"):
        for alpha in alpha_grid(args.alpha_step):
            tune_metrics = evaluate(
                ranked_by_query,
                tune_ids,
                qrels,
                mode,
                alpha,
                args.top_k,
                args.metric_k,
            )
            grid.append({"mode": mode, "alpha": alpha, "tune": tune_metrics})
    best = max(
        grid,
        key=lambda row: (
            float(row["tune"]["main_score"]),
            -abs(float(row["alpha"]) - 1.0),
        ),
    )
    best_mode = str(best["mode"])
    best_alpha = float(best["alpha"])
    best["holdout"] = evaluate(
        ranked_by_query,
        holdout_ids,
        qrels,
        best_mode,
        best_alpha,
        args.top_k,
        args.metric_k,
    )
    best["full"] = evaluate(
        ranked_by_query,
        query_ids,
        qrels,
        best_mode,
        best_alpha,
        args.top_k,
        args.metric_k,
    )
    best["delta_vs_reranker"] = {
        split: best[split]["main_score"] - baselines["reranker"][split]["main_score"]
        for split in ("tune", "holdout", "full")
    }
    best["delta_vs_first_stage"] = {
        split: best[split]["main_score"] - baselines["first_stage"][split]["main_score"]
        for split in ("tune", "holdout", "full")
    }
    best_ranked_by_query = {
        query_id: ranked_candidate_rows(
            ranked_by_query[query_id],
            best_mode,
            best_alpha,
            args.top_k,
        )
        for query_id in query_ids
    }
    run_file = output_dir / f"{args.subset}_{args.split}_{args.run_id}.txt"
    run_file_validation = write_trec_run(
        run_file,
        best_ranked_by_query,
        query_ids,
        args.run_id,
        args.top_k,
    )

    summary = {
        "experiment": "miracl-ar-reranker-score-blend-sweep",
        "ranked_by_query_source": str(args.ranked_by_query),
        "subset": args.subset,
        "split": args.split,
        "dataset_revision": args.revision,
        "query_count": len(query_ids),
        "tune_query_count": len(tune_ids),
        "holdout_query_count": len(holdout_ids),
        "top_k": args.top_k,
        "metric_k": args.metric_k,
        "alpha_step": args.alpha_step,
        "baselines": baselines,
        "best_by_tune": best,
        "run_file": {
            "path": str(run_file),
            **run_file_validation,
        },
        "grid": grid,
        "caution": (
            "This is a score-level dev sweep over an existing reranked dev run. "
            "Treat full-dev gains as diagnostic unless confirmed on a hidden or "
            "separate evaluation split."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["best_by_tune"], indent=2), flush=True)


if __name__ == "__main__":
    main()
