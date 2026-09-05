#!/usr/bin/env python3
"""Verify that cached scores and frozen metrics are invariant to candidate order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from finmteb_sota.data import flatten_records, load_reranking_records
from finmteb_sota.metrics import reranking_metrics
from finmteb_sota.qwen3 import DEFAULT_INSTRUCTION
from finmteb_sota.score_cache import build_candidate_ids, load_score_cache
from finmteb_sota.tasks import resolve_tasks
from scripts.eval_blend_strategy import (
    apply_strategy,
    feature_matrix,
    group_scores,
    load_strategies,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["zh"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--strategy-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/score_cache"))
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260905, 2234, 314159, 8675309])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def metrics_close(left: dict[str, float], right: dict[str, float], tolerance: float) -> bool:
    return set(left) == set(right) and all(
        math.isclose(left[key], right[key], rel_tol=0.0, abs_tol=tolerance) for key in left
    )


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Candidate-order seeds must be unique")
    strategies = load_strategies(args.strategy_file)
    results = []
    all_invariant = True

    for task in resolve_tasks(args.tasks):
        records = load_reranking_records(task.dataset_id, split=args.split)
        seed_results = []
        reference_metrics: dict[str, float] | None = None
        reference_ids: set[str] | None = None
        for seed in args.seeds:
            queries, docs, labels, qids = flatten_records(records, seed=seed)
            candidate_ids = build_candidate_ids(qids, queries, docs)
            model_scores, cache_path = load_score_cache(
                args.cache_dir,
                task,
                args.split,
                args.instruction,
                args.cache_tag,
                candidate_ids,
            )
            strategy = strategies.get(task.dataset_id, strategies.get(task.leaderboard_name))
            scores, applied = apply_strategy(
                qids, model_scores, feature_matrix(queries, docs), strategy
            )
            metrics = reranking_metrics(group_scores(qids, labels, scores))
            id_set = set(candidate_ids)
            if reference_metrics is None:
                reference_metrics = metrics
                reference_ids = id_set
            invariant = metrics_close(reference_metrics, metrics, args.tolerance)
            same_coverage = id_set == reference_ids
            all_invariant = all_invariant and invariant and same_coverage
            seed_results.append(
                {
                    "seed": seed,
                    "metrics": metrics,
                    "metric_invariant": invariant,
                    "same_candidate_coverage": same_coverage,
                    "candidate_count": len(candidate_ids),
                    "candidate_set_sha256": hashlib.sha256(
                        "\n".join(sorted(candidate_ids)).encode()
                    ).hexdigest(),
                    "cache_path": str(cache_path),
                }
            )
        results.append(
            {
                "dataset": task.dataset_id,
                "leaderboard_name": task.leaderboard_name,
                "split": args.split,
                "num_queries": len(records),
                "strategy": applied,
                "order_invariant": all(
                    row["metric_invariant"] and row["same_candidate_coverage"]
                    for row in seed_results
                ),
                "seeds": seed_results,
            }
        )

    payload = {
        "status": "complete" if all_invariant else "failed",
        "cache_format": "v3 candidate-ID keyed",
        "cache_tag": args.cache_tag,
        "candidate_order_seeds": args.seeds,
        "tolerance": args.tolerance,
        "all_tasks_order_invariant": all_invariant,
        "tasks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all_invariant:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
