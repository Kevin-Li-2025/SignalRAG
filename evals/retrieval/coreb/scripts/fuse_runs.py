from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from coreb_sota.data import load_task, task_names
from coreb_sota.metrics import evaluate_run, weighted_average


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse CoREB runs with weighted reciprocal-rank fusion.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/coreb"))
    parser.add_argument("--split", default="release_v2603")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def reciprocal_rank(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


def fuse_query_runs(query_runs: list[dict[str, float]], weights: list[float], top_k: int, rrf_k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for run, weight in zip(query_runs, weights):
        ranked = sorted(run.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * reciprocal_rank(rank, rrf_k)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])


def main() -> None:
    args = parse_args()
    if len(args.runs) != len(args.weights):
        raise ValueError("--runs and --weights must have the same length")
    reports = [json.loads(path.read_text()) for path in args.runs]
    task_metrics: dict[str, dict[str, Any]] = {}
    output_runs: dict[str, dict[str, dict[str, float]]] = {}

    for task_name in task_names(args.tasks):
        task = load_task(args.data_dir, args.split, task_name)
        fused_task: dict[str, dict[str, float]] = {}
        for query_id in task.queries:
            per_run = [report["runs"][task_name].get(query_id, {}) for report in reports]
            fused_task[query_id] = fuse_query_runs(per_run, args.weights, args.top_k, args.rrf_k)
        task_metrics[task_name] = evaluate_run(
            task.qrels,
            fused_task,
            k=args.eval_k,
            exclude_doc_ids=task.exclude_doc_ids,
        )
        output_runs[task_name] = fused_task

    metric = f"ndcg@{args.eval_k}"
    report = {
        "method": "weighted_rrf_fusion",
        "split": args.split,
        "input_runs": [str(path) for path in args.runs],
        "weights": args.weights,
        "top_k": args.top_k,
        "eval_k": args.eval_k,
        "rrf_k": args.rrf_k,
        "task_metrics": task_metrics,
        "overall": {
            metric: weighted_average(task_metrics, metric),
            f"recall@{args.eval_k}": weighted_average(task_metrics, f"recall@{args.eval_k}"),
            f"map@{args.eval_k}": weighted_average(task_metrics, f"map@{args.eval_k}"),
        },
        "runs": output_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
