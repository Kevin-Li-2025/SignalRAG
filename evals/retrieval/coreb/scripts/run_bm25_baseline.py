from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from coreb_sota.data import load_task, task_names
from coreb_sota.lexical import BM25Index, c2c_exclusions
from coreb_sota.metrics import evaluate_run, weighted_average


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic BM25/code-aware CoREB baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/coreb"))
    parser.add_argument("--split", default="release_v2603")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("reports/bm25_release_v2603.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    task_metrics: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, dict[str, float]]] = {}

    for task_name in task_names(args.tasks):
        task = load_task(args.data_dir, args.split, task_name)
        index = BM25Index.build(task.corpus)
        run: dict[str, dict[str, float]] = {}
        for query_id, query_row in tqdm(task.queries.items(), desc=f"bm25:{task_name}"):
            exclude = c2c_exclusions(query_row) if task_name == "code2code" else set()
            run[query_id] = index.search(query_row["query"], top_k=args.top_k, exclude=exclude)
        metrics = evaluate_run(task.qrels, run, k=args.eval_k, exclude_doc_ids=task.exclude_doc_ids)
        task_metrics[task_name] = metrics
        runs[task_name] = run

    metric = f"ndcg@{args.eval_k}"
    report = {
        "method": "bm25_code_aware",
        "split": args.split,
        "top_k": args.top_k,
        "eval_k": args.eval_k,
        "elapsed_seconds": time.perf_counter() - started,
        "task_metrics": task_metrics,
        "overall": {
            metric: weighted_average(task_metrics, metric),
            f"recall@{args.eval_k}": weighted_average(task_metrics, f"recall@{args.eval_k}"),
            f"map@{args.eval_k}": weighted_average(task_metrics, f"map@{args.eval_k}"),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
