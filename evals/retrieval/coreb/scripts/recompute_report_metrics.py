from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from coreb_sota.data import load_task, task_names
from coreb_sota.metrics import evaluate_run, weighted_average


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute stored CoREB run metrics with current protocol rules.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/coreb"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--eval-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = json.loads(args.input.read_text())
    split = args.split or report["split"]
    eval_k = args.eval_k or int(report.get("eval_k", 10))
    runs: dict[str, dict[str, dict[str, float]]] = report["runs"]

    task_metrics: dict[str, dict[str, Any]] = {}
    for task_name in task_names(args.tasks):
        task = load_task(args.data_dir, split, task_name)
        task_metrics[task_name] = evaluate_run(
            task.qrels,
            runs[task_name],
            k=eval_k,
            exclude_doc_ids=task.exclude_doc_ids,
        )

    metric = f"ndcg@{eval_k}"
    report["protocol"] = "coreb_official_anchor_exclusion"
    report["task_metrics"] = task_metrics
    report["overall"] = {
        metric: weighted_average(task_metrics, metric),
        f"recall@{eval_k}": weighted_average(task_metrics, f"recall@{eval_k}"),
        f"map@{eval_k}": weighted_average(task_metrics, f"map@{eval_k}"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
