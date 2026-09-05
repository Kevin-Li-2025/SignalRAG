#!/usr/bin/env python3
"""Fail-closed summary for the corrected BF16/NF4 FinanceMTEB matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = (
    ("bf16", "FinEvaReranking"),
    ("bf16", "DISCFinLLMReranking"),
    ("bnb_nf4", "FinEvaReranking"),
    ("bnb_nf4", "DISCFinLLMReranking"),
)


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for precision, task in ARMS:
        root = args.run_root / precision / task
        train = read(root / "train_strategy.json")
        cv = read(root / "train_nested_cv.json")
        test = read(root / "frozen_test.json")
        audit = read(root / "order_invariance.json")
        if audit.get("status") != "complete" or not audit.get("all_tasks_order_invariant"):
            raise ValueError(f"Order-invariance audit failed for {precision}/{task}")
        if len(train.get("tasks", [])) != 1 or len(test.get("tasks", [])) != 1:
            raise ValueError(f"Expected exactly one task in {precision}/{task}")
        train_task = train["tasks"][0]
        test_task = test["tasks"][0]
        if train_task["leaderboard_name"] != task or test_task["leaderboard_name"] != task:
            raise ValueError(f"Task identity mismatch for {precision}/{task}")
        rows.append(
            {
                "precision": precision,
                "task": task,
                "model": test["model"],
                "train_selected_strategy": train_task["best"],
                "nested_cv": cv,
                "frozen_test_metrics": test_task["metrics"],
                "num_test_queries": test_task["num_queries"],
                "num_test_pairs": test_task["num_pairs"],
                "order_invariance": audit["all_tasks_order_invariant"],
            }
        )

    averages = {}
    for precision in ("bf16", "bnb_nf4"):
        precision_rows = [row for row in rows if row["precision"] == precision]
        averages[precision] = {
            metric: sum(row["frozen_test_metrics"][metric] for row in precision_rows)
            / len(precision_rows)
            for metric in ("map", "mrr", "ndcg@10")
        }
    payload = {
        "status": "complete",
        "protocol_version": "finmteb-zh-corrected-v1",
        "arms": rows,
        "macro_average_by_precision": averages,
        "nf4_minus_bf16": {
            metric: averages["bnb_nf4"][metric] - averages["bf16"][metric]
            for metric in ("map", "mrr", "ndcg@10")
        },
        "claim_boundary": "Strategies were selected within each precision on train only; test was frozen and is not a selection set.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
