#!/usr/bin/env python3
"""Fail-closed summary for the corrected BF16/NF4 FinanceMTEB matrix."""

from __future__ import annotations

import argparse
import json
import math
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


def summarize(run_root: Path) -> dict:
    rows = []
    candidate_hashes: dict[str, str] = {}
    for precision, task in ARMS:
        root = run_root / precision / task
        train = read(root / "train_strategy.json")
        cv = read(root / "train_nested_cv.json")
        test = read(root / "frozen_test.json")
        audit = read(root / "order_invariance.json")
        if audit.get("status") != "complete" or audit.get("all_tasks_order_invariant") is not True:
            raise ValueError(f"Order-invariance audit failed for {precision}/{task}")
        if len(train.get("tasks", [])) != 1 or len(test.get("tasks", [])) != 1:
            raise ValueError(f"Expected exactly one task in {precision}/{task}")
        train_task = train["tasks"][0]
        test_task = test["tasks"][0]
        if train_task["leaderboard_name"] != task or test_task["leaderboard_name"] != task:
            raise ValueError(f"Task identity mismatch for {precision}/{task}")
        if train.get("split") != "train" or train_task.get("split") != "train":
            raise ValueError(f"Selection must use train only for {precision}/{task}")
        if test_task.get("split") != "test":
            raise ValueError(f"Frozen evaluation must use test for {precision}/{task}")
        if train["model"] != test["model"]:
            raise ValueError(f"Model identity mismatch for {precision}/{task}")
        for name, report, split in (("CV", cv, "train"), ("audit", audit, "test")):
            if len(report.get("tasks", [])) != 1:
                raise ValueError(f"Expected one {name} task for {precision}/{task}")
            row = report["tasks"][0]
            if row.get("leaderboard_name") != task or row.get("split") != split:
                raise ValueError(f"{name} identity/split mismatch for {precision}/{task}")
            if row.get("dataset") != test_task["dataset"]:
                raise ValueError(f"{name} dataset mismatch for {precision}/{task}")
        if cv.get("split") != "train" or train_task["dataset"] != test_task["dataset"]:
            raise ValueError(f"Train/CV identity mismatch for {precision}/{task}")
        selected = {key: train_task["best"][key] for key in ("method", "feature", "alpha")}
        audited = audit["tasks"][0]
        if test_task["strategy"] != selected or audited.get("strategy") != selected:
            raise ValueError(f"Frozen strategy differs from train selection for {precision}/{task}")
        if audited.get("order_invariant") is not True:
            raise ValueError(f"Per-task order audit failed for {precision}/{task}")
        for key in ("num_queries", "num_pairs"):
            if type(test_task[key]) is not int or test_task[key] <= 0:
                raise ValueError(f"Invalid {key} for {precision}/{task}")
        if audited["num_queries"] != test_task["num_queries"]:
            raise ValueError(f"Audit query count mismatch for {precision}/{task}")
        seeds = audited["seeds"]
        if len(seeds) != 4 or {row["seed"] for row in seeds} != {20260905, 2234, 314159, 8675309}:
            raise ValueError(f"Incomplete order seed coverage for {precision}/{task}")
        for metric in ("map", "mrr", "ndcg@10"):
            value = test_task["metrics"][metric]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid frozen metric for {precision}/{task}")
        for seed in seeds:
            if (
                seed.get("metric_invariant") is not True
                or seed.get("same_candidate_coverage") is not True
            ):
                raise ValueError(f"Incomplete order audit for {precision}/{task}")
            if seed["candidate_count"] != test_task["num_pairs"]:
                raise ValueError(f"Candidate count mismatch for {precision}/{task}")
            digest = seed["candidate_set_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(f"Invalid candidate identity for {precision}/{task}")
            if candidate_hashes.setdefault(task, digest) != digest:
                raise ValueError(
                    f"Candidate identity differs across seeds or precisions for {task}"
                )
            for metric, value in test_task["metrics"].items():
                if not math.isclose(seed["metrics"][metric], value, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(
                        f"Order audit metrics differ from frozen result for {precision}/{task}"
                    )
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
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.run_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
