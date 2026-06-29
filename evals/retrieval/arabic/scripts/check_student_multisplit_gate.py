#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from check_student_diagnostic_gate import (
    comparison_by_label,
    read_json,
    summarize_split,
    write_json,
)


def parse_split_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, path
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"split label is empty in {value!r}")
    return label, Path(path)


def common_comparison_labels(summaries: list[dict[str, Any]]) -> list[str]:
    common: set[str] | None = None
    for summary in summaries:
        labels = {row["weight_label"] for row in summary.get("comparisons", [])}
        common = labels if common is None else common.intersection(labels)
    return sorted(common or [])


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def select_same_weight_label(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    labels = common_comparison_labels(summaries)
    if not labels:
        raise ValueError("No common same-weight labels across split summaries")
    by_label = {}
    for label in labels:
        deltas = [float(comparison_by_label(summary, label)["delta"]) for summary in summaries]
        by_label[label] = {
            "label": label,
            "mean_delta": mean(deltas),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "deltas": deltas,
        }
    selected = max(
        by_label.values(),
        key=lambda item: (
            item["mean_delta"],
            item["min_delta"],
            item["label"],
        ),
    )
    return {
        "selected_label": selected["label"],
        "selection_metric": "highest mean same-weight delta across all supplied splits",
        "by_label": by_label,
    }


def aggregate_rows(rows: list[dict[str, Any]], required_delta: float) -> dict[str, Any]:
    deltas = [float(row["delta"]) for row in rows]
    pass_count = sum(delta >= required_delta for delta in deltas)
    return {
        "count": len(rows),
        "mean_delta": mean(deltas),
        "min_delta": min(deltas) if deltas else None,
        "max_delta": max(deltas) if deltas else None,
        "pass_count": pass_count,
        "pass_fraction": pass_count / len(rows) if rows else None,
        "all_pass": pass_count == len(rows) if rows else False,
    }


def check_multisplit_gate(
    *,
    labeled_summaries: list[tuple[str, dict[str, Any]]],
    metric_key: str,
    model_card_label: str,
    required_delta: float,
    min_pass_fraction: float,
) -> dict[str, Any]:
    if not 0.0 < min_pass_fraction <= 1.0:
        raise ValueError("--min-pass-fraction must be in (0, 1]")
    if len(labeled_summaries) < 2:
        raise ValueError("At least two split summaries are required")

    summaries = [summary for _label, summary in labeled_summaries]
    selection = select_same_weight_label(summaries)
    selected_label = selection["selected_label"]
    splits = [
        {
            "label": label,
            **summarize_split(
                summary,
                metric_key=metric_key,
                model_card_label=model_card_label,
                selected_label=selected_label,
            ),
        }
        for label, summary in labeled_summaries
    ]
    aggregate = {
        "best_vs_best": aggregate_rows(
            [split["best_vs_best"] for split in splits],
            required_delta,
        ),
        "model_card": aggregate_rows(
            [split["model_card"] for split in splits],
            required_delta,
        ),
        "selected_same_weight": aggregate_rows(
            [split["selected_same_weight"] for split in splits],
            required_delta,
        ),
    }
    criteria = {
        f"{surface}_pass_fraction": (
            stats["pass_fraction"] is not None
            and stats["pass_fraction"] >= min_pass_fraction
        )
        for surface, stats in aggregate.items()
    }
    criteria.update(
        {
            f"{surface}_mean_delta": (
                stats["mean_delta"] is not None
                and stats["mean_delta"] >= required_delta
            )
            for surface, stats in aggregate.items()
        }
    )
    return {
        "experiment": "student-multisplit-diagnostic-gate",
        "metric": metric_key,
        "model_card_label": model_card_label,
        "required_delta": required_delta,
        "min_pass_fraction": min_pass_fraction,
        "selected_same_weight_label": selected_label,
        "selection": selection,
        "splits": splits,
        "aggregate": aggregate,
        "criteria": criteria,
        "gate_pass": all(criteria.values()),
        "generated_outputs_committed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple BGE-M3 hybrid weight-sweep summaries into a "
            "split-stability gate before any larger student run or claim."
        )
    )
    parser.add_argument(
        "--split-summary",
        action="append",
        required=True,
        help="Split summary as label=path or just path. Provide at least two.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--metric-key", default="main_score")
    parser.add_argument("--model-card-label", default="model_card")
    parser.add_argument("--required-delta", type=float, default=0.005)
    parser.add_argument("--min-pass-fraction", type=float, default=1.0)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labeled_summaries = [
        (label, read_json(path))
        for label, path in (parse_split_arg(value) for value in args.split_summary)
    ]
    result = check_multisplit_gate(
        labeled_summaries=labeled_summaries,
        metric_key=args.metric_key,
        model_card_label=args.model_card_label,
        required_delta=args.required_delta,
        min_pass_fraction=args.min_pass_fraction,
    )
    write_json(Path(args.output_json), result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.require_pass and not result["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
