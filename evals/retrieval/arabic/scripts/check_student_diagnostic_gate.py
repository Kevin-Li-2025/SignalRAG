#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def metric(row: dict[str, Any], key: str) -> float:
    metrics = row["metrics"]
    if key in metrics:
        return float(metrics[key])
    if key == "main_score" and "ndcg_at_10" in metrics:
        return float(metrics["ndcg_at_10"])
    raise KeyError(f"metric {key!r} not found in row {row.get('weight_label')!r}")


def rows_by_label(model_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["weight_label"]: row for row in model_summary["results"]}


def comparison_by_label(summary: dict[str, Any], label: str) -> dict[str, Any]:
    comparisons = {row["weight_label"]: row for row in summary.get("comparisons", [])}
    if label not in comparisons:
        raise KeyError(f"weight label {label!r} not present in summary comparisons")
    return comparisons[label]


def best_same_weight_delta(summary: dict[str, Any]) -> dict[str, Any]:
    comparisons = summary.get("comparisons", [])
    if not comparisons:
        raise ValueError("summary has no student comparisons")
    return max(comparisons, key=lambda row: row["delta"])


def best_vs_best_delta(summary: dict[str, Any], *, metric_key: str) -> dict[str, Any]:
    base_best = summary["base"]["best"]
    student_best = summary["student"]["best"]
    base_score = metric(base_best, metric_key)
    student_score = metric(student_best, metric_key)
    return {
        "base_weight_label": base_best["weight_label"],
        "student_weight_label": student_best["weight_label"],
        "base_score": base_score,
        "student_score": student_score,
        "delta": student_score - base_score,
    }


def summarize_split(
    summary: dict[str, Any],
    *,
    metric_key: str,
    model_card_label: str,
    selected_label: str | None = None,
) -> dict[str, Any]:
    best_same = best_same_weight_delta(summary)
    effective_selected_label = selected_label or best_same["weight_label"]
    selected = comparison_by_label(summary, effective_selected_label)
    model_card = comparison_by_label(summary, model_card_label)
    return {
        "query_count": summary["query_count"],
        "query_offset": summary.get("query_offset"),
        "query_stride": summary.get("query_stride"),
        "metric": metric_key,
        "best_vs_best": best_vs_best_delta(summary, metric_key=metric_key),
        "model_card": model_card,
        "best_same_weight": best_same,
        "selected_same_weight": selected,
    }


def pass_delta(row: dict[str, Any], required_delta: float) -> bool:
    return float(row["delta"]) >= required_delta


def check_gate(
    *,
    tune_summary: dict[str, Any],
    heldout_summary: dict[str, Any],
    metric_key: str,
    model_card_label: str,
    required_delta: float,
) -> dict[str, Any]:
    tune = summarize_split(
        tune_summary,
        metric_key=metric_key,
        model_card_label=model_card_label,
    )
    selected_label = tune["best_same_weight"]["weight_label"]
    heldout = summarize_split(
        heldout_summary,
        metric_key=metric_key,
        model_card_label=model_card_label,
        selected_label=selected_label,
    )
    criteria = {
        "tune_best_vs_best": pass_delta(tune["best_vs_best"], required_delta),
        "tune_model_card": pass_delta(tune["model_card"], required_delta),
        "heldout_best_vs_best": pass_delta(heldout["best_vs_best"], required_delta),
        "heldout_model_card": pass_delta(heldout["model_card"], required_delta),
        "heldout_tune_selected_same_weight": pass_delta(
            heldout["selected_same_weight"],
            required_delta,
        ),
    }
    return {
        "experiment": "student-diagnostic-gate",
        "metric": metric_key,
        "model_card_label": model_card_label,
        "required_delta": required_delta,
        "selected_same_weight_label_from_tune": selected_label,
        "tune": tune,
        "heldout": heldout,
        "criteria": criteria,
        "gate_pass": all(criteria.values()),
        "generated_outputs_committed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine tune and held-out BGE-M3 hybrid weight-sweep summaries into "
            "a strict student gate before any full-dev evaluation or publication."
        )
    )
    parser.add_argument("--tune-summary", required=True)
    parser.add_argument("--heldout-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--metric-key", default="main_score")
    parser.add_argument("--model-card-label", default="model_card")
    parser.add_argument("--required-delta", type=float, default=0.005)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = check_gate(
        tune_summary=read_json(Path(args.tune_summary)),
        heldout_summary=read_json(Path(args.heldout_summary)),
        metric_key=args.metric_key,
        model_card_label=args.model_card_label,
        required_delta=args.required_delta,
    )
    write_json(Path(args.output_json), result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.require_pass and not result["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
