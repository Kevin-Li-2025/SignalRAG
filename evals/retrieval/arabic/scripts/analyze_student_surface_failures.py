#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
    return rows


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, path
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"split label is empty in {value!r}")
    return label, Path(path)


def metric_value(row: dict[str, Any], side: str, metric_key: str) -> float:
    try:
        return float(row[side]["metrics"][metric_key])
    except KeyError as exc:
        raise ValueError(
            f"per-query row for query={row.get('query_id')} weight={row.get('weight_label')} "
            f"is missing {side}.metrics.{metric_key}"
        ) from exc


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_deltas(deltas: list[float], required_delta: float) -> dict[str, Any]:
    regressions = [value for value in deltas if value < 0.0]
    passes = [value for value in deltas if value >= required_delta]
    return {
        "count": len(deltas),
        "mean_delta": mean(deltas),
        "min_delta": min(deltas) if deltas else None,
        "max_delta": max(deltas) if deltas else None,
        "pass_count": len(passes),
        "pass_fraction": len(passes) / len(deltas) if deltas else None,
        "regression_count": len(regressions),
        "regression_fraction": len(regressions) / len(deltas) if deltas else None,
    }


def surface_stats(rows: list[dict[str, Any]], required_delta: float) -> dict[str, Any]:
    by_label: dict[str, list[float]] = {}
    for row in rows:
        if "delta" not in row:
            continue
        by_label.setdefault(str(row["weight_label"]), []).append(float(row["delta"]))
    return {
        label: summarize_deltas(deltas, required_delta)
        for label, deltas in sorted(by_label.items())
    }


def focus_rows(
    rows: list[dict[str, Any]],
    *,
    focus_labels: list[str],
    metric_key: str,
    worst_limit: int,
) -> dict[str, Any]:
    result = {}
    for label in focus_labels:
        label_rows = [row for row in rows if row.get("weight_label") == label and "delta" in row]
        worst = sorted(label_rows, key=lambda row: float(row["delta"]))[:worst_limit]
        best = sorted(label_rows, key=lambda row: float(row["delta"]), reverse=True)[:worst_limit]
        result[label] = {
            "worst_queries": [
                {
                    "query_id": row["query_id"],
                    "delta": float(row["delta"]),
                    "base_score": metric_value(row, "base", metric_key),
                    "student_score": metric_value(row, "student", metric_key),
                    "base_top_docids": row["base"].get("top_docids", []),
                    "student_top_docids": row["student"].get("top_docids", []),
                }
                for row in worst
            ],
            "best_queries": [
                {
                    "query_id": row["query_id"],
                    "delta": float(row["delta"]),
                    "base_score": metric_value(row, "base", metric_key),
                    "student_score": metric_value(row, "student", metric_key),
                }
                for row in best
            ],
        }
    return result


def split_query_summary(
    *,
    label: str,
    rows: list[dict[str, Any]],
    focus_labels: list[str],
    metric_key: str,
    required_delta: float,
    worst_limit: int,
) -> dict[str, Any]:
    query_ids = sorted({str(row["query_id"]) for row in rows})
    return {
        "label": label,
        "query_count": len(query_ids),
        "row_count": len(rows),
        "surface_stats": surface_stats(rows, required_delta),
        "focus": focus_rows(
            rows,
            focus_labels=focus_labels,
            metric_key=metric_key,
            worst_limit=worst_limit,
        ),
    }


def add_focus_from_gate(focus_labels: list[str], gate_summary: dict[str, Any] | None) -> list[str]:
    labels = list(focus_labels)
    if gate_summary:
        selected = gate_summary.get("selected_same_weight_label")
        if selected:
            labels.append(str(selected))
    deduped = []
    seen = set()
    for label in labels:
        if label not in seen:
            deduped.append(label)
            seen.add(label)
    return deduped


def analyze_failures(
    *,
    labeled_rows: list[tuple[str, list[dict[str, Any]]]],
    gate_summary: dict[str, Any] | None,
    focus_labels: list[str],
    metric_key: str,
    required_delta: float,
    worst_limit: int,
) -> dict[str, Any]:
    focus_labels = add_focus_from_gate(focus_labels, gate_summary)
    all_rows = [row for _label, rows in labeled_rows for row in rows]
    return {
        "experiment": "student-surface-failure-analysis",
        "metric": metric_key,
        "required_delta": required_delta,
        "focus_labels": focus_labels,
        "gate_summary": {
            "gate_pass": gate_summary.get("gate_pass") if gate_summary else None,
            "selected_same_weight_label": gate_summary.get("selected_same_weight_label")
            if gate_summary
            else None,
            "aggregate": gate_summary.get("aggregate") if gate_summary else None,
        },
        "aggregate": {
            "query_count": len({str(row["query_id"]) for row in all_rows}),
            "row_count": len(all_rows),
            "surface_stats": surface_stats(all_rows, required_delta),
            "focus": focus_rows(
                all_rows,
                focus_labels=focus_labels,
                metric_key=metric_key,
                worst_limit=worst_limit,
            ),
        },
        "splits": [
            split_query_summary(
                label=label,
                rows=rows,
                focus_labels=focus_labels,
                metric_key=metric_key,
                required_delta=required_delta,
                worst_limit=worst_limit,
            )
            for label, rows in labeled_rows
        ],
        "decision_implications": [
            "Use this analysis to inspect query/surface regressions before launching another GPU run.",
            "A student should not be scaled when gains are concentrated in one slice or one weak surface.",
            "The next teacher-row selector or objective should explicitly target model-card and base-best stability.",
        ],
        "generated_outputs_committed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze query-level surface failures from sweep_bge_m3_hybrid_weights "
            "--write-per-query outputs."
        )
    )
    parser.add_argument(
        "--split-per-query",
        action="append",
        required=True,
        help="Split per-query JSONL as label=path or just path. Provide one or more.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--gate-summary", default="")
    parser.add_argument("--metric-key", default="main_score")
    parser.add_argument("--required-delta", type=float, default=0.005)
    parser.add_argument("--focus-label", action="append", default=["model_card"])
    parser.add_argument("--worst-limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate_summary = read_json(Path(args.gate_summary)) if args.gate_summary else None
    labeled_rows = [
        (label, read_jsonl(path))
        for label, path in (parse_labeled_path(value) for value in args.split_per_query)
    ]
    result = analyze_failures(
        labeled_rows=labeled_rows,
        gate_summary=gate_summary,
        focus_labels=args.focus_label,
        metric_key=args.metric_key,
        required_delta=args.required_delta,
        worst_limit=args.worst_limit,
    )
    write_json(Path(args.output_json), result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
