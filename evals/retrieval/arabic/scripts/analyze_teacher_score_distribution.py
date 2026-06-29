#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_teacher_jsonl import mean, pearson, write_json


MARGIN_BINS = [
    ("negative", float("-inf"), 0.0),
    ("0_to_0.5", 0.0, 0.5),
    ("0.5_to_1", 0.5, 1.0),
    ("1_to_2", 1.0, 2.0),
    ("2_to_4", 2.0, 4.0),
    ("4_plus", 4.0, float("inf")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze teacher-score distribution in reranker-scored distillation "
            "JSONL before launching another student training run."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", default="score_distribution_summary.json")
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument(
        "--temperature",
        action="append",
        type=float,
        default=None,
        help="Softmax temperature to summarize teacher target entropy. Can be repeated.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def float_list(values: Any) -> list[float]:
    if not values:
        return []
    return [float(value) for value in values]


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean(values),
        "min": min(values) if values else None,
        "p05": quantile(values, 0.05),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "max": max(values) if values else None,
    }


def margin_bin(value: float) -> str:
    for label, low, high in MARGIN_BINS:
        if low <= value < high:
            return label
    return MARGIN_BINS[-1][0]


def softmax(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        raise ValueError("scores must not be empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [score / temperature for score in scores]
    maximum = max(scaled)
    exps = [math.exp(score - maximum) for score in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def normalized_entropy(distribution: list[float]) -> float:
    if len(distribution) <= 1:
        return 0.0
    entropy = -sum(prob * math.log(prob) for prob in distribution if prob > 0)
    return entropy / math.log(len(distribution))


def row_scores(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    return float_list(row.get("pos_scores")), float_list(row.get("neg_scores"))


def best_margin(pos_scores: list[float], neg_scores: list[float]) -> float:
    return max(pos_scores) - max(neg_scores)


def analyze_rows(rows: list[dict[str, Any]], temperatures: list[float]) -> dict[str, Any]:
    rows_with_scores = 0
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    best_margins: list[float] = []
    all_margins: list[float] = []
    mean_margins: list[float] = []
    negative_margins: list[float] = []
    best_margin_bins: Counter[str] = Counter()
    negative_margin_bins: Counter[str] = Counter()
    neg_source_counts: Counter[str] = Counter()
    pos_above_max_neg = 0
    all_pos_above_all_neg = 0

    teacher_margin_for_hybrid: list[float] = []
    hybrid_margin_for_teacher: list[float] = []

    distribution_stats: dict[str, dict[str, list[float]]] = {
        f"{temperature:g}": {
            "pos_probability": [],
            "max_negative_probability": [],
            "normalized_entropy": [],
        }
        for temperature in temperatures
    }

    for row in rows:
        pos_scores, neg_scores = row_scores(row)
        if not pos_scores or not neg_scores:
            continue
        rows_with_scores += 1
        positive_scores.extend(pos_scores)
        negative_scores.extend(neg_scores)
        for source in row.get("neg_sources", []):
            neg_source_counts[str(source)] += 1

        max_pos = max(pos_scores)
        min_pos = min(pos_scores)
        max_neg = max(neg_scores)
        margin = max_pos - max_neg
        best_margins.append(margin)
        all_margins.append(min_pos - max_neg)
        mean_margins.append((mean(pos_scores) or 0.0) - (mean(neg_scores) or 0.0))
        best_margin_bins[margin_bin(margin)] += 1
        if margin > 0:
            pos_above_max_neg += 1
        if min_pos > max_neg:
            all_pos_above_all_neg += 1

        for neg_score in neg_scores:
            neg_margin = max_pos - neg_score
            negative_margins.append(neg_margin)
            negative_margin_bins[margin_bin(neg_margin)] += 1

        group_scores = [max_pos, *neg_scores]
        for temperature in temperatures:
            label = f"{temperature:g}"
            distribution = softmax(group_scores, temperature)
            distribution_stats[label]["pos_probability"].append(distribution[0])
            distribution_stats[label]["max_negative_probability"].append(max(distribution[1:]))
            distribution_stats[label]["normalized_entropy"].append(normalized_entropy(distribution))

        hybrid_pos = float_list(row.get("bge_m3_hybrid_pos_scores"))
        hybrid_neg = float_list(row.get("bge_m3_hybrid_neg_scores"))
        if hybrid_pos and hybrid_neg:
            teacher_margin_for_hybrid.append(margin)
            hybrid_margin_for_teacher.append(max(hybrid_pos) - max(hybrid_neg))

    by_temperature = {
        label: {
            "mean_pos_probability": mean(values["pos_probability"]),
            "mean_max_negative_probability": mean(values["max_negative_probability"]),
            "mean_normalized_entropy": mean(values["normalized_entropy"]),
            "pos_probability": numeric_stats(values["pos_probability"]),
            "normalized_entropy": numeric_stats(values["normalized_entropy"]),
        }
        for label, values in distribution_stats.items()
    }

    rows_count = len(rows)
    return {
        "rows_seen": rows_count,
        "rows_with_scores": rows_with_scores,
        "positive_pairs": len(positive_scores),
        "negative_pairs": len(negative_scores),
        "positives_above_max_negative": {
            "rows": pos_above_max_neg,
            "fraction_of_rows_with_scores": (
                pos_above_max_neg / rows_with_scores if rows_with_scores else None
            ),
        },
        "all_positives_above_all_negatives": {
            "rows": all_pos_above_all_neg,
            "fraction_of_rows_with_scores": (
                all_pos_above_all_neg / rows_with_scores if rows_with_scores else None
            ),
        },
        "negative_source_counts": dict(sorted(neg_source_counts.items())),
        "score_stats": {
            "positive": numeric_stats(positive_scores),
            "negative": numeric_stats(negative_scores),
        },
        "margin_stats": {
            "best_pos_minus_max_neg": numeric_stats(best_margins),
            "min_pos_minus_max_neg": numeric_stats(all_margins),
            "mean_pos_minus_mean_neg": numeric_stats(mean_margins),
            "best_pos_minus_each_neg": numeric_stats(negative_margins),
        },
        "margin_bins": {
            "best_pos_minus_max_neg": dict(best_margin_bins),
            "best_pos_minus_each_neg": dict(negative_margin_bins),
        },
        "teacher_distribution_by_temperature": by_temperature,
        "teacher_vs_bge_m3_hybrid_margin": {
            "paired_rows": len(teacher_margin_for_hybrid),
            "pearson": pearson(teacher_margin_for_hybrid, hybrid_margin_for_teacher),
            "teacher_margin": numeric_stats(teacher_margin_for_hybrid),
            "bge_m3_hybrid_margin": numeric_stats(hybrid_margin_for_teacher),
        },
        "sampling_implications": sampling_implications(
            best_margin_bins=best_margin_bins,
            negative_margin_bins=negative_margin_bins,
            rows_with_scores=rows_with_scores,
            negative_count=len(negative_margins),
        ),
    }


def fraction(counter: Counter[str], labels: list[str], denominator: int) -> float | None:
    if denominator == 0:
        return None
    return sum(counter.get(label, 0) for label in labels) / denominator


def sampling_implications(
    *,
    best_margin_bins: Counter[str],
    negative_margin_bins: Counter[str],
    rows_with_scores: int,
    negative_count: int,
) -> list[str]:
    implications: list[str] = []
    high_margin_row_fraction = fraction(best_margin_bins, ["4_plus"], rows_with_scores)
    low_margin_row_fraction = fraction(
        best_margin_bins,
        ["negative", "0_to_0.5", "0.5_to_1"],
        rows_with_scores,
    )
    mid_negative_fraction = fraction(
        negative_margin_bins,
        ["1_to_2", "2_to_4"],
        negative_count,
    )

    if high_margin_row_fraction is not None and high_margin_row_fraction > 0.5:
        implications.append(
            "Many selected rows have very high best-positive margins; avoid choosing only top-margin rows."
        )
    if low_margin_row_fraction is not None and low_margin_row_fraction < 0.2:
        implications.append(
            "Low-margin rows are underrepresented; add borderline teacher cases for calibration."
        )
    if mid_negative_fraction is not None and mid_negative_fraction < 0.3:
        implications.append(
            "Middle-score negatives are underrepresented; stratified sampling should add mid-band negatives."
        )
    implications.append(
        "Next training rows should mix positives with hard, middle-score, and easy negatives and keep a query-disjoint held-out slice."
    )
    return implications


def load_rows(path: Path, query_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if query_limit and len(rows) >= query_limit:
                break
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def main() -> None:
    args = parse_args()
    temperatures = args.temperature or [1.0, 0.5, 0.2, 0.1]
    if any(temperature <= 0 for temperature in temperatures):
        raise ValueError("--temperature values must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_json
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists; pass --force to overwrite")

    rows = load_rows(Path(args.input_jsonl), args.query_limit)
    summary = {
        "input_jsonl": args.input_jsonl,
        "query_limit": args.query_limit,
        "temperatures": temperatures,
        **analyze_rows(rows, temperatures),
        "raw_training_data_committed": False,
        "model_checkpoints_committed": False,
    }
    write_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
