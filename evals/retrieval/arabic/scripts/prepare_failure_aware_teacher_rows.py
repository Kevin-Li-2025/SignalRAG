#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_multisurface_teacher_rows as multisurface
from prepare_multisurface_teacher_rows import write_jsonl
from rerank_teacher_jsonl import write_json


DEFAULT_SURFACES = ["model_card", "training_default", "sparse_colbert", "colbert_heavy"]
DEFAULT_SURFACE_PRIORITIES = {
    "model_card": 1.6,
    "training_default": 1.4,
    "sparse_colbert": 0.85,
    "colbert_heavy": 0.75,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build failure-aware BGE-M3 student teacher rows from v75 component rows "
            "and v78 query-level failure analysis."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--failure-analysis-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-jsonl", default="teacher_train_failure_aware.jsonl")
    parser.add_argument("--eval-jsonl", default="teacher_eval_failure_aware.jsonl")
    parser.add_argument("--max-train-rows", type=int, default=512)
    parser.add_argument("--max-eval-rows", type=int, default=96)
    parser.add_argument("--heldout-ratio", type=float, default=0.15)
    parser.add_argument("--negatives-per-query", type=int, default=8)
    parser.add_argument("--surfaces", default=",".join(DEFAULT_SURFACES))
    parser.add_argument(
        "--surface-priorities",
        default=",".join(f"{label}={weight}" for label, weight in DEFAULT_SURFACE_PRIORITIES.items()),
    )
    parser.add_argument("--surface-hard-margin", type=float, default=0.05)
    parser.add_argument("--teacher-separation-margin", type=float, default=0.02)
    parser.add_argument("--teacher-hard-margin", type=float, default=0.10)
    parser.add_argument("--middle-margin", type=float, default=0.35)
    parser.add_argument("--min-surface-false-positive-available", type=int, default=1)
    parser.add_argument("--min-distinct-surfaces-available", type=int, default=2)
    parser.add_argument("--min-target-best-margin", type=float, default=0.0)
    parser.add_argument("--max-target-best-margin", type=float, default=None)
    parser.add_argument("--reranker-weight", type=float, default=0.50)
    parser.add_argument("--existing-weight", type=float, default=0.15)
    parser.add_argument("--surface-average-weight", type=float, default=0.35)
    parser.add_argument("--score-scale", type=float, default=1.0)
    parser.add_argument("--seed", default="20260605-v79-failure-aware-teacher")
    parser.add_argument("--allow-nontrain-source", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_label_values(value: str) -> dict[str, float]:
    result = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("label values must be comma-separated label=value entries")
        label, raw_value = part.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("label must not be empty")
        parsed = float(raw_value)
        if parsed < 0:
            raise ValueError("label values must be non-negative")
        result[label] = parsed
    if not result:
        raise ValueError("at least one label value is required")
    return result


def parse_surfaces(value: str) -> list[str]:
    surfaces = [part.strip() for part in value.split(",") if part.strip()]
    if not surfaces:
        raise ValueError("at least one surface is required")
    if len(set(surfaces)) != len(surfaces):
        raise ValueError("surfaces must be unique")
    return surfaces


def surface_stats_from_failure(
    analysis: dict[str, Any],
    surfaces: list[str],
) -> dict[str, dict[str, float]]:
    stats = analysis.get("aggregate", {}).get("surface_stats", {})
    missing = [surface for surface in surfaces if surface not in stats]
    if missing:
        raise ValueError(f"failure analysis missing surfaces: {', '.join(missing)}")
    return {surface: stats[surface] for surface in surfaces}


def surface_risk(stats: dict[str, float], *, required_delta: float) -> float:
    pass_fraction = float(stats.get("pass_fraction") or 0.0)
    regression_fraction = float(stats.get("regression_fraction") or 0.0)
    mean_delta = float(stats.get("mean_delta") or 0.0)
    mean_penalty = max(0.0, required_delta - mean_delta) / max(required_delta, 1e-12)
    return (1.0 - pass_fraction) + regression_fraction + mean_penalty


def allocate_counts(scores: dict[str, float], *, total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    positive_labels = [label for label, score in scores.items() if score > 0]
    if not positive_labels:
        raise ValueError("at least one score must be positive")
    if total < len(positive_labels):
        raise ValueError("total is smaller than the number of positive labels")
    counts = {label: 0 for label in scores}
    for label in positive_labels:
        counts[label] = 1
    remaining = total - len(positive_labels)
    total_score = sum(scores[label] for label in positive_labels)
    fractional = {
        label: (scores[label] / total_score) * remaining
        for label in positive_labels
    }
    floors = {label: int(value) for label, value in fractional.items()}
    for label, value in floors.items():
        counts[label] += value
    leftover = remaining - sum(floors.values())
    order = sorted(
        positive_labels,
        key=lambda label: (fractional[label] - floors[label], scores[label], label),
        reverse=True,
    )
    for label in order[:leftover]:
        counts[label] += 1
    return counts


def format_counts(counts: dict[str, int]) -> str:
    return ",".join(f"{label}={count}" for label, count in counts.items())


def format_weights(weights: dict[str, float]) -> str:
    return ",".join(f"{label}={weight:.6g}" for label, weight in weights.items())


def build_failure_surface_plan(
    failure_analysis: dict[str, Any],
    *,
    surfaces: list[str],
    surface_priorities: dict[str, float],
    negatives_per_query: int,
) -> dict[str, Any]:
    required_delta = float(failure_analysis.get("required_delta") or 0.005)
    stats = surface_stats_from_failure(failure_analysis, surfaces)
    priorities = {surface: float(surface_priorities.get(surface, 1.0)) for surface in surfaces}
    risk = {
        surface: surface_risk(stats[surface], required_delta=required_delta)
        for surface in surfaces
    }
    weighted_risk = {
        surface: risk[surface] * priorities[surface]
        for surface in surfaces
    }
    counts = allocate_counts(weighted_risk, total=negatives_per_query)
    source_weights = {
        surface: max(weighted_risk[surface], 1e-6)
        for surface in surfaces
    }
    return {
        "required_delta": required_delta,
        "surfaces": surfaces,
        "surface_priorities": priorities,
        "surface_failure_stats": stats,
        "surface_risk": risk,
        "weighted_surface_risk": weighted_risk,
        "surface_negative_counts": counts,
        "surface_average_source_weights": source_weights,
        "surface_negative_counts_spec": format_counts(counts),
        "surface_average_source_weights_spec": format_weights(source_weights),
    }


def multisurface_namespace(args: argparse.Namespace, plan: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        heldout_ratio=args.heldout_ratio,
        negatives_per_query=args.negatives_per_query,
        surface_negative_counts=plan["surface_negative_counts_spec"],
        min_surface_false_positive_available=args.min_surface_false_positive_available,
        min_distinct_surfaces_available=args.min_distinct_surfaces_available,
        surface_hard_margin=args.surface_hard_margin,
        teacher_separation_margin=args.teacher_separation_margin,
        teacher_hard_margin=args.teacher_hard_margin,
        middle_margin=args.middle_margin,
        reranker_weight=args.reranker_weight,
        existing_weight=args.existing_weight,
        surface_average_weight=args.surface_average_weight,
        surface_average_source_weights=plan["surface_average_source_weights_spec"],
        score_scale=args.score_scale,
        min_target_best_margin=args.min_target_best_margin,
        max_target_best_margin=args.max_target_best_margin,
        seed=args.seed,
        allow_nontrain_source=args.allow_nontrain_source,
        force=args.force,
    )


def prepare_failure_aware_splits(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    failure_analysis = json.loads(Path(args.failure_analysis_json).read_text(encoding="utf-8"))
    surfaces = parse_surfaces(args.surfaces)
    priorities = parse_label_values(args.surface_priorities)
    plan = build_failure_surface_plan(
        failure_analysis,
        surfaces=surfaces,
        surface_priorities=priorities,
        negatives_per_query=args.negatives_per_query,
    )
    train_rows, eval_rows, summary = multisurface.prepare_splits(
        multisurface_namespace(args, plan)
    )
    summary["failure_aware_selector"] = {
        "source": "prepare_failure_aware_teacher_rows.py",
        "failure_analysis_json": args.failure_analysis_json,
        "plan": plan,
        "rationale": (
            "Convert v78 dev-slice query-level failures into train-split surface "
            "risk weights without using dev queries as training examples."
        ),
    }
    summary["surface_target"]["method"] = "failure_weighted_row_minmax_multisurface_target"
    summary["surface_target"]["rationale"] = (
        "Use v78 failure-derived surface weights so model-card and base-best "
        "stability surfaces influence both negative selection quotas and the "
        "BGE-M3 surface-average target source."
    )
    return train_rows, eval_rows, summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / args.train_jsonl
    eval_path = output_dir / args.eval_jsonl
    summary_path = output_dir / "summary.json"
    if not args.force:
        for path in [train_path, eval_path, summary_path]:
            if path.exists():
                raise FileExistsError(f"{path} exists; pass --force to overwrite")
    train_rows, eval_rows, summary = prepare_failure_aware_splits(args)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    summary["outputs"] = {
        "train_jsonl": train_path.name,
        "eval_jsonl": eval_path.name,
        "summary_json": summary_path.name,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
