#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_teacher_score_distribution import analyze_rows
from prepare_student_distill_splits import source_split, stable_fraction
from rerank_teacher_jsonl import mean, write_json


DEFAULT_SURFACE_COUNTS = "model_card=2,colbert_heavy=2,sparse_colbert=2,training_default=2"
FALLBACK_ORDER = ["surface_false_positive", "teacher_hard", "middle", "easy"]
MARGIN_BIN_ORDER = ["negative", "0_to_0.02", "0.02_to_0.05", "0.05_to_0.15", "0.15_plus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build query-disjoint teacher rows from v75 BGE-M3 component-augmented "
            "rows. Negatives are balanced across model-card/base-best fusion "
            "surfaces so the next student smoke targets the surfaces that failed "
            "the v73/v74 gates."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-jsonl", default="teacher_train_multisurface.jsonl")
    parser.add_argument("--eval-jsonl", default="teacher_eval_multisurface.jsonl")
    parser.add_argument("--max-train-rows", type=int, default=512)
    parser.add_argument("--max-eval-rows", type=int, default=96)
    parser.add_argument("--heldout-ratio", type=float, default=0.15)
    parser.add_argument("--negatives-per-query", type=int, default=8)
    parser.add_argument(
        "--surface-negative-counts",
        default=DEFAULT_SURFACE_COUNTS,
        help=(
            "Comma-separated surface=count targets, for example "
            "model_card=2,colbert_heavy=2,sparse_colbert=2,training_default=2."
        ),
    )
    parser.add_argument("--min-surface-false-positive-available", type=int, default=1)
    parser.add_argument("--min-distinct-surfaces-available", type=int, default=1)
    parser.add_argument(
        "--surface-hard-margin",
        type=float,
        default=0.0,
        help="A negative is a surface false positive when pos_surface - neg_surface <= this.",
    )
    parser.add_argument(
        "--teacher-separation-margin",
        type=float,
        default=0.05,
        help="Require the blended teacher target to separate a surface false positive by this much.",
    )
    parser.add_argument("--teacher-hard-margin", type=float, default=0.10)
    parser.add_argument("--middle-margin", type=float, default=0.35)
    parser.add_argument("--reranker-weight", type=float, default=0.60)
    parser.add_argument("--existing-weight", type=float, default=0.25)
    parser.add_argument("--surface-average-weight", type=float, default=0.15)
    parser.add_argument(
        "--anti-regression-surfaces",
        default="",
        help=(
            "Optional comma-separated surfaces that should add a direct "
            "anti-regression target source. For those surfaces, false-positive "
            "negatives are kept hard but separated below the positive."
        ),
    )
    parser.add_argument("--anti-regression-weight", type=float, default=0.0)
    parser.add_argument(
        "--anti-regression-margin",
        type=float,
        default=0.20,
        help="Target margin used by the optional anti-regression source.",
    )
    parser.add_argument(
        "--anti-regression-surface-hard-margin",
        type=float,
        default=None,
        help=(
            "Surface false-positive margin for the anti-regression source. "
            "Defaults to --surface-hard-margin."
        ),
    )
    parser.add_argument(
        "--surface-average-source-weights",
        default="",
        help=(
            "Optional comma-separated surface=weight values used when building "
            "the BGE-M3 surface-average target source. Missing surfaces default "
            "to 1.0."
        ),
    )
    parser.add_argument("--score-scale", type=float, default=1.0)
    parser.add_argument("--min-target-best-margin", type=float, default=0.0)
    parser.add_argument("--max-target-best-margin", type=float, default=None)
    parser.add_argument("--seed", default="20260605-v76-multisurface-teacher")
    parser.add_argument(
        "--allow-nontrain-source",
        action="store_true",
        help="Allow rows whose source split metadata is present and not train.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_surface_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("--surface-negative-counts entries must be label=count")
        label, raw_count = part.split("=", 1)
        label = label.strip()
        count = int(raw_count)
        if not label:
            raise ValueError("surface label must not be empty")
        if count < 0:
            raise ValueError("surface counts must be non-negative")
        if label in counts:
            raise ValueError(f"duplicate surface label: {label}")
        counts[label] = count
    if not counts:
        raise ValueError("at least one surface count is required")
    return counts


def parse_surface_weights(value: str, surfaces: list[str]) -> dict[str, float]:
    weights = {surface: 1.0 for surface in surfaces}
    if not value.strip():
        return weights
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("--surface-average-source-weights entries must be label=weight")
        label, raw_weight = part.split("=", 1)
        label = label.strip()
        if label not in weights:
            raise ValueError(f"unknown surface weight label: {label}")
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError("surface average source weights must be non-negative")
        weights[label] = weight
    if sum(weights.values()) <= 0:
        raise ValueError("at least one surface average source weight must be positive")
    return weights


def parse_surface_list(value: str, known_surfaces: list[str]) -> list[str]:
    if not value.strip():
        return []
    known = set(known_surfaces)
    surfaces = [part.strip() for part in value.split(",") if part.strip()]
    if len(set(surfaces)) != len(surfaces):
        raise ValueError("surface list entries must be unique")
    unknown = [surface for surface in surfaces if surface not in known]
    if unknown:
        raise ValueError(f"unknown anti-regression surface: {unknown[0]}")
    return surfaces


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.heldout_ratio < 1.0:
        raise ValueError("--heldout-ratio must be between 0 and 1")
    if args.negatives_per_query <= 0:
        raise ValueError("--negatives-per-query must be positive")
    surface_counts = parse_surface_counts(args.surface_negative_counts)
    parse_surface_weights(args.surface_average_source_weights, list(surface_counts))
    if sum(surface_counts.values()) > args.negatives_per_query:
        raise ValueError("surface negative counts cannot exceed --negatives-per-query")
    if args.min_surface_false_positive_available < 0:
        raise ValueError("--min-surface-false-positive-available must be non-negative")
    if args.min_distinct_surfaces_available < 0:
        raise ValueError("--min-distinct-surfaces-available must be non-negative")
    if args.teacher_separation_margin < 0:
        raise ValueError("--teacher-separation-margin must be non-negative")
    if args.teacher_hard_margin < 0 or args.middle_margin < 0:
        raise ValueError("teacher margin thresholds must be non-negative")
    if args.middle_margin < args.teacher_hard_margin:
        raise ValueError("--middle-margin must be >= --teacher-hard-margin")
    if args.score_scale <= 0:
        raise ValueError("--score-scale must be positive")
    if (
        args.max_target_best_margin is not None
        and args.min_target_best_margin > args.max_target_best_margin
    ):
        raise ValueError("--min-target-best-margin must be <= --max-target-best-margin")
    weights = [args.reranker_weight, args.existing_weight, args.surface_average_weight]
    if args.anti_regression_weight < 0:
        raise ValueError("--anti-regression-weight must be non-negative")
    if args.anti_regression_margin <= 0 or args.anti_regression_margin >= 1:
        raise ValueError("--anti-regression-margin must be in (0, 1)")
    if (
        args.anti_regression_surface_hard_margin is not None
        and args.anti_regression_surface_hard_margin < 0
    ):
        raise ValueError("--anti-regression-surface-hard-margin must be non-negative")
    anti_surfaces = parse_surface_list(args.anti_regression_surfaces, list(surface_counts))
    if args.anti_regression_weight > 0 and not anti_surfaces:
        raise ValueError("--anti-regression-surfaces is required when weight is positive")
    weights.append(args.anti_regression_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError("target weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("at least one target weight must be positive")


def float_list(values: Any) -> list[float]:
    if not values:
        return []
    return [float(value) for value in values]


def row_query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id") or row.get("query"))


def best_index(scores: list[float]) -> int:
    return max(range(len(scores)), key=lambda index: scores[index])


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5 for _value in values]
    return [(value - low) / (high - low) for value in values]


def weighted_minmax_target(
    sources: list[tuple[list[float], float]],
    *,
    score_scale: float,
) -> list[float]:
    length = len(sources[0][0])
    totals = [0.0 for _value in range(length)]
    total_weight = 0.0
    for values, weight in sources:
        if weight <= 0:
            continue
        if len(values) != length:
            raise ValueError("all target sources must have the same length")
        for index, value in enumerate(minmax(values)):
            totals[index] += weight * value
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("at least one source weight is required")
    return [score_scale * value / total_weight for value in totals]


def reranker_scores_for_row(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    pos = float_list(row.get("original_reranker_pos_scores"))
    neg = float_list(row.get("original_reranker_neg_scores"))
    if pos and neg:
        return pos, neg
    return float_list(row.get("pos_scores")), float_list(row.get("neg_scores"))


def surface_scores_for_row(
    row: dict[str, Any],
    surfaces: list[str],
) -> tuple[dict[str, list[float]], dict[str, list[float]]] | None:
    pos_by_surface = row.get("bge_m3_surface_pos_scores", {})
    neg_by_surface = row.get("bge_m3_surface_neg_scores", {})
    if not isinstance(pos_by_surface, dict) or not isinstance(neg_by_surface, dict):
        return None
    pos_scores = {}
    neg_scores = {}
    for surface in surfaces:
        pos = float_list(pos_by_surface.get(surface))
        neg = float_list(neg_by_surface.get(surface))
        if not pos or not neg:
            return None
        pos_scores[surface] = pos
        neg_scores[surface] = neg
    return pos_scores, neg_scores


def average_surface_scores(
    pos_by_surface: dict[str, list[float]],
    neg_by_surface: dict[str, list[float]],
    surfaces: list[str],
    surface_weights: dict[str, float],
) -> tuple[list[float], list[float]]:
    pos_length = len(pos_by_surface[surfaces[0]])
    neg_length = len(neg_by_surface[surfaces[0]])
    pos = [0.0 for _value in range(pos_length)]
    neg = [0.0 for _value in range(neg_length)]
    total_weight = 0.0
    for surface in surfaces:
        if len(pos_by_surface[surface]) != pos_length or len(neg_by_surface[surface]) != neg_length:
            raise ValueError("surface score lengths do not match")
        weight = surface_weights[surface]
        if weight <= 0:
            continue
        for index, value in enumerate(pos_by_surface[surface]):
            pos[index] += weight * value
        for index, value in enumerate(neg_by_surface[surface]):
            neg[index] += weight * value
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("at least one surface weight must be positive")
    divisor = float(total_weight)
    return [value / divisor for value in pos], [value / divisor for value in neg]


def anti_regression_scores(
    *,
    pos_by_surface: dict[str, list[float]],
    neg_by_surface: dict[str, list[float]],
    surfaces: list[str],
    margin: float,
    surface_hard_margin: float,
) -> tuple[list[float], list[float], dict[str, Any]]:
    if not surfaces:
        raise ValueError("anti-regression surfaces must not be empty")
    pos_count = len(pos_by_surface[surfaces[0]])
    neg_count = len(neg_by_surface[surfaces[0]])
    pos_scores = [1.0 for _index in range(pos_count)]
    neg_scores = []
    labels_by_negative = []
    for neg_index in range(neg_count):
        labels = []
        for surface in surfaces:
            pos_surface_score = max(pos_by_surface[surface])
            neg_surface_score = neg_by_surface[surface][neg_index]
            if pos_surface_score - neg_surface_score <= surface_hard_margin:
                labels.append(surface)
        labels_by_negative.append(labels)
        neg_scores.append(1.0 - margin if labels else 0.0)
    return pos_scores, neg_scores, {
        "surfaces": surfaces,
        "margin": margin,
        "surface_hard_margin": surface_hard_margin,
        "false_positive_surfaces_by_negative": labels_by_negative,
    }


def target_scores_for_row(
    row: dict[str, Any],
    *,
    surfaces: list[str],
    reranker_weight: float,
    existing_weight: float,
    surface_average_weight: float,
    surface_average_source_weights: str,
    anti_regression_surfaces: str = "",
    anti_regression_weight: float = 0.0,
    anti_regression_margin: float = 0.20,
    anti_regression_surface_hard_margin: float | None = None,
    score_scale: float = 1.0,
) -> tuple[list[float], list[float], dict[str, Any]] | None:
    existing_pos = float_list(row.get("pos_scores"))
    existing_neg = float_list(row.get("neg_scores"))
    reranker_pos, reranker_neg = reranker_scores_for_row(row)
    surface_scores = surface_scores_for_row(row, surfaces)
    if surface_scores is None:
        return None
    surface_pos_by_label, surface_neg_by_label = surface_scores
    surface_pos, surface_neg = average_surface_scores(
        surface_pos_by_label,
        surface_neg_by_label,
        surfaces,
        parse_surface_weights(surface_average_source_weights, surfaces),
    )
    anti_surfaces = parse_surface_list(anti_regression_surfaces, surfaces)
    anti_payload = None
    anti_pos = []
    anti_neg = []
    if anti_regression_weight > 0 and anti_surfaces:
        anti_pos, anti_neg, anti_payload = anti_regression_scores(
            pos_by_surface=surface_pos_by_label,
            neg_by_surface=surface_neg_by_label,
            surfaces=anti_surfaces,
            margin=anti_regression_margin,
            surface_hard_margin=(
                anti_regression_surface_hard_margin
                if anti_regression_surface_hard_margin is not None
                else 0.0
            ),
        )
    if not existing_pos or not existing_neg or not reranker_pos or not reranker_neg:
        return None
    pos_count = len(existing_pos)
    neg_count = len(existing_neg)
    if any(
        len(values) != expected
        for values, expected in [
            (reranker_pos, pos_count),
            (surface_pos, pos_count),
            (reranker_neg, neg_count),
            (surface_neg, neg_count),
            (anti_pos, pos_count) if anti_pos else (existing_pos, pos_count),
            (anti_neg, neg_count) if anti_neg else (existing_neg, neg_count),
        ]
    ):
        return None
    target_sources = [
        ([*reranker_pos, *reranker_neg], reranker_weight),
        ([*existing_pos, *existing_neg], existing_weight),
        ([*surface_pos, *surface_neg], surface_average_weight),
    ]
    if anti_regression_weight > 0 and anti_pos and anti_neg:
        target_sources.append(
            ([*anti_pos, *anti_neg], anti_regression_weight)
        )
    target_all = weighted_minmax_target(
        target_sources,
        score_scale=score_scale,
    )
    source_payload = {
        "existing_pos": existing_pos,
        "existing_neg": existing_neg,
        "reranker_pos": reranker_pos,
        "reranker_neg": reranker_neg,
        "surface_pos_by_label": surface_pos_by_label,
        "surface_neg_by_label": surface_neg_by_label,
        "surface_average_pos": surface_pos,
        "surface_average_neg": surface_neg,
        "surface_average_source_weights": parse_surface_weights(
            surface_average_source_weights,
            surfaces,
        ),
        "anti_regression": anti_payload,
    }
    return target_all[:pos_count], target_all[pos_count:], source_payload


def margin_bin(value: float) -> str:
    if value < 0:
        return "negative"
    if value < 0.02:
        return "0_to_0.02"
    if value < 0.05:
        return "0.02_to_0.05"
    if value < 0.15:
        return "0.05_to_0.15"
    return "0.15_plus"


def stable_key(seed: str, row: dict[str, Any], item: dict[str, Any]) -> float:
    query = row_query_id(row)
    doc_id = item.get("doc_id") or item["index"]
    return stable_fraction(f"{query}:{doc_id}:{item['target_score']}", seed)


def surface_false_positive_labels(
    *,
    pos_index: int,
    neg_index: int,
    score_sources: dict[str, Any],
    surfaces: list[str],
    surface_hard_margin: float,
    teacher_margin: float,
    teacher_separation_margin: float,
) -> list[str]:
    if teacher_margin < teacher_separation_margin:
        return []
    labels = []
    for surface in surfaces:
        pos_score = score_sources["surface_pos_by_label"][surface][pos_index]
        neg_score = score_sources["surface_neg_by_label"][surface][neg_index]
        if pos_score - neg_score <= surface_hard_margin:
            labels.append(surface)
    return labels


def negative_items(
    row: dict[str, Any],
    *,
    pos_index: int,
    target_pos_score: float,
    target_neg_scores: list[float],
    score_sources: dict[str, Any],
    surfaces: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    neg_texts = row.get("neg", [])
    neg_doc_ids = row.get("neg_doc_ids", [None] * len(target_neg_scores))
    neg_sources = row.get("neg_sources", ["unknown"] * len(target_neg_scores))
    items = []
    for index, target_score in enumerate(target_neg_scores):
        target_margin = target_pos_score - target_score
        false_positive_surfaces = surface_false_positive_labels(
            pos_index=pos_index,
            neg_index=index,
            score_sources=score_sources,
            surfaces=surfaces,
            surface_hard_margin=args.surface_hard_margin,
            teacher_margin=target_margin,
            teacher_separation_margin=args.teacher_separation_margin,
        )
        surface_margins = {
            surface: (
                score_sources["surface_pos_by_label"][surface][pos_index]
                - score_sources["surface_neg_by_label"][surface][index]
            )
            for surface in surfaces
        }
        if false_positive_surfaces:
            stratum = "surface_false_positive"
        elif target_margin <= args.teacher_hard_margin:
            stratum = "teacher_hard"
        elif target_margin <= args.middle_margin:
            stratum = "middle"
        else:
            stratum = "easy"
        items.append(
            {
                "index": index,
                "text": str(neg_texts[index]),
                "doc_id": neg_doc_ids[index] if index < len(neg_doc_ids) else None,
                "source": str(neg_sources[index]) if index < len(neg_sources) else "unknown",
                "target_score": target_score,
                "target_margin": target_margin,
                "target_margin_bin": margin_bin(target_margin),
                "stratum": stratum,
                "false_positive_surfaces": false_positive_surfaces,
                "surface_margins": surface_margins,
                "reranker_score": score_sources["reranker_neg"][index],
                "existing_score": score_sources["existing_neg"][index],
                "surface_average_score": score_sources["surface_average_neg"][index],
            }
        )
    return items


def choose_surface_items(
    *,
    row: dict[str, Any],
    items: list[dict[str, Any]],
    surface_counts: dict[str, int],
    seed: str,
    selected_indices: set[int],
) -> list[dict[str, Any]]:
    selected = []
    for surface, count in surface_counts.items():
        candidates = [
            item
            for item in items
            if item["index"] not in selected_indices
            and surface in item["false_positive_surfaces"]
        ]
        candidates.sort(
            key=lambda item: (
                item["surface_margins"][surface],
                -item["target_margin"],
                stable_key(f"{seed}:{surface}", row, item),
            )
        )
        chosen = candidates[:count]
        selected.extend(chosen)
        selected_indices.update(item["index"] for item in chosen)
    return selected


def choose_fallback_items(
    *,
    row: dict[str, Any],
    items: list[dict[str, Any]],
    needed: int,
    seed: str,
    selected_indices: set[int],
) -> list[dict[str, Any]]:
    candidates = [item for item in items if item["index"] not in selected_indices]
    candidates.sort(
        key=lambda item: (
            FALLBACK_ORDER.index(item["stratum"]) if item["stratum"] in FALLBACK_ORDER else 99,
            item["target_margin"],
            min(item["surface_margins"].values()) if item["surface_margins"] else 0.0,
            stable_key(f"{seed}:fallback", row, item),
        )
    )
    chosen = candidates[:needed]
    selected_indices.update(item["index"] for item in chosen)
    return chosen


def choose_negatives(
    row: dict[str, Any],
    *,
    items: list[dict[str, Any]],
    surface_counts: dict[str, int],
    negatives_per_query: int,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_indices: set[int] = set()
    selected = choose_surface_items(
        row=row,
        items=items,
        surface_counts=surface_counts,
        seed=seed,
        selected_indices=selected_indices,
    )
    if len(selected) < negatives_per_query:
        selected.extend(
            choose_fallback_items(
                row=row,
                items=items,
                needed=negatives_per_query - len(selected),
                seed=seed,
                selected_indices=selected_indices,
            )
        )
    selected = selected[:negatives_per_query]
    selected_surface_counts = Counter(
        surface for item in selected for surface in item["false_positive_surfaces"]
    )
    selected_primary_surfaces = Counter(
        primary_surface(item) for item in selected if primary_surface(item)
    )
    selected_strata = Counter(item["stratum"] for item in selected)
    requested_met = all(
        selected_surface_counts.get(surface, 0) >= count
        for surface, count in surface_counts.items()
    )
    return selected, {
        "selected_surface_counts": dict(selected_surface_counts),
        "selected_primary_surfaces": dict(selected_primary_surfaces),
        "selected_strata": dict(selected_strata),
        "requested_surface_mix_met": requested_met,
    }


def primary_surface(item: dict[str, Any]) -> str | None:
    labels = item.get("false_positive_surfaces") or []
    if not labels:
        return None
    return min(labels, key=lambda label: item["surface_margins"][label])


def row_candidate(
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None:
    surface_counts = parse_surface_counts(args.surface_negative_counts)
    surfaces = list(surface_counts)
    target = target_scores_for_row(
        row,
        surfaces=surfaces,
        reranker_weight=args.reranker_weight,
        existing_weight=args.existing_weight,
        surface_average_weight=args.surface_average_weight,
        surface_average_source_weights=args.surface_average_source_weights,
        anti_regression_surfaces=args.anti_regression_surfaces,
        anti_regression_weight=args.anti_regression_weight,
        anti_regression_margin=args.anti_regression_margin,
        anti_regression_surface_hard_margin=(
            args.anti_regression_surface_hard_margin
            if args.anti_regression_surface_hard_margin is not None
            else args.surface_hard_margin
        ),
        score_scale=args.score_scale,
    )
    if target is None:
        return None
    target_pos_scores, target_neg_scores, score_sources = target
    if len(target_neg_scores) < args.negatives_per_query:
        return None
    pos_index = best_index(target_pos_scores)
    target_pos_score = target_pos_scores[pos_index]
    target_best_margin = target_pos_score - max(target_neg_scores)
    if target_best_margin < args.min_target_best_margin:
        return None
    if args.max_target_best_margin is not None and target_best_margin > args.max_target_best_margin:
        return None
    items = negative_items(
        row,
        pos_index=pos_index,
        target_pos_score=target_pos_score,
        target_neg_scores=target_neg_scores,
        score_sources=score_sources,
        surfaces=surfaces,
        args=args,
    )
    false_positive_items = [
        item for item in items if item["false_positive_surfaces"]
    ]
    distinct_surfaces = {
        surface for item in false_positive_items for surface in item["false_positive_surfaces"]
    }
    if len(false_positive_items) < args.min_surface_false_positive_available:
        return None
    if len(distinct_surfaces) < args.min_distinct_surfaces_available:
        return None
    diagnostics = {
        "target_best_margin": target_best_margin,
        "target_pos_score": target_pos_score,
        "available_surface_false_positive_items": len(false_positive_items),
        "available_distinct_false_positive_surfaces": len(distinct_surfaces),
        "available_surface_counts": dict(
            Counter(surface for item in false_positive_items for surface in item["false_positive_surfaces"])
        ),
        "available_primary_surfaces": dict(
            Counter(primary_surface(item) for item in false_positive_items if primary_surface(item))
        ),
    }
    return {
        "row": row,
        "pos_index": pos_index,
        "target_pos_scores": target_pos_scores,
        "score_sources": score_sources,
        "surfaces": surfaces,
    }, items, diagnostics


def prepare_output_row(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = candidate["row"]
    pos_index = candidate["pos_index"]
    score_sources = candidate["score_sources"]
    surfaces = candidate["surfaces"]
    pos_doc_ids = row.get("pos_doc_ids", [None] * len(candidate["target_pos_scores"]))
    selected_indices = [item["index"] for item in selected]
    output = {
        "query_id": row.get("query_id"),
        "query": row["query"],
        "pos_doc_ids": [pos_doc_ids[pos_index] if pos_index < len(pos_doc_ids) else None],
        "pos": [row["pos"][pos_index]],
        "pos_scores": [diagnostics["target_pos_score"]],
        "neg_doc_ids": [item["doc_id"] for item in selected],
        "neg": [item["text"] for item in selected],
        "neg_scores": [item["target_score"] for item in selected],
        "neg_sources": [item["source"] for item in selected],
        "neg_strata": [item["stratum"] for item in selected],
        "neg_primary_surfaces": [primary_surface(item) for item in selected],
        "neg_false_positive_surfaces": [item["false_positive_surfaces"] for item in selected],
        "neg_target_margin_bins": [item["target_margin_bin"] for item in selected],
        "surface_target_margins": [item["target_margin"] for item in selected],
        "surface_average_margins": [
            score_sources["surface_average_pos"][pos_index] - item["surface_average_score"]
            for item in selected
        ],
        "surface_margins_by_label": {
            surface: [item["surface_margins"][surface] for item in selected]
            for surface in surfaces
        },
        "original_existing_pos_scores": score_sources["existing_pos"],
        "original_existing_neg_scores": [score_sources["existing_neg"][index] for index in selected_indices],
        "original_reranker_pos_scores": score_sources["reranker_pos"],
        "original_reranker_neg_scores": [score_sources["reranker_neg"][index] for index in selected_indices],
        "bge_m3_surface_pos_scores": {
            surface: [score_sources["surface_pos_by_label"][surface][pos_index]]
            for surface in surfaces
        },
        "bge_m3_surface_neg_scores": {
            surface: [
                score_sources["surface_neg_by_label"][surface][index]
                for index in selected_indices
            ]
            for surface in surfaces
        },
        "multisurface_teacher": {
            "source": "prepare_multisurface_teacher_rows.py",
            "surfaces": surfaces,
            "surface_negative_counts": parse_surface_counts(args.surface_negative_counts),
            "surface_hard_margin": args.surface_hard_margin,
            "teacher_separation_margin": args.teacher_separation_margin,
            "target_weights": {
                "reranker": args.reranker_weight,
                "existing": args.existing_weight,
                "surface_average": args.surface_average_weight,
                "anti_regression": args.anti_regression_weight,
            },
            "surface_average_source_weights": score_sources["surface_average_source_weights"],
            "anti_regression": score_sources.get("anti_regression"),
            "score_scale": args.score_scale,
            "source_teacher": row.get("score_blend_teacher") or row.get("surface_teacher") or {},
        },
        "source": row.get("source", {}),
    }
    return output


def read_candidates(args: argparse.Namespace) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]], dict[str, Any]]:
    candidates = []
    total_rows = 0
    nontrain_rows = 0
    missing_or_short_rows = 0
    filtered_rows = 0
    with Path(args.input_jsonl).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            split = source_split(row)
            if split and split != "train":
                nontrain_rows += 1
                if not args.allow_nontrain_source:
                    raise ValueError(
                        f"{args.input_jsonl}:{line_number} has source.split={split!r}; "
                        "student distillation should not train from dev/test rows"
                    )
            candidate = row_candidate(row, args)
            if candidate is None:
                has_scores = bool(row.get("pos_scores")) and bool(row.get("neg_scores"))
                if has_scores:
                    filtered_rows += 1
                else:
                    missing_or_short_rows += 1
                continue
            candidates.append(candidate)
    return candidates, {
        "input_jsonl": args.input_jsonl,
        "total_rows": total_rows,
        "candidate_rows": len(candidates),
        "nontrain_source_rows_seen": nontrain_rows,
        "missing_or_short_rows": missing_or_short_rows,
        "filtered_rows": filtered_rows,
    }


def split_candidates(
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]],
    *,
    heldout_ratio: float,
    seed: str,
) -> tuple[
    list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]],
    list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]],
]:
    train = []
    heldout = []
    for candidate in candidates:
        query_id = row_query_id(candidate[0]["row"])
        if stable_fraction(query_id, seed) < heldout_ratio:
            heldout.append(candidate)
        else:
            train.append(candidate)
    if not train:
        raise ValueError("No train rows selected")
    if not heldout:
        raise ValueError("No held-out rows selected")
    return train, heldout


def candidate_sort_key(
    candidate: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
    seed: str,
) -> tuple[int, float, float]:
    diagnostics = candidate[2]
    query_id = row_query_id(candidate[0]["row"])
    return (
        -diagnostics["available_distinct_false_positive_surfaces"],
        diagnostics["target_best_margin"],
        stable_fraction(query_id, seed),
    )


def candidate_bin(candidate: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]) -> str:
    return margin_bin(candidate[2]["target_best_margin"])


def select_balanced_candidates(
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]],
    max_rows: int,
    seed: str,
) -> list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]]:
    if max_rows <= 0 or len(candidates) <= max_rows:
        return sorted(candidates, key=lambda candidate: candidate_sort_key(candidate, seed))
    buckets: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]]] = {
        label: [] for label in MARGIN_BIN_ORDER
    }
    for candidate in candidates:
        buckets[candidate_bin(candidate)].append(candidate)
    for label, bucket in buckets.items():
        bucket.sort(key=lambda candidate: candidate_sort_key(candidate, f"{seed}:{label}"))
    selected = []
    while len(selected) < max_rows and any(buckets.values()):
        for label in MARGIN_BIN_ORDER:
            bucket = buckets[label]
            if bucket:
                selected.append(bucket.pop(0))
                if len(selected) >= max_rows:
                    break
    return selected


def prepare_many(
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]],
    args: argparse.Namespace,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    row_diagnostics = []
    selection_diagnostics = []
    surface_counts = parse_surface_counts(args.surface_negative_counts)
    for candidate, items, diagnostics in candidates:
        selected, selected_diagnostics = choose_negatives(
            candidate["row"],
            items=items,
            surface_counts=surface_counts,
            negatives_per_query=args.negatives_per_query,
            seed=seed,
        )
        if len(selected) < args.negatives_per_query:
            continue
        rows.append(prepare_output_row(candidate, selected, diagnostics, args))
        row_diagnostics.append(diagnostics)
        selection_diagnostics.append(selected_diagnostics)
    return rows, row_diagnostics, selection_diagnostics


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_split(
    rows: list[dict[str, Any]],
    row_diagnostics: list[dict[str, Any]],
    selection_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_surface_counts = Counter(
        surface
        for row in rows
        for surface_list in row.get("neg_false_positive_surfaces", [])
        for surface in surface_list
    )
    primary_surface_counts = Counter(
        surface
        for row in rows
        for surface in row.get("neg_primary_surfaces", [])
        if surface
    )
    strata_counts = Counter(stratum for row in rows for stratum in row.get("neg_strata", []))
    target_margins = [margin for row in rows for margin in row.get("surface_target_margins", [])]
    surface_avg_margins = [margin for row in rows for margin in row.get("surface_average_margins", [])]
    requested_mix_met = sum(
        1 for diagnostic in selection_diagnostics if diagnostic.get("requested_surface_mix_met")
    )
    return {
        "rows": len(rows),
        "negative_pairs": sum(len(row.get("neg", [])) for row in rows),
        "selected_surface_counts": dict(selected_surface_counts),
        "selected_primary_surface_counts": dict(primary_surface_counts),
        "selected_strata_counts": dict(strata_counts),
        "rows_meeting_requested_surface_mix": requested_mix_met,
        "fraction_rows_meeting_requested_surface_mix": (
            requested_mix_met / len(selection_diagnostics) if selection_diagnostics else None
        ),
        "target_best_margin": numeric_summary(
            [diagnostic["target_best_margin"] for diagnostic in row_diagnostics]
        ),
        "available_surface_false_positive_items": numeric_summary(
            [diagnostic["available_surface_false_positive_items"] for diagnostic in row_diagnostics]
        ),
        "available_distinct_false_positive_surfaces": numeric_summary(
            [diagnostic["available_distinct_false_positive_surfaces"] for diagnostic in row_diagnostics]
        ),
        "target_margin": numeric_summary(target_margins),
        "surface_average_margin": numeric_summary(surface_avg_margins),
        "distribution": analyze_rows(rows, temperatures=[1.0, 0.5, 0.2, 0.1]),
    }


def prepare_splits(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validate_args(args)
    surface_counts = parse_surface_counts(args.surface_negative_counts)
    candidates, input_stats = read_candidates(args)
    train_candidates, heldout_candidates = split_candidates(
        candidates,
        heldout_ratio=args.heldout_ratio,
        seed=args.seed,
    )
    train_candidates = select_balanced_candidates(
        train_candidates,
        args.max_train_rows,
        f"{args.seed}:train",
    )
    heldout_candidates = select_balanced_candidates(
        heldout_candidates,
        args.max_eval_rows,
        f"{args.seed}:heldout",
    )
    train_rows, train_row_diags, train_select_diags = prepare_many(
        train_candidates,
        args,
        f"{args.seed}:train-neg",
    )
    heldout_rows, heldout_row_diags, heldout_select_diags = prepare_many(
        heldout_candidates,
        args,
        f"{args.seed}:heldout-neg",
    )
    train_queries = {row_query_id(row) for row in train_rows}
    heldout_queries = {row_query_id(row) for row in heldout_rows}
    overlap = train_queries.intersection(heldout_queries)
    if overlap:
        raise ValueError(f"Train/eval query overlap detected: {sorted(overlap)[:5]}")
    if not train_rows:
        raise ValueError("No train rows prepared")
    if not heldout_rows:
        raise ValueError("No held-out rows prepared")

    summary = {
        **input_stats,
        "splitter": {
            "seed": args.seed,
            "heldout_ratio": args.heldout_ratio,
            "max_train_rows": args.max_train_rows,
            "max_eval_rows": args.max_eval_rows,
            "query_overlap": 0,
            "row_selection": "query-disjoint split plus target-best-margin-bin round-robin",
        },
        "surface_target": {
            "method": "row_minmax_reranker_existing_surface_average_target",
            "surface_negative_counts": surface_counts,
            "surfaces": list(surface_counts),
            "reranker_weight": args.reranker_weight,
            "existing_weight": args.existing_weight,
            "surface_average_weight": args.surface_average_weight,
            "surface_average_source_weights": parse_surface_weights(
                args.surface_average_source_weights,
                list(surface_counts),
            ),
            "anti_regression_surfaces": parse_surface_list(
                args.anti_regression_surfaces,
                list(surface_counts),
            ),
            "anti_regression_weight": args.anti_regression_weight,
            "anti_regression_margin": args.anti_regression_margin,
            "anti_regression_surface_hard_margin": (
                args.anti_regression_surface_hard_margin
                if args.anti_regression_surface_hard_margin is not None
                else args.surface_hard_margin
            ),
            "score_scale": args.score_scale,
            "rationale": (
                "Use BGE-M3 component-augmented v75 rows to select negatives where "
                "strong fusion surfaces fail while the teacher target still separates them."
            ),
        },
        "negative_selection": {
            "negatives_per_query": args.negatives_per_query,
            "surface_hard_margin": args.surface_hard_margin,
            "teacher_separation_margin": args.teacher_separation_margin,
            "teacher_hard_margin": args.teacher_hard_margin,
            "middle_margin": args.middle_margin,
            "min_surface_false_positive_available": args.min_surface_false_positive_available,
            "min_distinct_surfaces_available": args.min_distinct_surfaces_available,
            "min_target_best_margin": args.min_target_best_margin,
            "max_target_best_margin": args.max_target_best_margin,
        },
        "train": summarize_split(train_rows, train_row_diags, train_select_diags),
        "heldout": summarize_split(heldout_rows, heldout_row_diags, heldout_select_diags),
        "raw_training_data_committed": False,
        "model_checkpoints_committed": False,
        "generated_embeddings_committed": False,
        "credentials_committed": False,
    }
    return train_rows, heldout_rows, summary


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
    train_rows, eval_rows, summary = prepare_splits(args)
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
