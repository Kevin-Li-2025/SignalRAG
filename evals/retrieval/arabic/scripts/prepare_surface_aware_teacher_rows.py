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


SURFACE_BIN_ORDER = [
    "negative",
    "0_to_0.05",
    "0.05_to_0.15",
    "0.15_to_0.45",
    "0.45_plus",
]
STRATUM_ORDER = [
    "agreement_hard",
    "hybrid_false_positive",
    "target_hard",
    "middle",
    "easy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build query-disjoint BGE-M3 student teacher rows whose labels and "
            "negative mix explicitly target the strong dense+sparse+ColBERT "
            "hybrid surfaces, not only reranker logits."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-jsonl", default="teacher_train_surface_aware.jsonl")
    parser.add_argument("--eval-jsonl", default="teacher_eval_surface_aware.jsonl")
    parser.add_argument("--max-train-rows", type=int, default=512)
    parser.add_argument("--max-eval-rows", type=int, default=96)
    parser.add_argument("--heldout-ratio", type=float, default=0.15)
    parser.add_argument("--negatives-per-query", type=int, default=8)
    parser.add_argument("--agreement-hard-negatives", type=int, default=1)
    parser.add_argument("--hybrid-false-positive-negatives", type=int, default=7)
    parser.add_argument("--target-hard-negatives", type=int, default=0)
    parser.add_argument("--middle-negatives", type=int, default=0)
    parser.add_argument("--easy-negatives", type=int, default=0)
    parser.add_argument("--min-hybrid-false-positive-available", type=int, default=1)
    parser.add_argument("--min-target-hard-available", type=int, default=1)
    parser.add_argument("--min-target-best-margin", type=float, default=0.0)
    parser.add_argument("--max-target-best-margin", type=float, default=None)
    parser.add_argument("--target-hard-margin", type=float, default=0.15)
    parser.add_argument("--hybrid-hard-margin", type=float, default=0.15)
    parser.add_argument("--middle-margin", type=float, default=0.45)
    parser.add_argument("--teacher-separation-margin", type=float, default=0.05)
    parser.add_argument("--reranker-weight", type=float, default=0.55)
    parser.add_argument("--hybrid-weight", type=float, default=0.45)
    parser.add_argument("--existing-weight", type=float, default=0.0)
    parser.add_argument("--score-scale", type=float, default=1.0)
    parser.add_argument(
        "--missing-hybrid",
        choices=["skip", "target"],
        default="skip",
        help="Skip rows with incomplete hybrid scores or fall back to target scores.",
    )
    parser.add_argument("--seed", default="20260605-v72-surface-aware-teacher")
    parser.add_argument(
        "--allow-nontrain-source",
        action="store_true",
        help="Allow rows whose source split metadata is present and not train.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.heldout_ratio < 1.0:
        raise ValueError("--heldout-ratio must be between 0 and 1")
    if args.negatives_per_query <= 0:
        raise ValueError("--negatives-per-query must be positive")
    requested = (
        args.agreement_hard_negatives
        + args.hybrid_false_positive_negatives
        + args.target_hard_negatives
        + args.middle_negatives
        + args.easy_negatives
    )
    if requested != args.negatives_per_query:
        raise ValueError("surface negative counts must sum to --negatives-per-query")
    if args.score_scale <= 0:
        raise ValueError("--score-scale must be positive")
    if args.target_hard_margin < 0 or args.hybrid_hard_margin < 0:
        raise ValueError("hard margins must be non-negative")
    if args.middle_margin < args.target_hard_margin:
        raise ValueError("--middle-margin must be >= --target-hard-margin")
    if args.teacher_separation_margin < 0:
        raise ValueError("--teacher-separation-margin must be non-negative")
    if args.min_hybrid_false_positive_available < 0 or args.min_target_hard_available < 0:
        raise ValueError("minimum available counts must be non-negative")
    if (
        args.max_target_best_margin is not None
        and args.min_target_best_margin > args.max_target_best_margin
    ):
        raise ValueError("--min-target-best-margin must be <= --max-target-best-margin")
    if args.reranker_weight < 0 or args.hybrid_weight < 0 or args.existing_weight < 0:
        raise ValueError("target-source weights must be non-negative")
    if args.reranker_weight + args.hybrid_weight + args.existing_weight <= 0:
        raise ValueError("at least one target-source weight must be positive")


def float_list(values: Any) -> list[float]:
    if not values:
        return []
    return [float(value) for value in values]


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5 for _value in values]
    return [(value - low) / (high - low) for value in values]


def row_query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id") or row.get("query"))


def best_index(scores: list[float]) -> int:
    return max(range(len(scores)), key=lambda index: scores[index])


def reranker_scores_for_row(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    pos = float_list(row.get("original_reranker_pos_scores"))
    neg = float_list(row.get("original_reranker_neg_scores"))
    if not pos and row.get("original_reranker_pos_score") is not None:
        pos = [float(row["original_reranker_pos_score"])]
    if not neg:
        neg = float_list(row.get("original_reranker_neg_score"))
    if pos and neg:
        return pos, neg
    return float_list(row.get("pos_scores")), float_list(row.get("neg_scores"))


def hybrid_scores_for_row(
    row: dict[str, Any],
    *,
    fallback_pos: list[float],
    fallback_neg: list[float],
    missing_hybrid: str,
) -> tuple[list[float], list[float], bool]:
    pos = float_list(row.get("bge_m3_hybrid_pos_scores"))
    neg = float_list(row.get("bge_m3_hybrid_neg_scores"))
    expected = len(fallback_pos), len(fallback_neg)
    complete = len(pos) == expected[0] and len(neg) == expected[1]
    if complete:
        return pos, neg, True
    if missing_hybrid == "target":
        return list(fallback_pos), list(fallback_neg), False
    return [], [], False


def source_weighted_scores(
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
            raise ValueError("all score sources must have the same length")
        normalized = minmax(values)
        for index, value in enumerate(normalized):
            totals[index] += weight * value
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("at least one positive source weight is required")
    return [score_scale * value / total_weight for value in totals]


def surface_target_scores(
    row: dict[str, Any],
    *,
    reranker_weight: float,
    hybrid_weight: float,
    existing_weight: float,
    score_scale: float,
    missing_hybrid: str,
) -> tuple[list[float], list[float], dict[str, Any]] | None:
    existing_pos = float_list(row.get("pos_scores"))
    existing_neg = float_list(row.get("neg_scores"))
    reranker_pos, reranker_neg = reranker_scores_for_row(row)
    if not existing_pos or not existing_neg or not reranker_pos or not reranker_neg:
        return None
    if len(existing_pos) != len(reranker_pos) or len(existing_neg) != len(reranker_neg):
        return None

    hybrid_pos, hybrid_neg, hybrid_complete = hybrid_scores_for_row(
        row,
        fallback_pos=existing_pos,
        fallback_neg=existing_neg,
        missing_hybrid=missing_hybrid,
    )
    if not hybrid_pos or not hybrid_neg:
        return None

    target_all = source_weighted_scores(
        [
            ([*reranker_pos, *reranker_neg], reranker_weight),
            ([*hybrid_pos, *hybrid_neg], hybrid_weight),
            ([*existing_pos, *existing_neg], existing_weight),
        ],
        score_scale=score_scale,
    )
    source_payload = {
        "existing_pos": existing_pos,
        "existing_neg": existing_neg,
        "reranker_pos": reranker_pos,
        "reranker_neg": reranker_neg,
        "hybrid_pos": hybrid_pos,
        "hybrid_neg": hybrid_neg,
        "hybrid_complete": hybrid_complete,
    }
    return target_all[: len(existing_pos)], target_all[len(existing_pos) :], source_payload


def surface_margin_bin(value: float) -> str:
    if value < 0:
        return "negative"
    if value < 0.05:
        return "0_to_0.05"
    if value < 0.15:
        return "0.05_to_0.15"
    if value < 0.45:
        return "0.15_to_0.45"
    return "0.45_plus"


def classify_negative(
    *,
    target_margin: float,
    hybrid_margin: float,
    target_hard_margin: float,
    hybrid_hard_margin: float,
    middle_margin: float,
    teacher_separation_margin: float,
) -> str:
    target_hard = target_margin <= target_hard_margin
    hybrid_hard = hybrid_margin <= hybrid_hard_margin
    if target_hard and hybrid_hard:
        return "agreement_hard"
    if hybrid_hard and target_margin >= teacher_separation_margin:
        return "hybrid_false_positive"
    if target_hard:
        return "target_hard"
    if target_margin <= middle_margin:
        return "middle"
    return "easy"


def stable_key(seed: str, row: dict[str, Any], item: dict[str, Any]) -> float:
    query = row_query_id(row)
    doc_id = item.get("doc_id") or item["index"]
    return stable_fraction(f"{query}:{doc_id}:{item['target_score']}", seed)


def negative_items(
    row: dict[str, Any],
    *,
    pos_index: int,
    target_pos_score: float,
    target_neg_scores: list[float],
    score_sources: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    neg_texts = row.get("neg", [])
    neg_doc_ids = row.get("neg_doc_ids", [None] * len(target_neg_scores))
    neg_sources = row.get("neg_sources", ["unknown"] * len(target_neg_scores))
    hybrid_pos_score = score_sources["hybrid_pos"][pos_index]
    reranker_pos_score = score_sources["reranker_pos"][pos_index]
    items = []
    for index, target_score in enumerate(target_neg_scores):
        hybrid_score = score_sources["hybrid_neg"][index]
        reranker_score = score_sources["reranker_neg"][index]
        target_margin = target_pos_score - target_score
        hybrid_margin = hybrid_pos_score - hybrid_score
        reranker_margin = reranker_pos_score - reranker_score
        stratum = classify_negative(
            target_margin=target_margin,
            hybrid_margin=hybrid_margin,
            target_hard_margin=args.target_hard_margin,
            hybrid_hard_margin=args.hybrid_hard_margin,
            middle_margin=args.middle_margin,
            teacher_separation_margin=args.teacher_separation_margin,
        )
        items.append(
            {
                "index": index,
                "text": str(neg_texts[index]),
                "doc_id": neg_doc_ids[index] if index < len(neg_doc_ids) else None,
                "source": str(neg_sources[index]) if index < len(neg_sources) else "unknown",
                "target_score": target_score,
                "target_margin": target_margin,
                "target_margin_bin": surface_margin_bin(target_margin),
                "hybrid_score": hybrid_score,
                "hybrid_margin": hybrid_margin,
                "reranker_score": reranker_score,
                "reranker_margin": reranker_margin,
                "existing_score": score_sources["existing_neg"][index],
                "stratum": stratum,
            }
        )
    return items


def item_sort_key(stratum: str, seed: str, row: dict[str, Any], item: dict[str, Any]) -> tuple[float, float, float]:
    stable = stable_key(seed, row, item)
    if stratum == "agreement_hard":
        return (item["target_margin"] + item["hybrid_margin"], item["target_margin"], stable)
    if stratum == "hybrid_false_positive":
        return (item["hybrid_margin"], -item["target_margin"], stable)
    if stratum == "target_hard":
        return (item["target_margin"], item["hybrid_margin"], stable)
    if stratum == "middle":
        return (abs(item["target_margin"] - 0.30), item["hybrid_margin"], stable)
    return (stable, item["target_margin"], item["hybrid_margin"])


def choose_from_stratum(
    *,
    row: dict[str, Any],
    items: list[dict[str, Any]],
    stratum: str,
    count: int,
    seed: str,
    selected_indices: set[int],
) -> list[dict[str, Any]]:
    candidates = [
        item for item in items if item["stratum"] == stratum and item["index"] not in selected_indices
    ]
    candidates.sort(key=lambda item: item_sort_key(stratum, seed, row, item))
    chosen = candidates[:count]
    selected_indices.update(item["index"] for item in chosen)
    return chosen


def choose_negatives(
    row: dict[str, Any],
    *,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_indices: set[int] = set()
    selected: list[dict[str, Any]] = []
    requested = {
        "agreement_hard": args.agreement_hard_negatives,
        "hybrid_false_positive": args.hybrid_false_positive_negatives,
        "target_hard": args.target_hard_negatives,
        "middle": args.middle_negatives,
        "easy": args.easy_negatives,
    }
    available_by_stratum = Counter(item["stratum"] for item in items)
    for stratum in STRATUM_ORDER:
        selected.extend(
            choose_from_stratum(
                row=row,
                items=items,
                stratum=stratum,
                count=requested[stratum],
                seed=seed,
                selected_indices=selected_indices,
            )
        )

    if len(selected) < args.negatives_per_query:
        remaining = [item for item in items if item["index"] not in selected_indices]
        remaining.sort(
            key=lambda item: (
                STRATUM_ORDER.index(item["stratum"]) if item["stratum"] in STRATUM_ORDER else 99,
                item_sort_key(item["stratum"], seed, row, item),
            )
        )
        selected.extend(remaining[: args.negatives_per_query - len(selected)])
    selected = selected[: args.negatives_per_query]
    selected_by_stratum = Counter(item["stratum"] for item in selected)
    selected_by_bin = Counter(item["target_margin_bin"] for item in selected)
    diagnostics = {
        "available_by_stratum": dict(available_by_stratum),
        "selected_by_stratum": dict(selected_by_stratum),
        "selected_by_target_margin_bin": dict(selected_by_bin),
        "requested_mix_met": all(
            selected_by_stratum.get(stratum, 0) >= requested_count
            for stratum, requested_count in requested.items()
        ),
    }
    return selected, diagnostics


def row_candidate(
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None:
    target = surface_target_scores(
        row,
        reranker_weight=args.reranker_weight,
        hybrid_weight=args.hybrid_weight,
        existing_weight=args.existing_weight,
        score_scale=args.score_scale,
        missing_hybrid=args.missing_hybrid,
    )
    if target is None:
        return None
    target_pos_scores, target_neg_scores, score_sources = target
    if len(target_neg_scores) < args.negatives_per_query:
        return None
    pos_index = best_index(target_pos_scores)
    target_pos_score = target_pos_scores[pos_index]
    best_margin = target_pos_score - max(target_neg_scores)
    if best_margin < args.min_target_best_margin:
        return None
    if args.max_target_best_margin is not None and best_margin > args.max_target_best_margin:
        return None
    items = negative_items(
        row,
        pos_index=pos_index,
        target_pos_score=target_pos_score,
        target_neg_scores=target_neg_scores,
        score_sources=score_sources,
        args=args,
    )
    counts = Counter(item["stratum"] for item in items)
    if counts.get("hybrid_false_positive", 0) < args.min_hybrid_false_positive_available:
        return None
    if counts.get("target_hard", 0) + counts.get("agreement_hard", 0) < args.min_target_hard_available:
        return None
    diagnostics = {
        "target_best_margin": best_margin,
        "available_by_stratum": dict(counts),
        "target_pos_score": target_pos_score,
        "hybrid_pos_score": score_sources["hybrid_pos"][pos_index],
        "reranker_pos_score": score_sources["reranker_pos"][pos_index],
        "hybrid_complete": score_sources["hybrid_complete"],
    }
    return {"row": row, "pos_index": pos_index, "target_pos_scores": target_pos_scores}, items, diagnostics


def prepare_output_row(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = candidate["row"]
    pos_index = candidate["pos_index"]
    pos_doc_ids = row.get("pos_doc_ids", [None] * len(candidate["target_pos_scores"]))
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
        "neg_target_margin_bins": [item["target_margin_bin"] for item in selected],
        "surface_target_margins": [item["target_margin"] for item in selected],
        "surface_hybrid_margins": [item["hybrid_margin"] for item in selected],
        "surface_reranker_margins": [item["reranker_margin"] for item in selected],
        "original_existing_pos_scores": float_list(row.get("pos_scores")),
        "original_existing_neg_scores": float_list(row.get("neg_scores")),
        "original_reranker_pos_scores": reranker_scores_for_row(row)[0],
        "original_reranker_neg_scores": [item["reranker_score"] for item in selected],
        "bge_m3_hybrid_pos_scores": [diagnostics["hybrid_pos_score"]],
        "bge_m3_hybrid_neg_scores": [item["hybrid_score"] for item in selected],
        "surface_teacher": {
            "source": "prepare_surface_aware_teacher_rows.py",
            "reranker_weight": args.reranker_weight,
            "hybrid_weight": args.hybrid_weight,
            "existing_weight": args.existing_weight,
            "score_scale": args.score_scale,
            "negative_mix": {
                "agreement_hard": args.agreement_hard_negatives,
                "hybrid_false_positive": args.hybrid_false_positive_negatives,
                "target_hard": args.target_hard_negatives,
                "middle": args.middle_negatives,
                "easy": args.easy_negatives,
            },
            "source_teacher": row.get("score_blend_teacher") or row.get("reranker_teacher", {}),
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
    stats = {
        "input_jsonl": args.input_jsonl,
        "total_rows": total_rows,
        "candidate_rows": len(candidates),
        "nontrain_source_rows_seen": nontrain_rows,
        "missing_or_short_rows": missing_or_short_rows,
        "filtered_rows": filtered_rows,
    }
    return candidates, stats


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
        row = candidate[0]["row"]
        query_id = row_query_id(row)
        if stable_fraction(query_id, seed) < heldout_ratio:
            heldout.append(candidate)
        else:
            train.append(candidate)
    if not train:
        raise ValueError("No train rows selected")
    if not heldout:
        raise ValueError("No held-out rows selected")
    return train, heldout


def row_bin(candidate: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]) -> str:
    return surface_margin_bin(candidate[2]["target_best_margin"])


def select_balanced_candidates(
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]],
    max_rows: int,
    seed: str,
) -> list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]]:
    if max_rows <= 0 or len(candidates) <= max_rows:
        return sorted(
            candidates,
            key=lambda candidate: stable_fraction(row_query_id(candidate[0]["row"]), seed),
        )
    buckets: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]]] = {
        label: [] for label in SURFACE_BIN_ORDER
    }
    for candidate in candidates:
        buckets[row_bin(candidate)].append(candidate)
    for label, bucket in buckets.items():
        bucket.sort(
            key=lambda candidate: stable_fraction(
                f"{label}:{row_query_id(candidate[0]['row'])}",
                seed,
            )
        )
    selected = []
    while len(selected) < max_rows and any(buckets.values()):
        for label in SURFACE_BIN_ORDER:
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
    for candidate, items, diagnostics in candidates:
        selected, selected_diagnostics = choose_negatives(
            candidate["row"],
            items=items,
            args=args,
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
    stratum_counts = Counter(stratum for row in rows for stratum in row.get("neg_strata", []))
    bin_counts = Counter(label for row in rows for label in row.get("neg_target_margin_bins", []))
    target_margins = [margin for row in rows for margin in row.get("surface_target_margins", [])]
    hybrid_margins = [margin for row in rows for margin in row.get("surface_hybrid_margins", [])]
    return {
        "rows": len(rows),
        "negative_pairs": sum(len(row.get("neg", [])) for row in rows),
        "negative_strata_counts": dict(stratum_counts),
        "negative_target_margin_bin_counts": dict(bin_counts),
        "rows_meeting_requested_mix": sum(
            1 for diagnostic in selection_diagnostics if diagnostic.get("requested_mix_met")
        ),
        "fraction_rows_meeting_requested_mix": (
            sum(1 for diagnostic in selection_diagnostics if diagnostic.get("requested_mix_met"))
            / len(selection_diagnostics)
            if selection_diagnostics
            else None
        ),
        "target_best_margin": numeric_summary(
            [diagnostic["target_best_margin"] for diagnostic in row_diagnostics]
        ),
        "target_margin": numeric_summary(target_margins),
        "hybrid_margin": numeric_summary(hybrid_margins),
        "distribution": analyze_rows(rows, temperatures=[1.0, 0.5, 0.2, 0.1]),
    }


def prepare_splits(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validate_args(args)
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
            "method": "row_minmax_reranker_hybrid_existing_weighted_target",
            "reranker_weight": args.reranker_weight,
            "hybrid_weight": args.hybrid_weight,
            "existing_weight": args.existing_weight,
            "score_scale": args.score_scale,
            "missing_hybrid": args.missing_hybrid,
            "rationale": (
                "Aim teacher-score KD at BGE-M3 hybrid/model-card surfaces by "
                "preserving hybrid-hard negatives instead of only reranker-hard negatives."
            ),
        },
        "negative_mix": {
            "negatives_per_query": args.negatives_per_query,
            "agreement_hard": args.agreement_hard_negatives,
            "hybrid_false_positive": args.hybrid_false_positive_negatives,
            "target_hard": args.target_hard_negatives,
            "middle": args.middle_negatives,
            "easy": args.easy_negatives,
            "minimum_available": {
                "hybrid_false_positive": args.min_hybrid_false_positive_available,
                "target_hard_or_agreement_hard": args.min_target_hard_available,
            },
            "target_hard_margin": args.target_hard_margin,
            "hybrid_hard_margin": args.hybrid_hard_margin,
            "middle_margin": args.middle_margin,
            "teacher_separation_margin": args.teacher_separation_margin,
            "min_target_best_margin": args.min_target_best_margin,
            "max_target_best_margin": args.max_target_best_margin,
        },
        "train": summarize_split(train_rows, train_row_diags, train_select_diags),
        "heldout": summarize_split(heldout_rows, heldout_row_diags, heldout_select_diags),
        "raw_training_data_committed": False,
        "model_checkpoints_committed": False,
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
