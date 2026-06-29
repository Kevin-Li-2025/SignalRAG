#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_teacher_score_distribution import (
    analyze_rows,
    best_margin,
    margin_bin,
)
from prepare_student_distill_splits import source_split, stable_fraction
from rerank_teacher_jsonl import write_json


ROW_BIN_ORDER = ["negative", "0_to_0.5", "0.5_to_1", "1_to_2", "2_to_4", "4_plus"]
NEGATIVE_STRATA = {
    "hard": {"negative", "0_to_0.5", "0.5_to_1"},
    "middle": {"1_to_2", "2_to_4"},
    "easy": {"4_plus"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build query-disjoint, score-stratified teacher rows for BGE-M3 "
            "student distillation from reranker-scored JSONL."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-jsonl", default="teacher_train_stratified.jsonl")
    parser.add_argument("--eval-jsonl", default="teacher_eval_stratified.jsonl")
    parser.add_argument("--max-train-rows", type=int, default=2048)
    parser.add_argument("--max-eval-rows", type=int, default=400)
    parser.add_argument("--heldout-ratio", type=float, default=0.15)
    parser.add_argument("--negatives-per-query", type=int, default=8)
    parser.add_argument("--hard-negatives", type=int, default=3)
    parser.add_argument("--middle-negatives", type=int, default=3)
    parser.add_argument("--easy-negatives", type=int, default=2)
    parser.add_argument(
        "--min-hard-available",
        type=int,
        default=0,
        help="Require at least this many hard negatives before row selection.",
    )
    parser.add_argument(
        "--min-middle-available",
        type=int,
        default=0,
        help="Require at least this many middle negatives before row selection.",
    )
    parser.add_argument(
        "--min-easy-available",
        type=int,
        default=0,
        help="Require at least this many easy negatives before row selection.",
    )
    parser.add_argument(
        "--score-scale",
        type=float,
        default=0.25,
        help="Multiply teacher scores before writing pos_scores/neg_scores.",
    )
    parser.add_argument(
        "--min-best-margin",
        type=float,
        default=None,
        help=(
            "Require best positive teacher score minus max negative teacher score "
            "to be at least this value."
        ),
    )
    parser.add_argument(
        "--max-best-margin",
        type=float,
        default=None,
        help=(
            "Require best positive teacher score minus max negative teacher score "
            "to be at most this value. Use with --min-best-margin to build "
            "curriculum bands instead of one broad split."
        ),
    )
    parser.add_argument("--seed", default="20260605-v66-stratified-teacher")
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
    requested = args.hard_negatives + args.middle_negatives + args.easy_negatives
    if requested != args.negatives_per_query:
        raise ValueError("hard + middle + easy negatives must equal --negatives-per-query")
    if args.score_scale <= 0:
        raise ValueError("--score-scale must be positive")
    for label in ["min_hard_available", "min_middle_available", "min_easy_available"]:
        if getattr(args, label) < 0:
            raise ValueError(f"--{label.replace('_', '-')} must be non-negative")
    if (
        args.min_best_margin is not None
        and args.max_best_margin is not None
        and args.min_best_margin > args.max_best_margin
    ):
        raise ValueError("--min-best-margin must be <= --max-best-margin")


def float_list(values: Any) -> list[float]:
    if not values:
        return []
    return [float(value) for value in values]


def best_index(scores: list[float]) -> int:
    return max(range(len(scores)), key=lambda index: scores[index])


def row_query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id") or row.get("query"))


def negative_items(row: dict[str, Any], pos_score: float) -> list[dict[str, Any]]:
    neg_scores = float_list(row.get("neg_scores"))
    neg_texts = row.get("neg", [])
    neg_doc_ids = row.get("neg_doc_ids", [None] * len(neg_scores))
    neg_sources = row.get("neg_sources", ["unknown"] * len(neg_scores))
    hybrid_scores = row.get("bge_m3_hybrid_neg_scores", [None] * len(neg_scores))
    items = []
    for index, score in enumerate(neg_scores):
        margin = pos_score - score
        label = margin_bin(margin)
        items.append(
            {
                "index": index,
                "text": str(neg_texts[index]),
                "doc_id": neg_doc_ids[index] if index < len(neg_doc_ids) else None,
                "source": str(neg_sources[index]) if index < len(neg_sources) else "unknown",
                "score": score,
                "margin": margin,
                "bin": label,
                "stratum": stratum_for_bin(label),
                "bge_m3_hybrid_score": (
                    hybrid_scores[index] if index < len(hybrid_scores) else None
                ),
            }
        )
    return items


def available_negative_strata(row: dict[str, Any]) -> Counter[str]:
    pos_scores = float_list(row.get("pos_scores"))
    if not pos_scores:
        return Counter()
    pos_score = pos_scores[best_index(pos_scores)]
    return Counter(item["stratum"] for item in negative_items(row, pos_score))


def meets_minimum_available_mix(row: dict[str, Any], args: argparse.Namespace) -> bool:
    counts = available_negative_strata(row)
    if counts.get("hard", 0) < args.min_hard_available:
        return False
    if counts.get("middle", 0) < args.min_middle_available:
        return False
    if counts.get("easy", 0) < args.min_easy_available:
        return False
    if args.min_best_margin is None and args.max_best_margin is None:
        return True
    pos_scores, neg_scores = row_scores(row)
    margin = best_margin(pos_scores, neg_scores)
    if args.min_best_margin is not None and margin < args.min_best_margin:
        return False
    if args.max_best_margin is not None and margin > args.max_best_margin:
        return False
    return True


def stratum_for_bin(label: str) -> str:
    for stratum, labels in NEGATIVE_STRATA.items():
        if label in labels:
            return stratum
    return "unknown"


def stable_key(seed: str, row: dict[str, Any], item: dict[str, Any]) -> float:
    query = row_query_id(row)
    doc_id = item.get("doc_id") or item["index"]
    return stable_fraction(f"{query}:{doc_id}:{item['score']}", seed)


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
        item
        for item in items
        if item["stratum"] == stratum and item["index"] not in selected_indices
    ]
    if stratum == "hard":
        candidates.sort(key=lambda item: (item["margin"], stable_key(seed, row, item)))
    elif stratum == "middle":
        candidates.sort(key=lambda item: (abs(item["margin"] - 2.5), stable_key(seed, row, item)))
    else:
        candidates.sort(key=lambda item: (stable_key(seed, row, item), item["margin"]))
    chosen = candidates[:count]
    selected_indices.update(item["index"] for item in chosen)
    return chosen


def choose_negatives(
    row: dict[str, Any],
    *,
    pos_score: float,
    negatives_per_query: int,
    hard_negatives: int,
    middle_negatives: int,
    easy_negatives: int,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = negative_items(row, pos_score)
    selected_indices: set[int] = set()
    selected: list[dict[str, Any]] = []
    requested = {
        "hard": hard_negatives,
        "middle": middle_negatives,
        "easy": easy_negatives,
    }
    selected_by_stratum: Counter[str] = Counter()
    available_by_stratum = Counter(item["stratum"] for item in items)

    for stratum, count in requested.items():
        chosen = choose_from_stratum(
            row=row,
            items=items,
            stratum=stratum,
            count=count,
            seed=seed,
            selected_indices=selected_indices,
        )
        selected.extend(chosen)
        selected_by_stratum.update(item["stratum"] for item in chosen)

    if len(selected) < negatives_per_query:
        remaining = [item for item in items if item["index"] not in selected_indices]
        remaining.sort(key=lambda item: (stable_key(seed, row, item), item["margin"]))
        needed = negatives_per_query - len(selected)
        fill = remaining[:needed]
        selected.extend(fill)
        selected_indices.update(item["index"] for item in fill)
        selected_by_stratum.update(item["stratum"] for item in fill)

    selected = selected[:negatives_per_query]
    selected_by_stratum = Counter(item["stratum"] for item in selected)
    selected_by_bin = Counter(item["bin"] for item in selected)
    diagnostics = {
        "available_by_stratum": dict(available_by_stratum),
        "selected_by_stratum": dict(selected_by_stratum),
        "selected_by_bin": dict(selected_by_bin),
        "requested_mix_met": all(
            selected_by_stratum.get(stratum, 0) >= count
            for stratum, count in requested.items()
        ),
    }
    return selected, diagnostics


def prepare_output_row(
    row: dict[str, Any],
    *,
    negatives_per_query: int,
    hard_negatives: int,
    middle_negatives: int,
    easy_negatives: int,
    score_scale: float,
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    pos_scores = float_list(row.get("pos_scores"))
    neg_scores = float_list(row.get("neg_scores"))
    if not pos_scores or len(neg_scores) < negatives_per_query:
        return None

    pos_idx = best_index(pos_scores)
    pos_score = pos_scores[pos_idx]
    selected, diagnostics = choose_negatives(
        row,
        pos_score=pos_score,
        negatives_per_query=negatives_per_query,
        hard_negatives=hard_negatives,
        middle_negatives=middle_negatives,
        easy_negatives=easy_negatives,
        seed=seed,
    )
    if len(selected) < negatives_per_query:
        return None

    pos_doc_ids = row.get("pos_doc_ids", [None] * len(pos_scores))
    hybrid_pos_scores = row.get("bge_m3_hybrid_pos_scores", [None] * len(pos_scores))
    output = {
        "query_id": row.get("query_id"),
        "query": row["query"],
        "pos_doc_ids": [pos_doc_ids[pos_idx] if pos_idx < len(pos_doc_ids) else None],
        "pos": [row["pos"][pos_idx]],
        "pos_scores": [pos_score * score_scale],
        "neg_doc_ids": [item["doc_id"] for item in selected],
        "neg": [item["text"] for item in selected],
        "neg_scores": [item["score"] * score_scale for item in selected],
        "neg_sources": [item["source"] for item in selected],
        "neg_strata": [item["stratum"] for item in selected],
        "neg_margin_bins": [item["bin"] for item in selected],
        "original_reranker_pos_score": pos_score,
        "original_reranker_neg_scores": [item["score"] for item in selected],
        "original_teacher_margins": [pos_score - item["score"] for item in selected],
        "teacher": {
            "source": "prepare_stratified_teacher_rows.py",
            "score_scale": score_scale,
            "negative_mix": {
                "hard": hard_negatives,
                "middle": middle_negatives,
                "easy": easy_negatives,
            },
            "source_teacher": row.get("reranker_teacher", {}),
        },
        "source": row.get("source", {}),
    }
    if hybrid_pos_scores and hybrid_pos_scores[pos_idx] is not None:
        output["bge_m3_hybrid_pos_scores"] = [hybrid_pos_scores[pos_idx]]
    selected_hybrid_scores = [item["bge_m3_hybrid_score"] for item in selected]
    if any(score is not None for score in selected_hybrid_scores):
        output["bge_m3_hybrid_neg_scores"] = selected_hybrid_scores

    return output, diagnostics


def read_candidate_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_rows = 0
    rows_with_scores = 0
    short_negative_rows = 0
    insufficient_mix_rows = 0
    nontrain_rows = 0
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
            pos_scores, neg_scores = float_list(row.get("pos_scores")), float_list(row.get("neg_scores"))
            if not pos_scores or not neg_scores:
                continue
            rows_with_scores += 1
            if len(neg_scores) < args.negatives_per_query:
                short_negative_rows += 1
                continue
            if not meets_minimum_available_mix(row, args):
                insufficient_mix_rows += 1
                continue
            rows.append(row)

    stats = {
        "input_jsonl": args.input_jsonl,
        "total_rows": total_rows,
        "rows_with_scores": rows_with_scores,
        "candidate_rows": len(rows),
        "nontrain_source_rows_seen": nontrain_rows,
        "short_negative_rows": short_negative_rows,
        "insufficient_minimum_mix_rows": insufficient_mix_rows,
    }
    return rows, stats


def split_candidates(
    rows: list[dict[str, Any]],
    *,
    heldout_ratio: float,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for row in rows:
        query_id = row_query_id(row)
        if stable_fraction(query_id, seed) < heldout_ratio:
            heldout.append(row)
        else:
            train.append(row)
    if not train:
        raise ValueError("No train rows selected")
    if not heldout:
        raise ValueError("No held-out rows selected")
    return train, heldout


def row_bin(row: dict[str, Any]) -> str:
    pos_scores, neg_scores = row_scores(row)
    return margin_bin(best_margin(pos_scores, neg_scores))


def row_scores(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    return float_list(row.get("pos_scores")), float_list(row.get("neg_scores"))


def select_balanced_rows(rows: list[dict[str, Any]], max_rows: int, seed: str) -> list[dict[str, Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return sorted(rows, key=lambda row: stable_fraction(row_query_id(row), seed))

    buckets: dict[str, list[dict[str, Any]]] = {label: [] for label in ROW_BIN_ORDER}
    for row in rows:
        buckets[row_bin(row)].append(row)
    for label, bucket in buckets.items():
        bucket.sort(key=lambda row: stable_fraction(f"{label}:{row_query_id(row)}", seed))

    selected: list[dict[str, Any]] = []
    while len(selected) < max_rows and any(buckets.values()):
        for label in ROW_BIN_ORDER:
            bucket = buckets[label]
            if bucket:
                selected.append(bucket.pop(0))
                if len(selected) >= max_rows:
                    break
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_split(
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    *,
    temperatures: list[float],
) -> dict[str, Any]:
    strata_counts = Counter(
        stratum
        for row in rows
        for stratum in row.get("neg_strata", [])
    )
    bin_counts = Counter(
        label
        for row in rows
        for label in row.get("neg_margin_bins", [])
    )
    return {
        "rows": len(rows),
        "negative_pairs": sum(len(row.get("neg", [])) for row in rows),
        "negative_strata_counts": dict(strata_counts),
        "negative_margin_bin_counts": dict(bin_counts),
        "rows_meeting_requested_mix": sum(
            1 for diagnostic in diagnostics if diagnostic.get("requested_mix_met")
        ),
        "distribution": analyze_rows(rows, temperatures),
    }


def prepare_splits(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validate_args(args)
    candidate_rows, input_stats = read_candidate_rows(args)
    train_candidates, eval_candidates = split_candidates(
        candidate_rows,
        heldout_ratio=args.heldout_ratio,
        seed=args.seed,
    )
    train_candidates = select_balanced_rows(
        train_candidates,
        args.max_train_rows,
        f"{args.seed}:train",
    )
    eval_candidates = select_balanced_rows(
        eval_candidates,
        args.max_eval_rows,
        f"{args.seed}:eval",
    )

    def prepare_many(source_rows: list[dict[str, Any]], split_seed: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        output_rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for row in source_rows:
            prepared = prepare_output_row(
                row,
                negatives_per_query=args.negatives_per_query,
                hard_negatives=args.hard_negatives,
                middle_negatives=args.middle_negatives,
                easy_negatives=args.easy_negatives,
                score_scale=args.score_scale,
                seed=split_seed,
            )
            if prepared is None:
                continue
            output_row, diagnostic = prepared
            output_rows.append(output_row)
            diagnostics.append(diagnostic)
        return output_rows, diagnostics

    train_rows, train_diagnostics = prepare_many(train_candidates, f"{args.seed}:train-neg")
    eval_rows, eval_diagnostics = prepare_many(eval_candidates, f"{args.seed}:eval-neg")
    train_queries = {row_query_id(row) for row in train_rows}
    eval_queries = {row_query_id(row) for row in eval_rows}
    overlap = train_queries.intersection(eval_queries)
    if overlap:
        raise ValueError(f"Train/eval query overlap detected: {sorted(overlap)[:5]}")

    temperatures = [1.0, 0.5, 0.2, 0.1]
    summary = {
        **input_stats,
        "splitter": {
            "seed": args.seed,
            "heldout_ratio": args.heldout_ratio,
            "max_train_rows": args.max_train_rows,
            "max_eval_rows": args.max_eval_rows,
            "query_overlap": 0,
            "row_selection": "query-disjoint stable split plus row-margin-bin round-robin",
        },
        "negative_mix": {
            "negatives_per_query": args.negatives_per_query,
            "hard": args.hard_negatives,
            "middle": args.middle_negatives,
            "easy": args.easy_negatives,
            "minimum_available": {
                "hard": args.min_hard_available,
                "middle": args.min_middle_available,
                "easy": args.min_easy_available,
            },
            "min_best_margin": args.min_best_margin,
            "max_best_margin": args.max_best_margin,
        },
        "score_transform": {
            "score_scale": args.score_scale,
            "rationale": "soften teacher-score softmax while preserving teacher ordering",
        },
        "train": summarize_split(train_rows, train_diagnostics, temperatures=temperatures),
        "heldout": summarize_split(eval_rows, eval_diagnostics, temperatures=temperatures),
        "raw_training_data_committed": False,
        "model_checkpoints_committed": False,
    }
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

    train_rows, eval_rows, summary = prepare_splits(args)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
