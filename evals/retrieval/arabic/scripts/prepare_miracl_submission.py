#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any


RUN_LINE_RE = re.compile(r"^\S+\s+Q0\s+\S+\s+\d+\s+[-+0-9.eE]+\s+\S+$")


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and package MIRACL official-format TREC run files."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--subset", default="ar")
    parser.add_argument("--splits", nargs="+", default=["dev", "test-a"])
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--package-dir-name", default="miracl_submission")
    parser.add_argument("--validation-json", default="")
    return parser.parse_args()


def sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):020d}")
    except ValueError:
        return (1, value)


def read_miracl_tsv(filename: str) -> list[list[str]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="miracl/miracl",
        filename=filename,
        repo_type="dataset",
        token=hf_token(),
    )
    text = Path(path).read_text(encoding="utf-8")
    return [line.rstrip("\n").split("\t") for line in text.splitlines() if line.strip()]


def load_topic_ids(subset: str, split: str) -> list[str]:
    topics_file = f"miracl-v1.0-{subset}/topics/topics.miracl-v1.0-{subset}-{split}.tsv"
    return sorted((row[0] for row in read_miracl_tsv(topics_file)), key=sort_key)


def load_qrels(subset: str, split: str) -> dict[str, set[str]] | None:
    if split != "dev":
        return None
    qrels_file = f"miracl-v1.0-{subset}/qrels/qrels.miracl-v1.0-{subset}-dev.tsv"
    qrels: dict[str, set[str]] = {}
    for qid, _unused, docid, rel in read_miracl_tsv(qrels_file):
        if int(rel) > 0:
            qrels.setdefault(qid, set()).add(docid)
    return qrels


def metrics_for_query(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int = 10) -> dict[str, float]:
    hits = [1 if doc_id in relevant_doc_ids else 0 for doc_id in ranked_doc_ids[:k]]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0

    precision_sum = 0.0
    found = 0
    reciprocal = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            found += 1
            precision_sum += found / rank
            if reciprocal == 0.0:
                reciprocal = 1.0 / rank
    denom = min(len(relevant_doc_ids), k)
    return {
        "ndcg_at_10": ndcg,
        "map_at_10": precision_sum / denom if denom else 0.0,
        "mrr_at_10": reciprocal,
        "recall_at_10": found / len(relevant_doc_ids) if relevant_doc_ids else 0.0,
        "precision_at_10": found / k,
        "hit_rate_at_10": 1.0 if found else 0.0,
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def validate_run_file(
    path: Path,
    expected_query_ids: list[str],
    depth: int,
    qrels: dict[str, set[str]] | None,
) -> dict[str, Any]:
    by_query: dict[str, list[tuple[int, str, float, str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not RUN_LINE_RE.match(line):
                raise ValueError(f"{path}:{line_number}: invalid TREC run line: {line!r}")
            qid, q0, docid, rank, score, run_id = line.split()
            if q0 != "Q0":
                raise ValueError(f"{path}:{line_number}: second field must be Q0")
            by_query.setdefault(qid, []).append((int(rank), docid, float(score), run_id))

    expected_set = set(expected_query_ids)
    actual_set = set(by_query)
    missing = sorted(expected_set - actual_set, key=sort_key)
    extra = sorted(actual_set - expected_set, key=sort_key)
    if missing or extra:
        raise ValueError(
            f"{path}: query mismatch: missing={missing[:5]} extra={extra[:5]} "
            f"counts=({len(missing)}, {len(extra)})"
        )

    run_ids: set[str] = set()
    ranked_doc_ids_by_query: dict[str, list[str]] = {}
    for qid in expected_query_ids:
        rows = by_query[qid]
        if len(rows) != depth:
            raise ValueError(f"{path}: query {qid} has {len(rows)} rows, expected {depth}")
        rows_by_rank = sorted(rows)
        ranks = [rank for rank, _docid, _score, _run_id in rows_by_rank]
        if ranks != list(range(1, depth + 1)):
            raise ValueError(f"{path}: query {qid} ranks are not 1..{depth}")
        docids = [docid for _rank, docid, _score, _run_id in rows_by_rank]
        if len(set(docids)) != len(docids):
            raise ValueError(f"{path}: query {qid} has duplicate docids")
        run_ids.update(run_id for _rank, _docid, _score, run_id in rows_by_rank)
        ranked_doc_ids_by_query[qid] = docids

    metrics = None
    if qrels is not None:
        metrics_rows = [
            metrics_for_query(ranked_doc_ids_by_query[qid], qrels.get(qid, set()), 10)
            for qid in expected_query_ids
        ]
        metrics = mean_metrics(metrics_rows)
        metrics["main_score"] = metrics["ndcg_at_10"]

    return {
        "file": str(path),
        "queries": len(expected_query_ids),
        "depth": depth,
        "lines": len(expected_query_ids) * depth,
        "run_ids": sorted(run_ids),
        "metrics": metrics,
    }


def write_submission_zip(run_files: list[Path], output_zip: Path, package_dir_name: str) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for run_file in run_files:
            archive.write(run_file, arcname=f"{package_dir_name}/{run_file.name}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_zip = Path(args.output_zip)

    validations: dict[str, Any] = {
        "subset": args.subset,
        "splits": args.splits,
        "depth": args.depth,
        "run_files": [],
    }
    run_files: list[Path] = []
    for split in args.splits:
        run_file = run_dir / f"{args.subset}_{split}.txt"
        if not run_file.exists():
            raise SystemExit(f"Missing required run file: {run_file}")
        topic_ids = load_topic_ids(args.subset, split)
        qrels = load_qrels(args.subset, split)
        validations["run_files"].append(
            validate_run_file(run_file, topic_ids, args.depth, qrels)
        )
        run_files.append(run_file)

    write_submission_zip(run_files, output_zip, args.package_dir_name)
    validations["zip_file"] = str(output_zip)
    validations["zip_entries"] = zipfile.ZipFile(output_zip).namelist()

    validation_json = Path(args.validation_json) if args.validation_json else output_zip.with_suffix(".validation.json")
    validation_json.write_text(json.dumps(validations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(validations, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
