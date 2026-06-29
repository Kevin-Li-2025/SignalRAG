#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_miracl_bge_m3_hybrid import (  # noqa: E402
    DATASET_NAME,
    DEFAULT_REVISION,
    build_sparse_query_postings,
    encode_bge,
    gather_candidates,
    load_bge_model,
    merge_topk,
    sort_key,
    sparse_scores_for_chunk,
    text_from_doc,
    write_json,
)

MIRACL_QRELS_NAME = "miracl/miracl"
MIRACL_CORPUS_NAME = "miracl/miracl-corpus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MIRACL Arabic teacher training data with BGE-M3 hybrid "
            "dense+sparse candidate mining and dense+sparse+ColBERT scores."
        )
    )
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--model-path", default="", help="Local model path, if pre-downloaded.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset", default="ar")
    parser.add_argument("--query-source", choices=["miracl", "mteb"], default="miracl")
    parser.add_argument("--corpus-source", choices=["miracl", "mteb"], default="mteb")
    parser.add_argument("--split", default="train")
    parser.add_argument("--corpus-split", default="", help="Corpus split; defaults to --split.")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-passage-length", type=int, default=256)
    parser.add_argument("--encode-chunk-size", type=int, default=1000)
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--rerank-limit", type=int, default=100)
    parser.add_argument("--negatives-per-query", type=int, default=8)
    parser.add_argument("--positives-per-query", type=int, default=2)
    parser.add_argument("--dense-weight", type=float, default=0.4)
    parser.add_argument("--sparse-weight", type=float, default=0.2)
    parser.add_argument("--colbert-weight", type=float, default=0.4)
    parser.add_argument("--corpus-limit", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--prefer-judged-negatives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-unjudged-negatives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scan-missing-positives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def load_dataset_with_optional_trust(
    path: str,
    name: str,
    split: str,
    *,
    revision: str | None = None,
):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(path, name, **kwargs)


def load_mteb_queries_and_qrels(
    args: argparse.Namespace,
) -> tuple[list[str], list[str], dict[str, set[str]], dict[str, set[str]]]:
    print(f"[teacher] loading MTEB {args.subset} queries/qrels split={args.split}", flush=True)
    queries_ds = load_dataset_with_optional_trust(
        DATASET_NAME,
        f"{args.subset}-queries",
        args.split,
        revision=args.revision,
    )
    qrels_ds = load_dataset_with_optional_trust(
        DATASET_NAME,
        f"{args.subset}-qrels",
        args.split,
        revision=args.revision,
    )

    positive_qrels: dict[str, set[str]] = {}
    judged_negative_qrels: dict[str, set[str]] = {}
    for row in qrels_ds:
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        if float(row.get("score", 0)) > 0:
            positive_qrels.setdefault(query_id, set()).add(corpus_id)
        else:
            judged_negative_qrels.setdefault(query_id, set()).add(corpus_id)

    query_ids = sorted(positive_qrels, key=sort_key)
    if args.query_limit:
        query_ids = query_ids[: args.query_limit]
    positive_qrels = {query_id: positive_qrels[query_id] for query_id in query_ids}
    judged_negative_qrels = {
        query_id: judged_negative_qrels.get(query_id, set())
        for query_id in query_ids
    }

    query_text_by_id = {
        str(row["_id"]): str(row["text"])
        for row in queries_ds
        if str(row["_id"]) in positive_qrels
    }
    query_ids = [query_id for query_id in query_ids if query_id in query_text_by_id]
    query_texts = [query_text_by_id[query_id] for query_id in query_ids]
    positive_qrels = {query_id: positive_qrels[query_id] for query_id in query_ids}
    judged_negative_qrels = {
        query_id: judged_negative_qrels.get(query_id, set())
        for query_id in query_ids
    }
    print(
        f"[teacher] selected queries={len(query_ids)} positives="
        f"{sum(len(values) for values in positive_qrels.values())} "
        f"judged_negatives={sum(len(values) for values in judged_negative_qrels.values())}",
        flush=True,
    )
    return query_ids, query_texts, positive_qrels, judged_negative_qrels


def download_with_fallback(path_parts: str, output_path: Path) -> None:
    endpoints = []
    configured = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if configured:
        endpoints.append(configured)
    endpoints.append("https://huggingface.co")

    errors = []
    for endpoint in dict.fromkeys(endpoints):
        url = f"{endpoint}/{path_parts.lstrip('/')}"
        try:
            print(f"[teacher] downloading {url}", flush=True)
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--retry",
                    "3",
                    "--retry-delay",
                    "2",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "120",
                    "-o",
                    str(output_path),
                    url,
                ],
                check=True,
            )
            return
        except Exception as curl_exc:  # pragma: no cover - exercised on remote network paths.
            try:
                urllib.request.urlretrieve(url, output_path)
                return
            except Exception as urllib_exc:  # pragma: no cover
                exc = f"curl={curl_exc}; urllib={urllib_exc}"
            errors.append(f"{url}: {exc}")
            if output_path.exists():
                output_path.unlink()
    raise RuntimeError("Unable to download MIRACL source TSV:\n" + "\n".join(errors))


def load_miracl_queries_and_qrels(
    args: argparse.Namespace,
) -> tuple[list[str], list[str], dict[str, set[str]], dict[str, set[str]]]:
    print(f"[teacher] loading MIRACL {args.subset} TSV queries/qrels split={args.split}", flush=True)
    source_cache_dir = Path(args.output_dir) / "source_cache"
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    topics_path = source_cache_dir / f"topics.miracl-v1.0-{args.subset}-{args.split}.tsv"
    qrels_path = source_cache_dir / f"qrels.miracl-v1.0-{args.subset}-{args.split}.tsv"
    topics_url_path = f"datasets/miracl/miracl/resolve/main/miracl-v1.0-{args.subset}/topics/{topics_path.name}"
    qrels_url_path = f"datasets/miracl/miracl/resolve/main/miracl-v1.0-{args.subset}/qrels/{qrels_path.name}"
    for url_path, path in [(topics_url_path, topics_path), (qrels_url_path, qrels_path)]:
        if not path.exists():
            download_with_fallback(url_path, path)

    query_text_by_id: dict[str, str] = {}
    positive_qrels: dict[str, set[str]] = {}
    judged_negative_qrels: dict[str, set[str]] = {}
    with topics_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            query_id, query = line.split("\t", 1)
            query_text_by_id[str(query_id)] = query

    with qrels_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            query_id, _unused, doc_id, relevance = parts[:4]
            if float(relevance) > 0:
                positive_qrels.setdefault(str(query_id), set()).add(str(doc_id))
            else:
                judged_negative_qrels.setdefault(str(query_id), set()).add(str(doc_id))

    query_ids = sorted(positive_qrels, key=sort_key)
    if args.query_limit:
        query_ids = query_ids[: args.query_limit]
    positive_qrels = {query_id: positive_qrels[query_id] for query_id in query_ids}
    judged_negative_qrels = {
        query_id: judged_negative_qrels.get(query_id, set())
        for query_id in query_ids
    }
    query_ids = [query_id for query_id in query_ids if query_id in query_text_by_id]
    query_texts = [query_text_by_id[query_id] for query_id in query_ids]
    print(
        f"[teacher] selected queries={len(query_ids)} positives="
        f"{sum(len(values) for values in positive_qrels.values())} "
        f"judged_negatives={sum(len(values) for values in judged_negative_qrels.values())}",
        flush=True,
    )
    return query_ids, query_texts, positive_qrels, judged_negative_qrels


def load_queries_and_qrels(
    args: argparse.Namespace,
) -> tuple[list[str], list[str], dict[str, set[str]], dict[str, set[str]]]:
    if args.query_source == "miracl":
        return load_miracl_queries_and_qrels(args)
    return load_mteb_queries_and_qrels(args)


def doc_id_from_row(row: dict[str, Any]) -> str:
    if "_id" in row:
        return str(row["_id"])
    return str(row["docid"])


def doc_ids_from_batch(batch: dict[str, list[Any]]) -> list[str]:
    if "_id" in batch:
        return [str(doc_id) for doc_id in batch["_id"]]
    return [str(doc_id) for doc_id in batch["docid"]]


def load_corpus(args: argparse.Namespace):
    corpus_split = args.corpus_split or ("train" if args.corpus_source == "miracl" else args.split)
    if args.corpus_source == "miracl":
        print(f"[teacher] loading MIRACL corpus split={corpus_split}", flush=True)
        return load_dataset_with_optional_trust(MIRACL_CORPUS_NAME, args.subset, corpus_split), corpus_split
    print(f"[teacher] loading MTEB corpus split={corpus_split}", flush=True)
    return (
        load_dataset_with_optional_trust(
            DATASET_NAME,
            f"{args.subset}-corpus",
            corpus_split,
            revision=args.revision,
        ),
        corpus_split,
    )


def scan_missing_positive_indices(
    corpus_ds: object,
    missing_doc_ids: set[str],
    doc_id_by_index: dict[int, str],
    doc_index_by_id: dict[str, int],
) -> None:
    if not missing_doc_ids:
        return
    started = time.monotonic()
    total = len(corpus_ds)
    initial_missing = len(missing_doc_ids)
    print(
        f"[teacher] scanning full corpus for {len(missing_doc_ids)} missing positive doc ids",
        flush=True,
    )
    for index in range(total):
        row = corpus_ds[index]
        doc_id = doc_id_from_row(row)
        if doc_id not in missing_doc_ids:
            continue
        doc_id_by_index[index] = doc_id
        doc_index_by_id[doc_id] = index
        missing_doc_ids.remove(doc_id)
        if not missing_doc_ids:
            break
    elapsed = time.monotonic() - started
    print(
        f"[teacher] positive scan complete found={initial_missing - len(missing_doc_ids)} "
        f"remaining_missing={len(missing_doc_ids)} elapsed={elapsed:.1f}s",
        flush=True,
    )


def score_pairs(
    model: object,
    pairs: list[list[str]],
    batch_size: int,
    max_passage_length: int,
    weights: list[float],
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        result = model.compute_score(
            batch,
            batch_size=batch_size,
            max_passage_length=max_passage_length,
            weights_for_different_modes=weights,
        )
        scores.extend(float(value) for value in result["colbert+sparse+dense"])
    return scores


def build_text_index(
    corpus_ds: object,
    indices: set[int],
) -> tuple[dict[int, str], dict[int, str]]:
    doc_text_by_index: dict[int, str] = {}
    doc_id_by_index: dict[int, str] = {}
    started = time.monotonic()
    ordered_indices = sorted(indices)
    for offset, index in enumerate(ordered_indices, start=1):
        row = corpus_ds[int(index)]
        doc_id_by_index[int(index)] = doc_id_from_row(row)
        doc_text_by_index[int(index)] = text_from_doc(row)
        if offset % 5000 == 0 or offset == len(ordered_indices):
            elapsed = time.monotonic() - started
            print(
                f"[teacher-progress] candidate/positive texts: "
                f"{offset}/{len(ordered_indices)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return doc_text_by_index, doc_id_by_index


def select_negative_rows(
    scored_candidates: list[dict[str, Any]],
    judged_negative_ids: set[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    judged: list[dict[str, Any]] = []
    unjudged: list[dict[str, Any]] = []
    for row in scored_candidates:
        if row["doc_id"] in judged_negative_ids:
            judged.append({**row, "negative_source": "judged_negative"})
        else:
            unjudged.append({**row, "negative_source": "unjudged_candidate"})

    if args.prefer_judged_negatives:
        ordered = judged
        if args.allow_unjudged_negatives:
            ordered = [*ordered, *unjudged]
    else:
        ordered = [*judged, *unjudged] if args.allow_unjudged_negatives else judged
        ordered.sort(key=lambda row: row["teacher_score"], reverse=True)
    return ordered[: args.negatives_per_query]


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.force_rebuild and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_ids, query_texts, positive_qrels, judged_negative_qrels = load_queries_and_qrels(args)
    model = load_bge_model(args)

    query_outputs = encode_bge(
        model,
        query_texts,
        args.batch_size,
        args.max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    query_dense = np.asarray(query_outputs["dense_vecs"], dtype=np.float32)
    query_sparse = query_outputs["lexical_weights"]
    query_postings = build_sparse_query_postings(query_sparse)
    query_count = len(query_ids)

    dense_top_scores = np.full((query_count, args.candidate_top_k), -np.inf, dtype=np.float32)
    dense_top_indices = np.full((query_count, args.candidate_top_k), -1, dtype=np.int64)
    sparse_top_scores = np.full((query_count, args.candidate_top_k), -np.inf, dtype=np.float32)
    sparse_top_indices = np.full((query_count, args.candidate_top_k), -1, dtype=np.int64)

    corpus_ds, corpus_split = load_corpus(args)
    corpus_count = min(len(corpus_ds), args.corpus_limit) if args.corpus_limit else len(corpus_ds)
    doc_id_by_index: dict[int, str] = {}
    doc_index_by_id: dict[str, int] = {}

    candidate_started = time.monotonic()
    for start in range(0, corpus_count, args.encode_chunk_size):
        end = min(start + args.encode_chunk_size, corpus_count)
        batch = corpus_ds[start:end]
        doc_ids = doc_ids_from_batch(batch)
        texts = [
            text_from_doc({"title": title, "text": text})
            for title, text in zip(batch.get("title", [""] * len(doc_ids)), batch["text"])
        ]
        for offset, doc_id in enumerate(doc_ids):
            index = start + offset
            doc_id_by_index[index] = doc_id
            doc_index_by_id[doc_id] = index

        outputs = encode_bge(
            model,
            texts,
            args.batch_size,
            args.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        corpus_dense = np.asarray(outputs["dense_vecs"], dtype=np.float32)
        dense_scores = query_dense @ corpus_dense.T
        sparse_scores = sparse_scores_for_chunk(
            query_postings,
            outputs["lexical_weights"],
            query_count,
        )
        dense_top_scores, dense_top_indices = merge_topk(
            dense_top_scores,
            dense_top_indices,
            dense_scores,
            start,
            args.candidate_top_k,
        )
        sparse_top_scores, sparse_top_indices = merge_topk(
            sparse_top_scores,
            sparse_top_indices,
            sparse_scores,
            start,
            args.candidate_top_k,
        )
        elapsed = max(time.monotonic() - candidate_started, 1e-6)
        docs_per_second = end / elapsed
        eta_seconds = (corpus_count - end) / max(docs_per_second, 1e-6)
        print(
            f"[teacher-progress] candidates: {end}/{corpus_count} docs "
            f"elapsed={elapsed:.1f}s rate={docs_per_second:.2f} docs/s "
            f"eta={eta_seconds / 3600:.2f}h",
            flush=True,
        )

    candidate_elapsed = time.monotonic() - candidate_started
    candidate_indices = gather_candidates(
        dense_top_scores,
        dense_top_indices,
        sparse_top_scores,
        sparse_top_indices,
        args.dense_weight,
        args.sparse_weight,
        args.rerank_limit,
    )

    if args.scan_missing_positives:
        all_positive_doc_ids = {
            doc_id
            for values in positive_qrels.values()
            for doc_id in values
        }
        scan_missing_positive_indices(
            corpus_ds,
            {doc_id for doc_id in all_positive_doc_ids if doc_id not in doc_index_by_id},
            doc_id_by_index,
            doc_index_by_id,
        )

    positive_indices_by_query: dict[str, list[int]] = {}
    missing_positive_doc_ids: list[str] = []
    for query_id in query_ids:
        indices: list[int] = []
        for doc_id in sorted(positive_qrels[query_id], key=sort_key):
            index = doc_index_by_id.get(doc_id)
            if index is None:
                missing_positive_doc_ids.append(doc_id)
                continue
            indices.append(index)
        positive_indices_by_query[query_id] = indices

    needed_indices = {
        index
        for row in candidate_indices
        for index in row
        if int(index) >= 0
    }
    needed_indices.update(
        index
        for indices in positive_indices_by_query.values()
        for index in indices
        if int(index) >= 0
    )
    doc_text_by_index, text_doc_id_by_index = build_text_index(corpus_ds, needed_indices)
    doc_id_by_index.update(text_doc_id_by_index)

    weights = [args.dense_weight, args.sparse_weight, args.colbert_weight]
    output_jsonl = output_dir / "teacher_train.jsonl"
    row_count = 0
    skipped_no_positive = 0
    skipped_no_negative = 0
    positive_scores_all: list[float] = []
    negative_scores_all: list[float] = []
    selected_negative_sources = {
        "judged_negative": 0,
        "unjudged_candidate": 0,
    }

    rerank_started = time.monotonic()
    rerank_pairs = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for query_index, query_id in enumerate(query_ids):
            query_text = query_texts[query_index]
            positive_indices = [
                index
                for index in positive_indices_by_query[query_id]
                if index in doc_text_by_index
            ]
            if not positive_indices:
                skipped_no_positive += 1
                continue

            positive_pairs = [[query_text, doc_text_by_index[index]] for index in positive_indices]
            positive_scores = score_pairs(
                model,
                positive_pairs,
                args.rerank_batch_size,
                args.max_passage_length,
                weights,
            )
            rerank_pairs += len(positive_pairs)
            scored_positives = sorted(
                [
                    {
                        "doc_id": doc_id_by_index[index],
                        "text": doc_text_by_index[index],
                        "teacher_score": score,
                    }
                    for index, score in zip(positive_indices, positive_scores)
                ],
                key=lambda row: row["teacher_score"],
                reverse=True,
            )[: args.positives_per_query]

            positive_doc_ids = {row["doc_id"] for row in scored_positives}
            all_positive_doc_ids = positive_qrels[query_id]
            candidate_rows = []
            for rank, index in enumerate(candidate_indices[query_index], start=1):
                doc_id = doc_id_by_index.get(index)
                if not doc_id or doc_id in all_positive_doc_ids:
                    continue
                if index not in doc_text_by_index:
                    continue
                candidate_rows.append(
                    {
                        "candidate_rank": rank,
                        "doc_id": doc_id,
                        "text": doc_text_by_index[index],
                    }
                )

            candidate_pairs = [[query_text, row["text"]] for row in candidate_rows]
            candidate_scores = score_pairs(
                model,
                candidate_pairs,
                args.rerank_batch_size,
                args.max_passage_length,
                weights,
            )
            rerank_pairs += len(candidate_pairs)
            scored_candidates = sorted(
                [
                    {
                        **row,
                        "teacher_score": score,
                    }
                    for row, score in zip(candidate_rows, candidate_scores)
                ],
                key=lambda row: row["teacher_score"],
                reverse=True,
            )
            selected_negatives = select_negative_rows(
                scored_candidates,
                judged_negative_qrels[query_id],
                args,
            )
            if not selected_negatives:
                skipped_no_negative += 1
                continue

            pos_scores = [float(row["teacher_score"]) for row in scored_positives]
            neg_scores = [float(row["teacher_score"]) for row in selected_negatives]
            positive_scores_all.extend(pos_scores)
            negative_scores_all.extend(neg_scores)
            for row in selected_negatives:
                selected_negative_sources[row["negative_source"]] += 1

            output_row = {
                "query_id": query_id,
                "query": query_text,
                "pos_doc_ids": [row["doc_id"] for row in scored_positives],
                "pos": [row["text"] for row in scored_positives],
                "pos_scores": pos_scores,
                "neg_doc_ids": [row["doc_id"] for row in selected_negatives],
                "neg": [row["text"] for row in selected_negatives],
                "neg_scores": neg_scores,
                "neg_sources": [row["negative_source"] for row in selected_negatives],
                "neg_candidate_ranks": [row["candidate_rank"] for row in selected_negatives],
                "positive": scored_positives[0]["text"],
                "negative": selected_negatives[0]["text"],
                "teacher": {
                    "model": args.model,
                    "weights_for_different_modes": weights,
                    "candidate_top_k": args.candidate_top_k,
                    "rerank_limit": args.rerank_limit,
                },
                "source": {
                    "query_dataset": MIRACL_QRELS_NAME if args.query_source == "miracl" else DATASET_NAME,
                    "corpus_dataset": MIRACL_CORPUS_NAME if args.corpus_source == "miracl" else DATASET_NAME,
                    "subset": args.subset,
                    "split": args.split,
                    "corpus_split": corpus_split,
                },
            }
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            row_count += 1
            if (query_index + 1) % 25 == 0 or query_index + 1 == len(query_ids):
                elapsed = time.monotonic() - rerank_started
                print(
                    f"[teacher-progress] rows: query={query_index + 1}/{len(query_ids)} "
                    f"written={row_count} pairs_scored={rerank_pairs} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    rerank_elapsed = time.monotonic() - rerank_started
    avg_pos = mean_or_none(positive_scores_all)
    avg_neg = mean_or_none(negative_scores_all)
    summary = {
        "experiment": "miracl-ar-bge-m3-hybrid-teacher-data",
        "model": args.model,
        "model_path": args.model_path,
        "query_dataset": MIRACL_QRELS_NAME if args.query_source == "miracl" else DATASET_NAME,
        "corpus_dataset": MIRACL_CORPUS_NAME if args.corpus_source == "miracl" else DATASET_NAME,
        "mteb_dataset_revision": args.revision if args.query_source == "mteb" or args.corpus_source == "mteb" else None,
        "subset": args.subset,
        "split": args.split,
        "corpus_split": corpus_split,
        "corpus_count": corpus_count,
        "query_count": query_count,
        "positive_qrel_count": sum(len(values) for values in positive_qrels.values()),
        "judged_negative_qrel_count": sum(len(values) for values in judged_negative_qrels.values()),
        "candidate_top_k": args.candidate_top_k,
        "rerank_limit": args.rerank_limit,
        "negatives_per_query": args.negatives_per_query,
        "positives_per_query": args.positives_per_query,
        "weights_for_different_modes": weights,
        "batch_size": args.batch_size,
        "rerank_batch_size": args.rerank_batch_size,
        "max_length": args.max_length,
        "max_passage_length": args.max_passage_length,
        "outputs": {
            "teacher_train_jsonl": output_jsonl.name,
            "raw_training_data_committed": False,
        },
        "counts": {
            "rows_written": row_count,
            "skipped_no_positive": skipped_no_positive,
            "skipped_no_negative": skipped_no_negative,
            "missing_positive_doc_ids": len(missing_positive_doc_ids),
            "selected_negatives": sum(selected_negative_sources.values()),
            "selected_negative_sources": selected_negative_sources,
        },
        "teacher_score_stats": {
            "mean_positive_score": avg_pos,
            "mean_negative_score": avg_neg,
            "mean_positive_minus_negative": (avg_pos - avg_neg) if avg_pos is not None and avg_neg is not None else None,
        },
        "timings": {
            "candidate_elapsed_seconds": candidate_elapsed,
            "candidate_docs_per_second": corpus_count / max(candidate_elapsed, 1e-6),
            "teacher_scoring_elapsed_seconds": rerank_elapsed,
            "teacher_scoring_pairs": rerank_pairs,
            "total_measured_seconds": candidate_elapsed + rerank_elapsed,
        },
        "note": (
            "Generated JSONL contains query, pos/neg texts, doc ids, and BGE-M3 hybrid "
            "teacher scores. Keep it out of git because it is generated training data."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
