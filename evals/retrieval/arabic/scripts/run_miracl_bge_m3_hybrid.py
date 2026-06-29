#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # Allows lightweight CI tests to import formatting helpers.
    np = None  # type: ignore[assignment]


DATASET_NAME = "mteb/MIRACLRetrieval"
DEFAULT_REVISION = "9c09abc13478308c27598f350e31d8f06b9b5481"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BGE-M3 hybrid retrieval on MIRACL Arabic: dense+sparse candidate "
            "generation followed by official dense+sparse+ColBERT reranking."
        )
    )
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--model-path", default="", help="Local model path, if pre-downloaded.")
    parser.add_argument(
        "--head-checkpoint",
        default="",
        help="Optional checkpoint with BGE-M3 sparse_linear and colbert_linear state dicts.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset", default="ar")
    parser.add_argument("--split", default="dev", choices=["dev", "test-a", "test-b"])
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--query-source",
        choices=["auto", "mteb", "miracl-raw"],
        default="auto",
        help="Use MTEB dev queries/qrels, or official MIRACL topic TSV files.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-passage-length", type=int, default=256)
    parser.add_argument("--encode-chunk-size", type=int, default=1000)
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--rerank-limit", type=int, default=100)
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Final ranked hits to keep and export. MIRACL submissions require 100.",
    )
    parser.add_argument("--metric-k", type=int, default=10)
    parser.add_argument("--dense-weight", type=float, default=0.4)
    parser.add_argument("--sparse-weight", type=float, default=0.2)
    parser.add_argument("--colbert-weight", type=float, default=0.4)
    parser.add_argument("--run-id", default="bge-m3-hybrid-r100")
    parser.add_argument("--corpus-limit", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--use-fp16", action="store_true", default=True)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):020d}")
    except ValueError:
        return (1, value)


def text_from_doc(doc: dict[str, Any]) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return text or title


def metrics_for_query(
    ranked_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int = 10,
) -> dict[str, float]:
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
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def miracl_split_for_url(split: str) -> str:
    return {"dev": "dev", "test-a": "test-a", "test-b": "test-b"}[split]


def official_run_split(split: str) -> str:
    return {"dev": "dev", "test-a": "test-a", "test-b": "test-b"}[split]


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


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


def load_miracl_raw_topics_qrels(
    subset: str,
    split: str,
    query_limit: int,
) -> tuple[list[str], list[str], dict[str, set[str]] | None]:
    url_split = miracl_split_for_url(split)
    topics_file = (
        f"miracl-v1.0-{subset}/topics/topics.miracl-v1.0-{subset}-{url_split}.tsv"
    )
    topic_rows = read_miracl_tsv(topics_file)
    query_text_by_id = {qid: text for qid, text in topic_rows}
    query_ids = sorted(query_text_by_id, key=sort_key)
    if query_limit:
        query_ids = query_ids[:query_limit]
    query_texts = [query_text_by_id[query_id] for query_id in query_ids]

    if split != "dev":
        return query_ids, query_texts, None

    qrels_file = f"miracl-v1.0-{subset}/qrels/qrels.miracl-v1.0-{subset}-dev.tsv"
    qrels: dict[str, set[str]] = {}
    for qid, _unused, docid, rel in read_miracl_tsv(qrels_file):
        if int(rel) > 0 and qid in query_text_by_id:
            qrels.setdefault(qid, set()).add(docid)
    qrels = {query_id: qrels.get(query_id, set()) for query_id in query_ids}
    return query_ids, query_texts, qrels


def load_mteb_relevance(
    subset: str,
    split: str,
    revision: str,
    query_limit: int,
) -> tuple[list[str], list[str], dict[str, set[str]]]:
    if split != "dev":
        raise ValueError("mteb/MIRACLRetrieval only exposes the dev split")

    from datasets import load_dataset

    queries_ds = load_dataset(
        DATASET_NAME,
        f"{subset}-queries",
        split=split,
        revision=revision,
    )
    qrels_ds = load_dataset(
        DATASET_NAME,
        f"{subset}-qrels",
        split=split,
        revision=revision,
    )

    qrels: dict[str, set[str]] = {}
    for row in qrels_ds:
        if row.get("score", 0) <= 0:
            continue
        qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    query_ids = sorted(qrels, key=sort_key)
    if query_limit:
        query_ids = query_ids[:query_limit]
        qrels = {query_id: qrels[query_id] for query_id in query_ids}

    query_text_by_id = {
        str(row["_id"]): str(row["text"])
        for row in queries_ds
        if str(row["_id"]) in qrels
    }
    query_ids = [query_id for query_id in query_ids if query_id in query_text_by_id]
    query_texts = [query_text_by_id[query_id] for query_id in query_ids]
    qrels = {query_id: qrels[query_id] for query_id in query_ids}
    return query_ids, query_texts, qrels


def load_queries_and_qrels(
    args: argparse.Namespace,
) -> tuple[list[str], list[str], dict[str, set[str]] | None, str]:
    if args.query_source == "mteb" or (args.query_source == "auto" and args.split == "dev"):
        query_ids, query_texts, qrels = load_mteb_relevance(
            args.subset,
            args.split,
            args.revision,
            args.query_limit,
        )
        return query_ids, query_texts, qrels, "mteb"
    query_ids, query_texts, qrels = load_miracl_raw_topics_qrels(
        args.subset,
        args.split,
        args.query_limit,
    )
    return query_ids, query_texts, qrels, "miracl-raw"


def load_bge_model(args: argparse.Namespace):
    from FlagEmbedding import BGEM3FlagModel

    model_name_or_path = args.model_path or args.model
    model = BGEM3FlagModel(model_name_or_path, use_fp16=args.use_fp16)
    if args.head_checkpoint:
        import torch

        checkpoint_path = Path(args.head_checkpoint)
        device = model.model.sparse_linear.weight.device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.model.sparse_linear.load_state_dict(checkpoint["sparse_linear"])
        model.model.colbert_linear.load_state_dict(checkpoint["colbert_linear"])
        print(f"[hybrid] loaded head checkpoint {checkpoint_path}", flush=True)
    return model


def encode_bge(
    model: object,
    texts: list[str],
    batch_size: int,
    max_length: int,
    *,
    return_dense: bool,
    return_sparse: bool,
    return_colbert_vecs: bool,
) -> dict[str, Any]:
    return model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=return_dense,
        return_sparse=return_sparse,
        return_colbert_vecs=return_colbert_vecs,
    )


def merge_topk(
    scores: np.ndarray,
    indices: np.ndarray,
    chunk_scores: np.ndarray,
    chunk_start: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    chunk_k = min(top_k, chunk_scores.shape[1])
    chunk_part = np.argpartition(-chunk_scores, kth=chunk_k - 1, axis=1)[:, :chunk_k]
    chunk_values = np.take_along_axis(chunk_scores, chunk_part, axis=1)
    chunk_indices = chunk_part + chunk_start

    merged_scores = np.concatenate([scores, chunk_values], axis=1)
    merged_indices = np.concatenate([indices, chunk_indices], axis=1)
    keep = np.argpartition(-merged_scores, kth=top_k - 1, axis=1)[:, :top_k]
    kept_scores = np.take_along_axis(merged_scores, keep, axis=1)
    kept_indices = np.take_along_axis(merged_indices, keep, axis=1)
    order = np.argsort(-kept_scores, axis=1)
    return (
        np.take_along_axis(kept_scores, order, axis=1),
        np.take_along_axis(kept_indices, order, axis=1),
    )


def build_sparse_query_postings(
    query_sparse: list[dict[str, float]],
) -> dict[str, list[tuple[int, float]]]:
    postings: dict[str, list[tuple[int, float]]] = {}
    for query_index, weights in enumerate(query_sparse):
        for token, weight in weights.items():
            postings.setdefault(str(token), []).append((query_index, float(weight)))
    return postings


def sparse_scores_for_chunk(
    query_postings: dict[str, list[tuple[int, float]]],
    doc_sparse: list[dict[str, float]],
    query_count: int,
) -> np.ndarray:
    scores = np.zeros((query_count, len(doc_sparse)), dtype=np.float32)
    for doc_index, weights in enumerate(doc_sparse):
        for token, doc_weight in weights.items():
            for query_index, query_weight in query_postings.get(str(token), ()):
                scores[query_index, doc_index] += query_weight * float(doc_weight)
    return scores


def gather_candidates(
    dense_scores: np.ndarray,
    dense_indices: np.ndarray,
    sparse_scores: np.ndarray,
    sparse_indices: np.ndarray,
    dense_weight: float,
    sparse_weight: float,
    rerank_limit: int,
) -> list[list[int]]:
    all_candidates: list[list[int]] = []
    for query_index in range(dense_indices.shape[0]):
        combined: dict[int, dict[str, float]] = {}
        for score, index in zip(dense_scores[query_index], dense_indices[query_index]):
            if int(index) < 0:
                continue
            combined.setdefault(int(index), {"dense": 0.0, "sparse": 0.0})["dense"] = float(score)
        for score, index in zip(sparse_scores[query_index], sparse_indices[query_index]):
            if int(index) < 0:
                continue
            combined.setdefault(int(index), {"dense": 0.0, "sparse": 0.0})["sparse"] = float(score)
        ranked = sorted(
            combined.items(),
            key=lambda item: dense_weight * item[1]["dense"] + sparse_weight * item[1]["sparse"],
            reverse=True,
        )
        all_candidates.append([index for index, _scores in ranked[:rerank_limit]])
    return all_candidates


def write_trec_run(
    path: Path,
    ranked_by_query: dict[str, list[dict[str, float | int | str]]],
    query_ids: list[str],
    run_id: str,
    depth: int,
) -> dict[str, int]:
    line_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for query_id in query_ids:
            rows = ranked_by_query.get(query_id, [])
            if len(rows) < depth:
                raise ValueError(
                    f"Query {query_id} has {len(rows)} hits, but MIRACL requires {depth}."
                )
            seen_doc_ids: set[str] = set()
            for expected_rank, row in enumerate(rows[:depth], start=1):
                doc_id = str(row["docid"])
                rank = int(row["rank"])
                score = float(row["score"])
                if rank != expected_rank:
                    raise ValueError(
                        f"Query {query_id} rank mismatch: {rank} != {expected_rank}"
                    )
                if doc_id in seen_doc_ids:
                    raise ValueError(f"Query {query_id} has duplicate docid {doc_id}")
                seen_doc_ids.add(doc_id)
                handle.write(f"{query_id} Q0 {doc_id} {rank} {score:.10f} {run_id}\n")
                line_count += 1
    return {"queries": len(query_ids), "depth": depth, "lines": line_count}


def rerank_candidates(
    model: object,
    args: argparse.Namespace,
    query_ids: list[str],
    query_texts: list[str],
    candidate_indices: list[list[int]],
    doc_text_by_index: dict[int, str],
    doc_id_by_index: dict[int, str],
) -> tuple[dict[str, list[dict[str, float | int | str]]], dict[str, Any]]:
    weights = [args.dense_weight, args.sparse_weight, args.colbert_weight]
    ranked_by_query: dict[str, list[dict[str, float | int | str]]] = {}
    timings = {
        "rerank_elapsed_seconds": 0.0,
        "rerank_pairs": 0,
    }
    started = time.monotonic()
    for query_index, query_id in enumerate(query_ids):
        candidates = candidate_indices[query_index]
        pairs = [[query_texts[query_index], doc_text_by_index[index]] for index in candidates]
        if not pairs:
            ranked_by_query[query_id] = []
            continue
        scores: list[float] = []
        for start in range(0, len(pairs), args.rerank_batch_size):
            batch = pairs[start : start + args.rerank_batch_size]
            result = model.compute_score(
                batch,
                batch_size=args.rerank_batch_size,
                max_passage_length=args.max_passage_length,
                weights_for_different_modes=weights,
            )
            scores.extend(float(value) for value in result["colbert+sparse+dense"])
        order = np.argsort(-np.asarray(scores))[: args.top_k]
        ranked_by_query[query_id] = [
            {
                "docid": doc_id_by_index[candidates[int(index)]],
                "rank": rank,
                "score": float(scores[int(index)]),
            }
            for rank, index in enumerate(order, start=1)
        ]
        timings["rerank_pairs"] += len(pairs)
        if (query_index + 1) % 25 == 0 or query_index + 1 == len(query_ids):
            elapsed = time.monotonic() - started
            print(
                f"[hybrid-progress] rerank: {query_index + 1}/{len(query_ids)} queries "
                f"pairs={timings['rerank_pairs']} elapsed={elapsed:.1f}s",
                flush=True,
            )
    timings["rerank_elapsed_seconds"] = time.monotonic() - started
    return ranked_by_query, timings


def main() -> None:
    if np is None:
        raise SystemExit("numpy is required to run MIRACL hybrid retrieval.")
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.force_rebuild and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.top_k > args.rerank_limit:
        raise SystemExit("--top-k cannot be greater than --rerank-limit")
    query_ids, query_texts, qrels, query_source = load_queries_and_qrels(args)
    positive_count = sum(len(values) for values in qrels.values()) if qrels is not None else 0
    print(
        f"[hybrid] queries={len(query_ids)} positives={positive_count} "
        f"query_source={query_source}",
        flush=True,
    )
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

    from datasets import load_dataset

    corpus_ds = load_dataset(
        DATASET_NAME,
        f"{args.subset}-corpus",
        split="dev",
        revision=args.revision,
    )
    corpus_count = min(len(corpus_ds), args.corpus_limit) if args.corpus_limit else len(corpus_ds)
    doc_id_by_index: dict[int, str] = {}

    candidate_started = time.monotonic()
    for start in range(0, corpus_count, args.encode_chunk_size):
        end = min(start + args.encode_chunk_size, corpus_count)
        batch = corpus_ds[start:end]
        doc_ids = [str(doc_id) for doc_id in batch["_id"]]
        texts = [
            text_from_doc({"title": title, "text": text})
            for title, text in zip(batch.get("title", [""] * len(doc_ids)), batch["text"])
        ]
        for offset, doc_id in enumerate(doc_ids):
            doc_id_by_index[start + offset] = doc_id

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
            f"[hybrid-progress] candidates: {end}/{corpus_count} docs "
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
    write_json(
        output_dir / "candidate_metadata.json",
        {
            "candidate_top_k": args.candidate_top_k,
            "rerank_limit": args.rerank_limit,
            "candidate_elapsed_seconds": candidate_elapsed,
            "candidate_docs_per_second": corpus_count / max(candidate_elapsed, 1e-6),
        },
    )
    unique_candidate_indices = sorted({index for row in candidate_indices for index in row})
    doc_text_by_index: dict[int, str] = {}
    text_load_started = time.monotonic()
    for offset, index in enumerate(unique_candidate_indices, start=1):
        row = corpus_ds[int(index)]
        doc_text_by_index[int(index)] = text_from_doc(row)
        if offset % 5000 == 0 or offset == len(unique_candidate_indices):
            elapsed = time.monotonic() - text_load_started
            print(
                f"[hybrid-progress] candidate texts: {offset}/"
                f"{len(unique_candidate_indices)} elapsed={elapsed:.1f}s",
                flush=True,
            )

    ranked_by_query, rerank_timing = rerank_candidates(
        model,
        args,
        query_ids,
        query_texts,
        candidate_indices,
        doc_text_by_index,
        doc_id_by_index,
    )
    top_doc_ids_by_query = {
        query_id: [str(row["docid"]) for row in rows]
        for query_id, rows in ranked_by_query.items()
    }
    metrics = None
    if qrels is not None:
        per_query_metrics = [
            metrics_for_query(top_doc_ids_by_query[query_id], qrels[query_id], args.metric_k)
            for query_id in query_ids
        ]
        metrics = mean_metrics(per_query_metrics)
        metrics["main_score"] = metrics["ndcg_at_10"]
    run_file = output_dir / f"{args.subset}_{official_run_split(args.split)}.txt"
    run_file_validation = write_trec_run(
        run_file,
        ranked_by_query,
        query_ids,
        args.run_id,
        min(args.top_k, 100),
    )
    summary = {
        "experiment": "miracl-ar-bge-m3-hybrid",
        "model": args.model,
        "model_path": args.model_path,
        "head_checkpoint": args.head_checkpoint or None,
        "dataset": DATASET_NAME,
        "dataset_revision": args.revision,
        "query_source": query_source,
        "subset": args.subset,
        "split": args.split,
        "corpus_count": corpus_count,
        "query_count": query_count,
        "positive_count": positive_count,
        "candidate_top_k": args.candidate_top_k,
        "rerank_limit": args.rerank_limit,
        "top_k": args.top_k,
        "metric_k": args.metric_k,
        "run_id": args.run_id,
        "weights_for_different_modes": [
            args.dense_weight,
            args.sparse_weight,
            args.colbert_weight,
        ],
        "batch_size": args.batch_size,
        "rerank_batch_size": args.rerank_batch_size,
        "max_length": args.max_length,
        "max_passage_length": args.max_passage_length,
        "metrics": metrics,
        "run_file": {
            "path": str(run_file),
            **run_file_validation,
        },
        "timings": {
            "candidate_elapsed_seconds": candidate_elapsed,
            "candidate_docs_per_second": corpus_count / max(candidate_elapsed, 1e-6),
            **rerank_timing,
        },
        "note": (
            "Hybrid candidate generation uses dense and lexical sparse top-k union; "
            "reranking uses FlagEmbedding compute_score with dense+sparse+ColBERT weights."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "ranked_by_query.json", ranked_by_query)
    write_json(
        output_dir / "top10_by_query.json",
        {query_id: doc_ids[:10] for query_id, doc_ids in top_doc_ids_by_query.items()},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
