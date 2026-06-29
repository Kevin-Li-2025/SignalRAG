#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic MIRACL Arabic retrieval sample."
    )
    parser.add_argument("--model", required=True, help="Model ID from the MTEB registry.")
    parser.add_argument("--output-dir", required=True, help="Directory for derived JSON outputs.")
    parser.add_argument("--corpus-size", type=int, default=10_000)
    parser.add_argument("--query-count", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=62)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument(
        "--progress-chunk-size",
        type=int,
        default=0,
        help="Texts per progress log chunk; 0 uses batch_size * 8.",
    )
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--revision", default="9c09abc13478308c27598f350e31d8f06b9b5481")
    parser.add_argument("--negative-pool-size", type=int, default=20_000)
    parser.add_argument(
        "--sample-mode",
        choices=["positive-plus-negatives", "head-corpus"],
        default="positive-plus-negatives",
        help=(
            "positive-plus-negatives scans for positives then fills negatives; "
            "head-corpus uses dev[:corpus_size] and keeps queries with qrels inside it."
        ),
    )
    return parser.parse_args()


def sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):020d}")
    except ValueError:
        return (1, value)


def text_from_doc(doc: dict[str, str]) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return text or title


def normalize(embeddings: object) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def encode_with_progress(
    model: object,
    texts: list[str],
    batch_size: int,
    progress_chunk_size: int,
    label: str,
    method_name: str = "encode",
) -> tuple[np.ndarray, dict[str, float]]:
    method = getattr(model, method_name, None) or getattr(model, "encode")
    started = time.monotonic()
    chunk_size = progress_chunk_size or max(batch_size, batch_size * 8)
    print(
        f"[sample-progress] {label}: start {len(texts)} texts "
        f"batch_size={batch_size} chunk_size={chunk_size}",
        flush=True,
    )
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), chunk_size):
        end = min(start + chunk_size, len(texts))
        chunk_embeddings = method(texts[start:end], batch_size=batch_size)
        chunks.append(np.asarray(chunk_embeddings))
        elapsed = max(time.monotonic() - started, 1e-6)
        print(
            f"[sample-progress] {label}: {end}/{len(texts)} texts "
            f"in {elapsed:.1f}s ({end / elapsed:.2f} texts/s)",
            flush=True,
        )
    elapsed = max(time.monotonic() - started, 1e-6)
    embeddings = np.concatenate(chunks, axis=0)
    print(
        f"[sample-progress] {label}: done {len(texts)} texts in "
        f"{elapsed:.1f}s ({len(texts) / elapsed:.2f} texts/s)",
        flush=True,
    )
    return normalize(embeddings), {
        "elapsed_seconds": elapsed,
        "texts_per_second": len(texts) / elapsed,
    }


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


def main() -> None:
    args = parse_args()

    from datasets import load_dataset
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[sample] loading MIRACLRetrieval ar queries/qrels", flush=True)
    queries_ds = load_dataset(
        "mteb/MIRACLRetrieval",
        "ar-queries",
        split="dev",
        revision=args.revision,
    )
    qrels_ds = load_dataset(
        "mteb/MIRACLRetrieval",
        "ar-qrels",
        split="dev",
        revision=args.revision,
    )

    rng = random.Random(args.seed)
    selected_qrels: dict[str, set[str]] = {}
    positive_doc_ids: set[str] = set()
    doc_text_by_id: dict[str, str] = {}

    if args.sample_mode == "head-corpus":
        print(f"[sample] loading head corpus dev[:{args.corpus_size}]", flush=True)
        corpus_ds = load_dataset(
            "mteb/MIRACLRetrieval",
            "ar-corpus",
            split=f"dev[:{args.corpus_size}]",
            revision=args.revision,
        )
        sampled_doc_ids = [str(doc["_id"]) for doc in corpus_ds]
        doc_text_by_id = {str(doc["_id"]): text_from_doc(doc) for doc in corpus_ds}
        sampled_doc_set = set(sampled_doc_ids)
        for row in qrels_ds:
            if row.get("score", 0) <= 0:
                continue
            corpus_id = str(row["corpus-id"])
            if corpus_id not in sampled_doc_set:
                continue
            query_id = str(row["query-id"])
            selected_qrels.setdefault(query_id, set()).add(corpus_id)
            positive_doc_ids.add(corpus_id)
        query_ids_for_limit = sorted(selected_qrels, key=sort_key)[: args.query_count]
        selected_qrels = {query_id: selected_qrels[query_id] for query_id in query_ids_for_limit}
        positive_doc_ids = {
            doc_id for doc_ids in selected_qrels.values() for doc_id in doc_ids
        }
    else:
        all_query_ids = sorted(
            {str(row["query-id"]) for row in qrels_ds},
            key=sort_key,
        )
        selected_query_ids = set(all_query_ids[: args.query_count])
        for row in qrels_ds:
            query_id = str(row["query-id"])
            if query_id not in selected_query_ids or row.get("score", 0) <= 0:
                continue
            corpus_id = str(row["corpus-id"])
            selected_qrels.setdefault(query_id, set()).add(corpus_id)
            positive_doc_ids.add(corpus_id)

        if len(positive_doc_ids) > args.corpus_size:
            raise SystemExit(
                f"{len(positive_doc_ids)} positive docs exceed corpus size {args.corpus_size}; "
                "reduce --query-count or increase --corpus-size."
            )

        print("[sample] loading negative pool from head corpus slice", flush=True)
        negative_pool_ds = load_dataset(
            "mteb/MIRACLRetrieval",
            "ar-corpus",
            split=f"dev[:{max(args.negative_pool_size, args.corpus_size)}]",
            revision=args.revision,
        )
        negative_candidates: list[str] = []
        for doc in negative_pool_ds:
            doc_id = str(doc["_id"])
            if doc_id not in positive_doc_ids:
                negative_candidates.append(doc_id)
                doc_text_by_id[doc_id] = text_from_doc(doc)
        print(
            f"[sample] loaded negative pool size={len(negative_candidates)}",
            flush=True,
        )
        print("[sample] filtering positive corpus docs", flush=True)
        positive_ds = load_dataset(
            "mteb/MIRACLRetrieval",
            "ar-corpus",
            split="dev",
            revision=args.revision,
        ).filter(
            lambda row: str(row["_id"]) in positive_doc_ids,
            num_proc=4,
        )
        for index, doc in enumerate(positive_ds):
            doc_id = str(doc["_id"])
            doc_text_by_id[doc_id] = text_from_doc(doc)
            if index and index % 100 == 0:
                print(f"[sample-progress] positives loaded: {index}", flush=True)

        missing_positive_ids = positive_doc_ids.difference(doc_text_by_id)
        if missing_positive_ids:
            raise SystemExit(f"Missing positive corpus docs: {sorted(missing_positive_ids)[:5]}")

        rng.shuffle(negative_candidates)
        needed_negatives = args.corpus_size - len(positive_doc_ids)
        sampled_doc_ids = sorted(positive_doc_ids) + negative_candidates[:needed_negatives]
        rng.shuffle(sampled_doc_ids)
        sampled_doc_set = set(sampled_doc_ids)

    print(
        f"[sample] selected {len(selected_qrels)} queries with "
        f"{len(positive_doc_ids)} positive docs",
        flush=True,
    )

    query_text_by_id = {
        str(row["_id"]): str(row["text"])
        for row in queries_ds
        if str(row["_id"]) in selected_qrels
    }
    query_ids = [
        query_id
        for query_id in sorted(selected_qrels, key=sort_key)
        if query_id in query_text_by_id and selected_qrels[query_id].intersection(sampled_doc_set)
    ]

    corpus_texts = [doc_text_by_id[doc_id] for doc_id in sampled_doc_ids]
    query_texts = [query_text_by_id[query_id] for query_id in query_ids]

    print(
        f"[sample] final sample: corpus={len(corpus_texts)} queries={len(query_texts)}",
        flush=True,
    )
    if len(corpus_texts) != args.corpus_size:
        raise SystemExit(f"Expected {args.corpus_size} corpus docs, got {len(corpus_texts)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    model = SentenceTransformer(
        args.model,
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )
    if hasattr(model, "model") and hasattr(model.model, "max_seq_length"):
        model.model.max_seq_length = args.max_seq_length
    elif hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_seq_length

    query_embeddings, query_timing = encode_with_progress(
        model,
        query_texts,
        args.batch_size,
        args.progress_chunk_size,
        "queries",
        method_name="encode_queries",
    )
    corpus_embeddings, corpus_timing = encode_with_progress(
        model,
        corpus_texts,
        args.batch_size,
        args.progress_chunk_size,
        "corpus",
        method_name="encode_corpus",
    )

    doc_ids = np.asarray(sampled_doc_ids)
    per_query_metrics: list[dict[str, float]] = []
    topk_by_query: dict[str, list[str]] = {}
    started = time.monotonic()
    scores = query_embeddings @ corpus_embeddings.T
    top_indices = np.argpartition(-scores, kth=9, axis=1)[:, :10]
    for row_index, query_id in enumerate(query_ids):
        candidates = top_indices[row_index]
        ordered = candidates[np.argsort(-scores[row_index, candidates])]
        ranked_doc_ids = [str(doc_ids[index]) for index in ordered]
        topk_by_query[query_id] = ranked_doc_ids
        per_query_metrics.append(
            metrics_for_query(ranked_doc_ids, selected_qrels[query_id].intersection(sampled_doc_set))
        )
    eval_elapsed = time.monotonic() - started
    metrics = mean_metrics(per_query_metrics)
    metrics["main_score"] = metrics["ndcg_at_10"]

    summary = {
        "experiment": "miracl-ar-10k-sample",
        "model": args.model,
        "dataset": "mteb/MIRACLRetrieval",
        "dataset_revision": args.revision,
        "subset": "ar",
        "split": "dev",
        "seed": args.seed,
        "corpus_size": len(corpus_texts),
        "query_count": len(query_texts),
        "positive_doc_count": len(positive_doc_ids),
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "timings": {
            "query_encode": query_timing,
            "corpus_encode": corpus_timing,
        },
        "metrics": metrics,
        "score_elapsed_seconds": eval_elapsed,
        "note": "Sample result only; not comparable to full MIRACLRetrieval ar scores.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "top10_by_query.json").write_text(
        json.dumps(topk_by_query, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
