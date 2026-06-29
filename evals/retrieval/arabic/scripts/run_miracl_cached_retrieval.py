#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


DATASET_NAME = "mteb/MIRACLRetrieval"
DEFAULT_REVISION = "9c09abc13478308c27598f350e31d8f06b9b5481"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MIRACL Arabic retrieval with a resumable corpus embedding cache "
            "and chunked top-k scoring."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset", default="ar")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=62)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--encode-chunk-size", type=int, default=1000)
    parser.add_argument("--score-corpus-chunk-size", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--corpus-limit", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
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


def normalize(embeddings: object) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


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


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def load_relevance(
    subset: str,
    split: str,
    revision: str,
    query_limit: int,
) -> tuple[list[str], list[str], dict[str, set[str]]]:
    from datasets import load_dataset

    print(f"[cache] loading {subset} queries/qrels split={split}", flush=True)
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
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        qrels.setdefault(query_id, set()).add(corpus_id)

    ordered_query_ids = sorted(qrels, key=sort_key)
    if query_limit:
        ordered_query_ids = ordered_query_ids[:query_limit]
        qrels = {query_id: qrels[query_id] for query_id in ordered_query_ids}

    query_text_by_id = {
        str(row["_id"]): str(row["text"])
        for row in queries_ds
        if str(row["_id"]) in qrels
    }
    query_ids = [query_id for query_id in ordered_query_ids if query_id in query_text_by_id]
    query_texts = [query_text_by_id[query_id] for query_id in query_ids]
    qrels = {query_id: qrels[query_id] for query_id in query_ids}
    print(
        f"[cache] selected queries={len(query_ids)} positives="
        f"{sum(len(values) for values in qrels.values())}",
        flush=True,
    )
    return query_ids, query_texts, qrels


def load_model(model_name: str, max_seq_length: int):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )
    if hasattr(model, "model") and hasattr(model.model, "max_seq_length"):
        model.model.max_seq_length = max_seq_length
    elif hasattr(model, "max_seq_length"):
        model.max_seq_length = max_seq_length
    return model, device


def encode_texts(
    model: object,
    texts: list[str],
    batch_size: int,
    method_name: str,
    label: str,
) -> tuple[np.ndarray, dict[str, float]]:
    method = getattr(model, method_name, None) or getattr(model, "encode")
    started = time.monotonic()
    print(
        f"[cache-progress] {label}: start {len(texts)} texts batch_size={batch_size}",
        flush=True,
    )
    embeddings = normalize(method(texts, batch_size=batch_size))
    elapsed = max(time.monotonic() - started, 1e-6)
    print(
        f"[cache-progress] {label}: done {len(texts)} texts in "
        f"{elapsed:.1f}s ({len(texts) / elapsed:.2f} texts/s)",
        flush=True,
    )
    return embeddings, {
        "elapsed_seconds": elapsed,
        "texts_per_second": len(texts) / elapsed,
    }


def encode_corpus_cache(
    args: argparse.Namespace,
    output_dir: Path,
    model: object,
    embedding_dim: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    metadata_path = output_dir / "cache_metadata.json"
    doc_ids_path = output_dir / "doc_ids.txt"
    embeddings_path = output_dir / "corpus_embeddings.float16.memmap"

    corpus_ds = load_dataset(
        DATASET_NAME,
        f"{args.subset}-corpus",
        split=args.split,
        revision=args.revision,
    )
    corpus_count = min(len(corpus_ds), args.corpus_limit) if args.corpus_limit else len(corpus_ds)

    expected = {
        "model": args.model,
        "dataset": DATASET_NAME,
        "dataset_revision": args.revision,
        "subset": args.subset,
        "split": args.split,
        "corpus_count": corpus_count,
        "embedding_dim": embedding_dim,
        "storage_dtype": "float16",
        "max_seq_length": args.max_seq_length,
    }

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise SystemExit(
                    f"Existing cache metadata mismatch for {key}: "
                    f"{metadata.get(key)!r} != {value!r}. Use --force-rebuild."
                )
    else:
        metadata = {
            **expected,
            "completed_docs": 0,
            "timings": {
                "corpus_encode_elapsed_seconds": 0.0,
            },
        }
        write_json(metadata_path, metadata)

    completed = int(metadata.get("completed_docs", 0))
    existing_doc_ids = line_count(doc_ids_path)
    if existing_doc_ids != completed:
        raise SystemExit(
            f"doc_ids count {existing_doc_ids} does not match completed_docs {completed}; "
            "use --force-rebuild to reset the cache."
        )

    embeddings = np.memmap(
        embeddings_path,
        mode="r+" if embeddings_path.exists() else "w+",
        dtype=np.float16,
        shape=(corpus_count, embedding_dim),
    )
    method = getattr(model, "encode_corpus", None) or getattr(model, "encode")
    started = time.monotonic()
    total_started = started - float(metadata.get("timings", {}).get("corpus_encode_elapsed_seconds", 0.0))
    print(
        f"[cache] corpus cache target docs={corpus_count} dim={embedding_dim} "
        f"completed={completed}",
        flush=True,
    )

    with doc_ids_path.open("a", encoding="utf-8") as doc_ids_file:
        for start in range(completed, corpus_count, args.encode_chunk_size):
            end = min(start + args.encode_chunk_size, corpus_count)
            batch = corpus_ds[start:end]
            doc_ids = [str(doc_id) for doc_id in batch["_id"]]
            texts = [
                text_from_doc({"title": title, "text": text})
                for title, text in zip(batch.get("title", [""] * len(doc_ids)), batch["text"])
            ]
            chunk_started = time.monotonic()
            chunk_embeddings = normalize(method(texts, batch_size=args.batch_size)).astype(np.float16)
            embeddings[start:end] = chunk_embeddings
            embeddings.flush()
            for doc_id in doc_ids:
                doc_ids_file.write(doc_id + "\n")
            doc_ids_file.flush()

            metadata["completed_docs"] = end
            elapsed_total = time.monotonic() - total_started
            metadata["timings"]["corpus_encode_elapsed_seconds"] = elapsed_total
            write_json(metadata_path, metadata)

            docs_per_second = end / max(elapsed_total, 1e-6)
            remaining = corpus_count - end
            eta_seconds = remaining / max(docs_per_second, 1e-6)
            chunk_elapsed = max(time.monotonic() - chunk_started, 1e-6)
            print(
                f"[cache-progress] corpus: {end}/{corpus_count} docs "
                f"chunk={chunk_elapsed:.1f}s total={elapsed_total:.1f}s "
                f"rate={docs_per_second:.2f} docs/s eta={eta_seconds / 3600:.2f}h",
                flush=True,
            )

    metadata["completed_docs"] = corpus_count
    metadata["cache_complete"] = True
    metadata["timings"]["corpus_encode_elapsed_seconds"] = time.monotonic() - total_started
    metadata["timings"]["corpus_encode_texts_per_second"] = corpus_count / max(
        metadata["timings"]["corpus_encode_elapsed_seconds"], 1e-6
    )
    write_json(metadata_path, metadata)
    return metadata


def score_cached_embeddings(
    args: argparse.Namespace,
    output_dir: Path,
    query_ids: list[str],
    query_embeddings: np.ndarray,
    qrels: dict[str, set[str]],
) -> dict[str, Any]:
    import torch

    metadata = json.loads((output_dir / "cache_metadata.json").read_text(encoding="utf-8"))
    corpus_count = int(metadata["corpus_count"])
    embedding_dim = int(metadata["embedding_dim"])
    if int(metadata.get("completed_docs", 0)) != corpus_count:
        raise SystemExit("Corpus cache is incomplete; cannot score.")

    doc_ids = (output_dir / "doc_ids.txt").read_text(encoding="utf-8").splitlines()
    if len(doc_ids) != corpus_count:
        raise SystemExit(f"Expected {corpus_count} doc ids, found {len(doc_ids)}")

    corpus_embeddings = np.memmap(
        output_dir / "corpus_embeddings.float16.memmap",
        mode="r",
        dtype=np.float16,
        shape=(corpus_count, embedding_dim),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_tensor = torch.from_numpy(query_embeddings.astype(np.float32)).to(device)
    top_scores = torch.full(
        (len(query_ids), args.top_k),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    top_indices = torch.full(
        (len(query_ids), args.top_k),
        -1,
        dtype=torch.long,
        device=device,
    )

    print(
        f"[score] start queries={len(query_ids)} corpus={corpus_count} "
        f"chunk={args.score_corpus_chunk_size}",
        flush=True,
    )
    started = time.monotonic()
    for start in range(0, corpus_count, args.score_corpus_chunk_size):
        end = min(start + args.score_corpus_chunk_size, corpus_count)
        corpus_chunk = np.asarray(corpus_embeddings[start:end], dtype=np.float32)
        corpus_tensor = torch.from_numpy(corpus_chunk).to(device)
        scores = query_tensor @ corpus_tensor.T
        chunk_k = min(args.top_k, end - start)
        chunk_scores, chunk_indices = torch.topk(scores, k=chunk_k, dim=1)
        chunk_indices = chunk_indices + start

        merged_scores = torch.cat([top_scores, chunk_scores], dim=1)
        merged_indices = torch.cat([top_indices, chunk_indices], dim=1)
        top_scores, order = torch.topk(merged_scores, k=args.top_k, dim=1)
        top_indices = torch.gather(merged_indices, 1, order)

        elapsed = max(time.monotonic() - started, 1e-6)
        docs_per_second = end / elapsed
        eta_seconds = (corpus_count - end) / max(docs_per_second, 1e-6)
        print(
            f"[score-progress] corpus: {end}/{corpus_count} docs "
            f"elapsed={elapsed:.1f}s rate={docs_per_second:.2f} docs/s "
            f"eta={eta_seconds / 3600:.2f}h",
            flush=True,
        )
        del corpus_tensor, scores, chunk_scores, chunk_indices, merged_scores, merged_indices, order

    score_elapsed = time.monotonic() - started
    indices = top_indices.cpu().numpy()
    topk_by_query: dict[str, list[str]] = {}
    per_query_metrics: list[dict[str, float]] = []
    for row_index, query_id in enumerate(query_ids):
        ranked_doc_ids = [doc_ids[index] for index in indices[row_index] if index >= 0]
        topk_by_query[query_id] = ranked_doc_ids
        per_query_metrics.append(metrics_for_query(ranked_doc_ids, qrels[query_id], args.top_k))

    metrics = mean_metrics(per_query_metrics)
    metrics["main_score"] = metrics["ndcg_at_10"]
    result = {
        "experiment": "miracl-ar-cached-retrieval",
        "model": args.model,
        "dataset": DATASET_NAME,
        "dataset_revision": args.revision,
        "subset": args.subset,
        "split": args.split,
        "corpus_count": corpus_count,
        "query_count": len(query_ids),
        "positive_count": sum(len(values) for values in qrels.values()),
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "top_k": args.top_k,
        "metrics": metrics,
        "timings": {
            **metadata.get("timings", {}),
            "score_elapsed_seconds": score_elapsed,
        },
        "cache_files": {
            "metadata": "cache_metadata.json",
            "doc_ids": "doc_ids.txt",
            "corpus_embeddings": "corpus_embeddings.float16.memmap",
            "query_embeddings": "query_embeddings.float16.npy",
        },
    }
    write_json(output_dir / "summary.json", result)
    write_json(output_dir / "top10_by_query.json", topk_by_query)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.encode_only and args.score_only:
        raise SystemExit("Use at most one of --encode-only or --score-only.")

    output_dir = Path(args.output_dir)
    if args.force_rebuild and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_ids, query_texts, qrels = load_relevance(
        args.subset,
        args.split,
        args.revision,
        args.query_limit,
    )
    query_ids_path = output_dir / "query_ids.txt"
    query_embeddings_path = output_dir / "query_embeddings.float16.npy"
    query_timing_path = output_dir / "query_timing.json"

    model = None
    if args.score_only:
        if not query_embeddings_path.exists():
            raise SystemExit("Missing query embedding cache; cannot use --score-only.")
        query_embeddings = np.load(query_embeddings_path).astype(np.float32)
    else:
        model, _device = load_model(args.model, args.max_seq_length)
        query_embeddings, query_timing = encode_texts(
            model,
            query_texts,
            args.batch_size,
            "encode_queries",
            "queries",
        )
        np.save(query_embeddings_path, query_embeddings.astype(np.float16))
        query_ids_path.write_text("\n".join(query_ids) + "\n", encoding="utf-8")
        write_json(query_timing_path, query_timing)

    if not args.score_only:
        metadata = encode_corpus_cache(
            args,
            output_dir,
            model,
            query_embeddings.shape[1],
        )
        if query_timing_path.exists():
            metadata.setdefault("timings", {})["query_encode"] = json.loads(
                query_timing_path.read_text(encoding="utf-8")
            )
            write_json(output_dir / "cache_metadata.json", metadata)

    if args.encode_only:
        print("[cache] encode-only complete; skipping scoring.", flush=True)
        return

    if model is not None:
        del model
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.score_only:
        query_ids = query_ids_path.read_text(encoding="utf-8").splitlines()
        qrels = {query_id: qrels[query_id] for query_id in query_ids}
    score_cached_embeddings(args, output_dir, query_ids, query_embeddings, qrels)


if __name__ == "__main__":
    main()
