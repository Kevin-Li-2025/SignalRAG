#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # Allows lightweight CI tests to import helpers.
    np = None  # type: ignore[assignment]


DATASET_NAME = "mteb/MIRACLRetrieval"
DEFAULT_REVISION = "9c09abc13478308c27598f350e31d8f06b9b5481"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank a MIRACL Arabic candidate run with a cross-encoder reranker. "
            "Use this after a strong first-stage run such as BGE-M3 hybrid r100."
        )
    )
    parser.add_argument("--candidate-run-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument(
        "--reranker-backend",
        choices=[
            "flag",
            "flag-llm",
            "sequence-classification",
            "cross-encoder",
            "qwen3-causal",
        ],
        default="flag",
        help=(
            "Use FlagEmbedding, FlagEmbedding LLM reranker, sentence-transformers, "
            "or a direct Qwen3 causal reranker."
        ),
    )
    parser.add_argument("--subset", default="ar")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--metric-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prompt",
        default="Given an Arabic search query, retrieve relevant Arabic passages that answer the query.",
        help="Prompt/instruction for instruction-aware CrossEncoder rerankers.",
    )
    parser.add_argument("--run-id", default="candidate-rerank")
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


def trec_lines(path: Path) -> list[str]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            candidates = [
                name for name in names if name.endswith("/ar_dev.txt") or name == "ar_dev.txt"
            ]
            if not candidates:
                raise ValueError(f"{path} does not contain ar_dev.txt")
            return archive.read(candidates[0]).decode("utf-8").splitlines()
    return path.read_text(encoding="utf-8").splitlines()


def read_trec_run(path: Path, depth: int) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    for line_number, line in enumerate(trec_lines(path), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 6:
            raise ValueError(f"{path}:{line_number} expected 6 TREC columns")
        query_id, _q0, doc_id, rank, score, run_id = parts
        rows = candidates.setdefault(query_id, [])
        if len(rows) < depth:
            rows.append(
                {
                    "docid": doc_id,
                    "rank": int(rank),
                    "first_stage_score": float(score),
                    "first_stage_run_id": run_id,
                }
            )
    for query_id, rows in candidates.items():
        ranks = [row["rank"] for row in rows]
        if ranks != list(range(1, len(rows) + 1)):
            raise ValueError(f"{path}: query {query_id} has non-contiguous ranks")
    return candidates


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
    if np is None:
        keys = rows[0].keys()
        return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_relevance(
    subset: str,
    split: str,
    revision: str,
    selected_query_ids: list[str],
) -> tuple[dict[str, str], dict[str, set[str]]]:
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
    selected = set(selected_query_ids)
    query_text_by_id = {
        str(row["_id"]): str(row["text"])
        for row in queries_ds
        if str(row["_id"]) in selected
    }
    qrels: dict[str, set[str]] = {query_id: set() for query_id in selected_query_ids}
    for row in qrels_ds:
        if row.get("score", 0) <= 0:
            continue
        query_id = str(row["query-id"])
        if query_id in qrels:
            qrels[query_id].add(str(row["corpus-id"]))
    return query_text_by_id, qrels


def load_candidate_texts(
    subset: str,
    split: str,
    revision: str,
    needed_doc_ids: set[str],
) -> dict[str, str]:
    from datasets import load_dataset

    corpus_ds = load_dataset(
        DATASET_NAME,
        f"{subset}-corpus",
        split=split,
        revision=revision,
    )
    found: dict[str, str] = {}
    started = time.monotonic()
    total = len(corpus_ds)
    for index in range(total):
        row = corpus_ds[index]
        doc_id = str(row["_id"])
        if doc_id in needed_doc_ids:
            found[doc_id] = text_from_doc(row)
            if len(found) == len(needed_doc_ids):
                break
        if (index + 1) % 100000 == 0 or len(found) == len(needed_doc_ids):
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = (index + 1) / elapsed
            eta = (total - index - 1) / max(rate, 1e-6)
            print(
                f"[candidate-texts] scanned={index + 1}/{total} "
                f"found={len(found)}/{len(needed_doc_ids)} "
                f"rate={rate:.1f} docs/s eta={eta / 3600:.2f}h",
                flush=True,
            )
    missing = needed_doc_ids - found.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} candidate docs; first={sorted(missing)[:5]}")
    return found


def build_flag_reranker(args: argparse.Namespace):
    import torch
    from FlagEmbedding import FlagReranker

    devices = ["cuda:0"] if torch.cuda.is_available() else None
    kwargs: dict[str, Any] = {"use_fp16": args.use_fp16}
    if devices is not None:
        kwargs["devices"] = devices
    return FlagReranker(args.reranker_model, **kwargs)


def build_flag_llm_reranker(args: argparse.Namespace):
    import torch
    from FlagEmbedding import FlagLLMReranker

    devices = ["cuda:0"] if torch.cuda.is_available() else None
    kwargs: dict[str, Any] = {
        "use_fp16": args.use_fp16,
        "trust_remote_code": args.trust_remote_code,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    if devices is not None:
        kwargs["devices"] = devices
    return FlagLLMReranker(args.reranker_model, **kwargs)


def build_cross_encoder(args: argparse.Namespace):
    import torch
    from sentence_transformers import CrossEncoder

    model_kwargs = {}
    if args.use_fp16 and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    return CrossEncoder(
        args.reranker_model,
        model_kwargs=model_kwargs,
        trust_remote_code=args.trust_remote_code,
        max_length=args.max_length,
        prompts={"arabic_retrieval": args.prompt},
        default_prompt_name="arabic_retrieval",
    )


def build_sequence_classifier(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.use_fp16 and device == "cuda" else None
    tokenizer = AutoTokenizer.from_pretrained(
        args.reranker_model,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForSequenceClassification.from_pretrained(
        args.reranker_model,
        **model_kwargs,
    )
    model.to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


def build_qwen3_causal_reranker(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.use_fp16 and device == "cuda" else None
    tokenizer = AutoTokenizer.from_pretrained(
        args.reranker_model,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.reranker_model, **model_kwargs)
    model.to(device)
    model.eval()
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and "
        'the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "token_true_id": tokenizer.convert_tokens_to_ids("yes"),
        "token_false_id": tokenizer.convert_tokens_to_ids("no"),
        "prefix_tokens": tokenizer.encode(prefix, add_special_tokens=False),
        "suffix_tokens": tokenizer.encode(suffix, add_special_tokens=False),
    }


def format_qwen3_instruction(instruction: str | None, query: str, doc: str) -> str:
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query."
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def score_pairs(model: object, args: argparse.Namespace, pairs: list[tuple[str, str]]) -> list[float]:
    if args.reranker_backend == "qwen3-causal":
        import torch

        tokenizer = model["tokenizer"]
        classifier = model["model"]
        device = model["device"]
        prefix_tokens = model["prefix_tokens"]
        suffix_tokens = model["suffix_tokens"]
        content_max_length = args.max_length - len(prefix_tokens) - len(suffix_tokens)
        if content_max_length <= 0:
            raise ValueError(
                f"max_length={args.max_length} is too short for Qwen3 prompt wrapper"
            )
        scores: list[float] = []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            formatted = [
                format_qwen3_instruction(args.prompt, query, doc)
                for query, doc in batch
            ]
            encoded = tokenizer(
                formatted,
                padding=False,
                truncation="longest_first",
                max_length=content_max_length,
                return_attention_mask=False,
            )
            encoded["input_ids"] = [
                prefix_tokens + input_ids + suffix_tokens
                for input_ids in encoded["input_ids"]
            ]
            encoded = tokenizer.pad(
                encoded,
                padding=True,
                return_tensors="pt",
                max_length=args.max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = classifier(**encoded).logits[:, -1, :].float()
            true_logits = logits[:, model["token_true_id"]]
            false_logits = logits[:, model["token_false_id"]]
            batch_scores = torch.stack([false_logits, true_logits], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)[:, 1]
            scores.extend(float(value) for value in batch_scores.cpu())
        return scores

    if args.reranker_backend == "sequence-classification":
        import torch

        tokenizer = model["tokenizer"]
        classifier = model["model"]
        device = model["device"]
        scores: list[float] = []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            encoded = tokenizer(
                [query for query, _doc in batch],
                [doc for _query, doc in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = classifier(**encoded).logits.float()
            if logits.ndim == 1 or logits.shape[-1] == 1:
                batch_scores = logits.reshape(-1)
            else:
                batch_scores = logits[:, -1] - logits[:, 0]
            scores.extend(float(value) for value in batch_scores.cpu())
        return scores

    scores: list[float] = []
    if args.reranker_backend == "flag-llm":
        batch = [[query, doc] for query, doc in pairs]
        result = model.compute_score(batch, batch_size=args.batch_size, max_length=args.max_length)
        if isinstance(result, float):
            return [float(result)]
        return [float(value) for value in result]

    if args.reranker_backend == "flag":
        for start in range(0, len(pairs), args.batch_size):
            batch = [[query, doc] for query, doc in pairs[start : start + args.batch_size]]
            result = model.compute_score(batch, batch_size=args.batch_size, max_length=args.max_length)
            if isinstance(result, float):
                scores.append(float(result))
            else:
                scores.extend(float(value) for value in result)
        return scores

    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        result = model.predict(batch, batch_size=args.batch_size)
        scores.extend(float(value) for value in result)
    return scores


def write_trec_run(
    path: Path,
    ranked_by_query: dict[str, list[dict[str, Any]]],
    query_ids: list[str],
    run_id: str,
    depth: int,
) -> dict[str, int]:
    line_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for query_id in query_ids:
            rows = ranked_by_query[query_id]
            if len(rows) < depth:
                raise ValueError(f"Query {query_id} has only {len(rows)} hits")
            for expected_rank, row in enumerate(rows[:depth], start=1):
                handle.write(
                    f"{query_id} Q0 {row['docid']} {expected_rank} "
                    f"{float(row['score']):.10f} {run_id}\n"
                )
                line_count += 1
    return {"queries": len(query_ids), "depth": depth, "lines": line_count}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_run = read_trec_run(Path(args.candidate_run_file), args.candidate_depth)
    query_ids = sorted(candidate_run, key=sort_key)
    if args.query_limit:
        query_ids = query_ids[: args.query_limit]
    candidate_run = {query_id: candidate_run[query_id][: args.candidate_depth] for query_id in query_ids}
    query_text_by_id, qrels = load_relevance(args.subset, args.split, args.revision, query_ids)
    query_ids = [query_id for query_id in query_ids if query_id in query_text_by_id and qrels[query_id]]
    candidate_run = {query_id: candidate_run[query_id] for query_id in query_ids}
    needed_doc_ids = {row["docid"] for rows in candidate_run.values() for row in rows}

    print(
        f"[rerank] queries={len(query_ids)} candidate_docs={len(needed_doc_ids)} "
        f"pairs={sum(len(rows) for rows in candidate_run.values())}",
        flush=True,
    )
    doc_text_by_id = load_candidate_texts(args.subset, args.split, args.revision, needed_doc_ids)
    print(f"[rerank] loaded candidate texts={len(doc_text_by_id)}", flush=True)
    print(
        f"[rerank] loading reranker model={args.reranker_model} backend={args.reranker_backend}",
        flush=True,
    )
    if args.reranker_backend == "flag":
        reranker = build_flag_reranker(args)
    elif args.reranker_backend == "flag-llm":
        reranker = build_flag_llm_reranker(args)
    elif args.reranker_backend == "sequence-classification":
        reranker = build_sequence_classifier(args)
    elif args.reranker_backend == "qwen3-causal":
        reranker = build_qwen3_causal_reranker(args)
    else:
        reranker = build_cross_encoder(args)
    print("[rerank] reranker loaded; scoring pairs", flush=True)

    ranked_by_query: dict[str, list[dict[str, Any]]] = {}
    all_first_stage_metrics: list[dict[str, float]] = []
    all_rerank_metrics: list[dict[str, float]] = []
    rerank_started = time.monotonic()
    scored_pairs = 0
    for offset, query_id in enumerate(query_ids, start=1):
        query = query_text_by_id[query_id]
        candidates = candidate_run[query_id]
        pairs = [(query, doc_text_by_id[row["docid"]]) for row in candidates]
        scores = score_pairs(reranker, args, pairs)
        scored_pairs += len(pairs)
        ordered = sorted(
            [
                {
                    "docid": row["docid"],
                    "rank": rank,
                    "score": score,
                    "first_stage_score": row["first_stage_score"],
                }
                for rank, (row, score) in enumerate(zip(candidates, scores, strict=True), start=1)
            ],
            key=lambda row: row["score"],
            reverse=True,
        )
        ranked_by_query[query_id] = [
            {**row, "rank": rank}
            for rank, row in enumerate(ordered[: args.top_k], start=1)
        ]
        first_stage_doc_ids = [row["docid"] for row in candidates[: args.top_k]]
        reranked_doc_ids = [row["docid"] for row in ranked_by_query[query_id]]
        all_first_stage_metrics.append(metrics_for_query(first_stage_doc_ids, qrels[query_id], args.metric_k))
        all_rerank_metrics.append(metrics_for_query(reranked_doc_ids, qrels[query_id], args.metric_k))
        if offset % 25 == 0 or offset == len(query_ids):
            elapsed = max(time.monotonic() - rerank_started, 1e-6)
            rate = scored_pairs / elapsed
            remaining = (len(query_ids) - offset) * args.candidate_depth
            eta = remaining / max(rate, 1e-6)
            print(
                f"[rerank-progress] queries={offset}/{len(query_ids)} "
                f"pairs={scored_pairs} rate={rate:.2f} pairs/s eta={eta / 3600:.2f}h",
                flush=True,
            )

    first_stage_metrics = mean_metrics(all_first_stage_metrics)
    first_stage_metrics["main_score"] = first_stage_metrics["ndcg_at_10"]
    rerank_metrics = mean_metrics(all_rerank_metrics)
    rerank_metrics["main_score"] = rerank_metrics["ndcg_at_10"]
    run_file = output_dir / f"{args.subset}_{args.split}_{args.run_id}.txt"
    run_file_validation = write_trec_run(
        run_file,
        ranked_by_query,
        query_ids,
        args.run_id,
        min(args.top_k, args.candidate_depth),
    )
    summary = {
        "experiment": "miracl-ar-candidate-rerank",
        "candidate_run_file": str(args.candidate_run_file),
        "reranker_model": args.reranker_model,
        "reranker_backend": args.reranker_backend,
        "dataset": DATASET_NAME,
        "dataset_revision": args.revision,
        "subset": args.subset,
        "split": args.split,
        "query_count": len(query_ids),
        "positive_count": sum(len(values) for values in qrels.values()),
        "candidate_depth": args.candidate_depth,
        "top_k": args.top_k,
        "metric_k": args.metric_k,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "prompt": args.prompt
        if args.reranker_backend in {"cross-encoder", "qwen3-causal"}
        else None,
        "metrics": rerank_metrics,
        "first_stage_metrics_on_same_queries": first_stage_metrics,
        "delta_main_score_vs_first_stage_same_queries": (
            rerank_metrics["main_score"] - first_stage_metrics["main_score"]
        ),
        "run_file": {
            "path": str(run_file),
            **run_file_validation,
        },
        "timings": {
            "rerank_elapsed_seconds": time.monotonic() - rerank_started,
            "rerank_pairs": scored_pairs,
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "ranked_by_query.json", ranked_by_query)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
