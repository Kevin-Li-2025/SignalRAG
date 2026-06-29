from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from coreb_sota.data import doc_text, load_task, query_text, task_names
from coreb_sota.metrics import evaluate_run, weighted_average
from coreb_sota.qwen3 import TASK_INSTRUCTIONS


DOC_INSTRUCTIONS = {
    "text2code": "Represent this code solution for retrieval by natural language programming tasks: ",
    "code2code": "Represent this code solution for retrieval by semantically equivalent code snippets: ",
    "code2text": "Represent this problem statement for retrieval by code snippets: ",
}

QUERY_INSTRUCTIONS = {
    "text2code": "Represent this natural language programming task for retrieving code solutions: ",
    "code2code": "Represent this code snippet for retrieving semantically equivalent code solutions: ",
    "code2text": "Represent this code snippet for retrieving matching problem statements: ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense CoREB retrieval with an HF embedding model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/coreb"))
    parser.add_argument("--split", default="release_v2603")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--instruction-mode", choices=["none", "query", "query_doc", "coreb_task"], default="none")
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/embedding_cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("reports/dense_retrieval_release_v2603.json"))
    return parser.parse_args()


def model_tag(model_name: str) -> str:
    return model_name.strip("/").replace("/", "--").replace(":", "_")


def text_fingerprint(texts: list[str]) -> str:
    h = hashlib.sha1()
    for text in texts:
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def prepare_text(task_name: str, role: str, text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "coreb_task":
        return TASK_INSTRUCTIONS[task_name] + "\n" + text
    if mode == "query" and role == "query":
        return QUERY_INSTRUCTIONS[task_name] + text
    if mode == "query_doc":
        prefix = QUERY_INSTRUCTIONS[task_name] if role == "query" else DOC_INSTRUCTIONS[task_name]
        return prefix + text
    return text


def load_encoder(args: argparse.Namespace):
    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    dtype: Any
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        dtype = "auto"

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if getattr(config, "tokenizer_name_or_path", None) and Path(args.model).exists():
        config.tokenizer_name_or_path = args.model

    model_class = get_class_from_dynamic_module(
        "modeling_c2llm.C2LLMForEmbedding",
        args.model,
        local_files_only=args.local_files_only,
    )
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}

    model = model_class.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    )
    if args.device != "auto":
        model.to(args.device)
    model.eval()
    return model


def encode_texts(model, texts: list[str], args: argparse.Namespace, cache_key: str) -> np.ndarray:
    cache_file = args.cache_dir / f"{cache_key}.npy"
    if not args.no_cache and cache_file.exists():
        return np.load(cache_file)
    if not hasattr(model, "encode"):
        raise TypeError(f"{args.model} does not expose an encode() method; use a model with pooled embeddings.")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        convert_to_tensor=False,
        show_progress_bar=True,
        max_seq_length=args.max_length,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    if not args.no_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, embeddings)
    return embeddings


def topk_scores(query_embeddings: np.ndarray, doc_embeddings: np.ndarray, doc_ids: list[str], k: int) -> dict[int, dict[str, float]]:
    k = min(k, len(doc_ids))
    scores = query_embeddings @ doc_embeddings.T
    runs: dict[int, dict[str, float]] = {}
    for row_idx in tqdm(range(scores.shape[0]), desc="topk"):
        row = scores[row_idx]
        if k == len(doc_ids):
            candidate_idx = np.arange(len(doc_ids))
        else:
            candidate_idx = np.argpartition(-row, k - 1)[:k]
        ranked_idx = candidate_idx[np.argsort(-row[candidate_idx])]
        runs[row_idx] = {doc_ids[idx]: float(row[idx]) for idx in ranked_idx}
    return runs


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    model = load_encoder(args)
    task_metrics: dict[str, dict[str, Any]] = {}
    output_runs: dict[str, dict[str, dict[str, float]]] = {}
    embedding_files: dict[str, dict[str, str]] = {}

    for task_name in task_names(args.tasks):
        task = load_task(args.data_dir, args.split, task_name)
        query_ids = list(task.queries)
        doc_ids = list(task.corpus)
        queries = [prepare_text(task_name, "query", query_text(task, qid), args.instruction_mode) for qid in query_ids]
        docs = [prepare_text(task_name, "doc", doc_text(task, did), args.instruction_mode) for did in doc_ids]
        base_key = (
            f"{model_tag(args.model)}_{args.split}_{task_name}_{args.instruction_mode}_"
            f"len{args.max_length}_{text_fingerprint(queries + docs)}"
        )
        query_key = base_key + "_queries"
        doc_key = base_key + "_docs"
        query_embeddings = encode_texts(model, queries, args, query_key)
        doc_embeddings = encode_texts(model, docs, args, doc_key)
        index_runs = topk_scores(query_embeddings, doc_embeddings, doc_ids, args.top_k)
        task_run = {qid: index_runs[idx] for idx, qid in enumerate(query_ids)}
        metrics = evaluate_run(task.qrels, task_run, k=args.eval_k, exclude_doc_ids=task.exclude_doc_ids)
        task_metrics[task_name] = metrics
        output_runs[task_name] = task_run
        embedding_files[task_name] = {
            "queries": str(args.cache_dir / f"{query_key}.npy"),
            "docs": str(args.cache_dir / f"{doc_key}.npy"),
        }

    metric = f"ndcg@{args.eval_k}"
    report = {
        "method": "dense_retrieval",
        "model": args.model,
        "split": args.split,
        "top_k": args.top_k,
        "eval_k": args.eval_k,
        "instruction_mode": args.instruction_mode,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "elapsed_seconds": time.perf_counter() - started,
        "embedding_files": embedding_files,
        "task_metrics": task_metrics,
        "overall": {
            metric: weighted_average(task_metrics, metric),
            f"recall@{args.eval_k}": weighted_average(task_metrics, f"recall@{args.eval_k}"),
            f"map@{args.eval_k}": weighted_average(task_metrics, f"map@{args.eval_k}"),
        },
        "runs": output_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
