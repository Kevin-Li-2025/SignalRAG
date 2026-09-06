from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from finmteb_sota.tasks import RerankingTask


def model_cache_tag(model_name: str) -> str:
    """Return a compact, stable score-cache tag for a model id."""
    tail = model_name.rsplit("/", 1)[-1].lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", tail).strip("_")
    qwen_match = re.fullmatch(r"qwen3_reranker_(\d+)b", normalized)
    if qwen_match:
        return f"qwen3_{qwen_match.group(1)}b"
    return normalized or "model"


def instruction_digest(instruction: str) -> str:
    return hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:12]


def build_candidate_ids(
    qids: list[str], queries: list[str], docs: list[str]
) -> list[str]:
    """Build stable, order-independent IDs for query/document candidates."""
    if len(qids) != len(queries) or len(qids) != len(docs):
        raise ValueError("qids, queries, and docs must have identical length")
    occurrences: dict[str, int] = {}
    candidate_ids: list[str] = []
    for qid, query, doc in zip(qids, queries, docs, strict=True):
        digest = hashlib.sha256(f"{qid}\0{query}\0{doc}".encode()).hexdigest()
        occurrence = occurrences.get(digest, 0)
        occurrences[digest] = occurrence + 1
        candidate_ids.append(f"{digest}:{occurrence}")
    return candidate_ids


def score_cache_key(
    task: RerankingTask,
    split: str,
    instruction: str,
    cache_tag: str,
) -> str:
    digest = instruction_digest(instruction)
    # v3 stores candidate fingerprints and reconstructs the requested order.
    return f"{task.leaderboard_name}_{split}_{cache_tag}_{digest}_v3.json"


def load_score_cache(
    cache_dir: Path,
    task: RerankingTask,
    split: str,
    instruction: str,
    cache_tag: str,
    candidate_ids: list[str],
) -> tuple[list[float], Path]:
    cache_path = cache_dir / score_cache_key(task, split, instruction, cache_tag)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing score cache: {cache_path}")
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 3:
        raise ValueError(f"Unsupported score-cache format in {cache_path}")
    cached_ids = [str(value) for value in payload.get("candidate_ids", [])]
    cached_scores = [float(value) for value in payload.get("scores", [])]
    if len(cached_ids) != len(cached_scores):
        raise ValueError(f"Candidate ID/score length mismatch in {cache_path}")
    if len(cached_ids) != len(set(cached_ids)):
        raise ValueError(f"Duplicate candidate IDs in {cache_path}")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate expected candidate IDs")
    missing = set(candidate_ids) - set(cached_ids)
    extra = set(cached_ids) - set(candidate_ids)
    if missing or extra:
        raise ValueError(
            f"Candidate coverage mismatch in {cache_path}: missing={len(missing)}, extra={len(extra)}"
        )
    score_by_id = dict(zip(cached_ids, cached_scores, strict=True))
    return [score_by_id[candidate_id] for candidate_id in candidate_ids], cache_path


def write_score_cache(
    cache_dir: Path,
    task: RerankingTask,
    split: str,
    instruction: str,
    cache_tag: str,
    scores: list[float],
    candidate_ids: list[str],
    extra: dict[str, Any] | None = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / score_cache_key(task, split, instruction, cache_tag)
    if len(candidate_ids) != len(scores):
        raise ValueError(
            f"candidate_ids and scores must have identical length: {len(candidate_ids)} != {len(scores)}"
        )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate candidate IDs")
    payload: dict[str, Any] = {
        "format_version": 3,
        "dataset": task.dataset_id,
        "leaderboard_name": task.leaderboard_name,
        "split": split,
        "instruction": instruction,
        "cache_tag": cache_tag,
        "candidate_ids": candidate_ids,
        "scores": scores,
    }
    if extra:
        overlap = set(payload) & set(extra)
        if overlap:
            raise ValueError(f"extra cannot replace reserved cache fields: {sorted(overlap)}")
        payload.update(extra)
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return cache_path
