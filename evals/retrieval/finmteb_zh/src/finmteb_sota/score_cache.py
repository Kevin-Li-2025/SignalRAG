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


def score_cache_key(
    task: RerankingTask,
    split: str,
    instruction: str,
    cache_tag: str,
) -> str:
    digest = instruction_digest(instruction)
    return f"{task.leaderboard_name}_{split}_{cache_tag}_{digest}.json"


def load_score_cache(
    cache_dir: Path,
    task: RerankingTask,
    split: str,
    instruction: str,
    cache_tag: str,
) -> tuple[list[float], Path]:
    cache_path = cache_dir / score_cache_key(task, split, instruction, cache_tag)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing score cache: {cache_path}")
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return [float(value) for value in payload["scores"]], cache_path


def write_score_cache(
    cache_dir: Path,
    task: RerankingTask,
    split: str,
    instruction: str,
    cache_tag: str,
    scores: list[float],
    extra: dict[str, Any] | None = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / score_cache_key(task, split, instruction, cache_tag)
    payload: dict[str, Any] = {
        "dataset": task.dataset_id,
        "leaderboard_name": task.leaderboard_name,
        "split": split,
        "instruction": instruction,
        "cache_tag": cache_tag,
        "scores": scores,
    }
    if extra:
        payload.update(extra)
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return cache_path
