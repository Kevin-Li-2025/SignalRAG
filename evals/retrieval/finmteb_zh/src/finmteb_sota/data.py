from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RerankRecord:
    query_id: str
    query: str
    positives: tuple[str, ...]
    negatives: tuple[str, ...]


def _as_text_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, dict):
        texts = []
        for key in ("text", "contents", "content", "passage", "document", "doc"):
            if value.get(key):
                texts.append(str(value[key]))
        if texts:
            return tuple(texts)
        return tuple(str(item) for item in value.values() if item)
    if isinstance(value, Iterable):
        texts: list[str] = []
        for item in value:
            texts.extend(_as_text_list(item))
        return tuple(text for text in texts if text)
    return (str(value),)


def normalize_record(raw: dict[str, Any], idx: int) -> RerankRecord:
    query = raw.get("query") or raw.get("question") or raw.get("sentence") or raw.get("text")
    if query is None:
        raise ValueError(f"Cannot find query field in row {idx}: {sorted(raw)}")

    positives = (
        _as_text_list(raw.get("positive"))
        or _as_text_list(raw.get("positives"))
        or _as_text_list(raw.get("positive_passages"))
        or _as_text_list(raw.get("relevant_docs"))
    )
    negatives = (
        _as_text_list(raw.get("negative"))
        or _as_text_list(raw.get("negatives"))
        or _as_text_list(raw.get("negative_passages"))
        or _as_text_list(raw.get("hard_negatives"))
    )
    if not positives or not negatives:
        raise ValueError(f"Cannot find positive/negative passages in row {idx}: {sorted(raw)}")

    query_id = str(raw.get("query_id") or raw.get("qid") or raw.get("id") or idx)
    return RerankRecord(query_id=query_id, query=str(query), positives=positives, negatives=negatives)


def load_reranking_records(dataset_id: str, split: str) -> list[RerankRecord]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    records = []
    skipped = 0
    for idx, row in enumerate(dataset):
        try:
            records.append(normalize_record(dict(row), idx))
        except ValueError:
            skipped += 1
    if not records:
        raise RuntimeError(f"No usable records loaded from {dataset_id}:{split}; skipped={skipped}")
    return records


def train_validation_split(
    records: list[RerankRecord], validation_ratio: float, seed: int
) -> tuple[list[RerankRecord], list[RerankRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_ratio)) if len(shuffled) > 1 else 0
    return shuffled[validation_size:], shuffled[:validation_size]


def iter_pair_examples(
    records: list[RerankRecord],
    negatives_per_query: int,
    rng: random.Random,
) -> Iterable[dict[str, str | int]]:
    for record in records:
        candidates = [
            {
                "query_id": record.query_id,
                "query": record.query,
                "document": positive,
                "label": 1,
            }
            for positive in record.positives
        ]

        negatives = list(record.negatives)
        rng.shuffle(negatives)
        candidates.extend(
            {
                "query_id": record.query_id,
                "query": record.query,
                "document": negative,
                "label": 0,
            }
            for negative in negatives[:negatives_per_query]
        )
        rng.shuffle(candidates)
        yield from candidates


def flatten_records(
    records: list[RerankRecord], seed: int = 20260905
) -> tuple[list[str], list[str], list[int], list[str]]:
    """Flatten candidates in a deterministic order unrelated to their labels."""
    queries: list[str] = []
    docs: list[str] = []
    labels: list[int] = []
    qids: list[str] = []
    for record in records:
        candidates = [*(record.positives), *(record.negatives)]
        digest = hashlib.sha256(
            f"{seed}\0{record.query_id}\0{record.query}".encode()
        ).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(candidates)
        positive_counts: dict[str, int] = {}
        for text in record.positives:
            positive_counts[text] = positive_counts.get(text, 0) + 1
        for document in candidates:
            label = int(positive_counts.get(document, 0) > 0)
            if label:
                positive_counts[document] -= 1
            queries.append(record.query)
            docs.append(document)
            labels.append(label)
            qids.append(record.query_id)
    return queries, docs, labels, qids
