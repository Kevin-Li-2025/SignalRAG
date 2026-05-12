from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import clean_text
from .models import Evidence
from .rank import domain_for, query_phrases, tokenize


SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
MODE_CHAR_BUDGET = {
    "fast": 4_800,
    "pro": 7_200,
    "deep": 12_000,
}
MODE_ITEM_BUDGET = {
    "fast": 780,
    "pro": 980,
    "deep": 1_100,
}


@dataclass(frozen=True)
class PackedContext:
    evidence: list[Evidence]
    meta: dict


def pack_answer_context(query: str, evidence: list[Evidence], mode: str) -> PackedContext:
    """Compress answer evidence without losing citation IDs or source context."""

    budget = MODE_CHAR_BUDGET.get(mode, MODE_CHAR_BUDGET["fast"])
    per_item = MODE_ITEM_BUDGET.get(mode, MODE_ITEM_BUDGET["fast"])
    original_chars = sum(len(item.passage) for item in evidence)
    contextual_original_chars = sum(len(_source_context_prefix(item)) + len(item.passage) + 1 for item in evidence)
    packed: list[Evidence] = []
    used_chars = 0

    for item in _sandwich_reorder(evidence):
        remaining = budget - used_chars
        if remaining <= 220:
            break
        target = min(per_item, remaining)
        passage = _compress_passage(query, item, target)
        if not passage:
            continue
        used_chars += len(passage)
        packed.append(
            Evidence(
                id=item.id,
                title=item.title,
                url=item.url,
                passage=passage,
                score=item.score,
                provider=item.provider,
                signals={
                    **item.signals,
                    "packed_chars": len(passage),
                    "original_chars": len(item.passage),
                },
            )
        )

    packed_chars = sum(len(item.passage) for item in packed)
    ratio = packed_chars / max(contextual_original_chars, 1)
    return PackedContext(
        evidence=packed,
        meta={
            "strategy": "query_aware_contextual_sandwich",
            "budget_chars": budget,
            "input_evidence": len(evidence),
            "packed_evidence": len(packed),
            "original_chars": original_chars,
            "contextual_original_chars": contextual_original_chars,
            "packed_chars": packed_chars,
            "compression_ratio": round(ratio, 4),
        },
    )


def contextual_passage_text(title: str, url: str, snippet: str, passage: str) -> str:
    domain = domain_for(url)
    parts = [
        f"Source title: {clean_text(title)}",
        f"Source domain: {domain}",
    ]
    snippet = clean_text(snippet)
    if snippet:
        parts.append(f"Source snippet: {snippet}")
    parts.append(passage)
    return ". ".join(part for part in parts if part)


def _compress_passage(query: str, item: Evidence, max_chars: int) -> str:
    prefix = _source_context_prefix(item)
    passage_budget = max(180, max_chars - len(prefix) - 1)
    body = _select_sentences(query, item, passage_budget)
    body = clean_text(body)
    if not body:
        return ""
    return clean_text(f"{prefix} {body}")[:max_chars].rstrip()


def _source_context_prefix(item: Evidence) -> str:
    domain = domain_for(item.url)
    return clean_text(
        f"Source context: title={item.title}; domain={domain}; provider={item.provider}. Passage:"
    )


def _select_sentences(query: str, item: Evidence, budget: int) -> str:
    sentences = [clean_text(sentence) for sentence in SENTENCE_RE.split(item.passage) if clean_text(sentence)]
    if not sentences:
        return item.passage[:budget].rstrip()

    query_tokens = set(tokenize(query))
    title_tokens = set(tokenize(item.title))
    phrases = query_phrases(query)
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_tokens = set(tokenize(sentence))
        if not sentence_tokens:
            continue
        score = len(query_tokens & sentence_tokens) * 2.0
        score += len(title_tokens & sentence_tokens) * 0.35
        lowered = sentence.lower()
        score += sum(1.25 for phrase in phrases if phrase in lowered)
        if index == 0:
            score += 0.35
        if index == len(sentences) - 1:
            score += 0.2
        if len(sentence) < 45:
            score -= 0.25
        scored.append((score, index, sentence))

    if not scored:
        return item.passage[:budget].rstrip()

    selected_indexes: list[int] = []
    used = 0
    for _, index, sentence in sorted(scored, key=lambda row: row[0], reverse=True):
        if used + len(sentence) + 1 > budget and selected_indexes:
            continue
        selected_indexes.append(index)
        used += len(sentence) + 1
        if used >= budget:
            break

    selected_indexes.sort()
    output = " ".join(sentences[index] for index in selected_indexes)
    return output[:budget].rstrip()


def _sandwich_reorder(evidence: list[Evidence]) -> list[Evidence]:
    if len(evidence) <= 2:
        return evidence
    front: list[Evidence] = []
    back: list[Evidence] = []
    for index, item in enumerate(evidence):
        if index % 2 == 0:
            front.append(item)
        else:
            back.append(item)
    return front + list(reversed(back))
