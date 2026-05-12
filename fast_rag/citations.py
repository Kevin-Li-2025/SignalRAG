from __future__ import annotations

import json
import os
import re

import httpx

from .config import settings
from .extract import clean_text
from .models import Evidence
from .rank import tokenize


CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SOURCE_FOOTER_RE = re.compile(
    r"\n+\s*(?:\*\*)?(?:sources checked|sources|references)(?:\*\*)?\s*:.*$",
    re.IGNORECASE | re.DOTALL,
)
CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+(?!\[\d)")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s*)")
GENERIC_UNCITED_PREFIXES = (
    "based on",
    "基于",
    "下面",
    "以下",
)
SUPPORT_STOPWORDS = {
    "also",
    "answer",
    "based",
    "because",
    "claim",
    "could",
    "does",
    "from",
    "have",
    "into",
    "means",
    "more",
    "only",
    "should",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "through",
    "with",
    "would",
}


def serialize_evidence(item: Evidence) -> dict:
    return {
        "id": item.id,
        "title": item.title,
        "url": item.url,
        "passage": item.passage,
        "score": round(item.score, 4),
        "provider": item.provider,
        "signals": item.signals,
    }


def select_answer_evidence(evidence: list[Evidence]) -> list[Evidence]:
    selected: list[Evidence] = []
    seen_urls: set[str] = set()
    for item in evidence:
        key = item.url.rstrip("/")
        if key in seen_urls:
            continue
        seen_urls.add(key)
        selected.append(item)
    return selected


def normalize_answer_citations(answer: str, evidence: list[Evidence]) -> tuple[str, list[dict], list[int]]:
    valid_ids = {item.id for item in evidence}
    used_ids: list[int] = []
    answer = SOURCE_FOOTER_RE.sub("", answer).strip()

    def replace(match: re.Match[str]) -> str:
        ids = [int(value) for value in re.split(r"\s*,\s*", match.group(1)) if value.strip()]
        valid = []
        for citation_id in ids:
            if citation_id not in valid_ids:
                continue
            valid.append(citation_id)
            if citation_id not in used_ids:
                used_ids.append(citation_id)
        if not valid:
            return ""
        return "".join(f"[{citation_id}]" for citation_id in valid)

    normalized = CITATION_RE.sub(replace, answer)
    evidence_by_id = {item.id: item for item in evidence}
    used_citations = [serialize_evidence(evidence_by_id[citation_id]) for citation_id in used_ids]
    return normalized, used_citations, used_ids


def verify_claim_citations(answer: str, evidence: list[Evidence]) -> list[dict]:
    evidence_by_id = {item.id: item for item in evidence}
    verified: list[dict] = []
    for raw_claim, inherited_ids in _split_claims_with_inherited_citations(answer):
        own_ids = _citation_ids(raw_claim)
        citation_ids = own_ids or inherited_ids
        claim = _clean_claim(raw_claim)
        if not _keep_claim(claim, citation_ids):
            continue
        status, score = _support_status(claim, citation_ids, evidence_by_id)
        citations = [
            serialize_evidence(evidence_by_id[citation_id])
            for citation_id in citation_ids
            if citation_id in evidence_by_id
        ]
        verified.append(
            {
                "claim": claim,
                "citation_ids": citation_ids,
                "citations": citations,
                "status": status,
                "support_score": round(score, 4),
                "verifier": "lexical",
            }
        )
    return verified


async def verify_claim_citations_with_judge(
    query: str,
    claims: list[dict],
    evidence: list[Evidence],
) -> list[dict]:
    if not claims or not evidence or not settings.deepseek_api_key:
        return claims
    judgeable = [
        {**claim, "_original_index": index}
        for index, claim in enumerate(claims)
        if claim.get("citation_ids") and _needs_judge(claim)
    ]
    if not judgeable:
        return claims
    payload = {
        "model": os.getenv("DEEPSEEK_VERIFIER_MODEL", os.getenv("DEEPSEEK_MODEL", settings.deepseek_model)),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a citation verifier for a web-search RAG system. "
                    "Use only the supplied evidence. For each claim, decide whether the cited evidence supports it. "
                    "Return only valid JSON with a 'claims' array. Status must be one of: "
                    "supported, weak, contradicted, insufficient. Keep rationale brief."
                ),
            },
            {
                "role": "user",
                "content": _judge_prompt(query, judgeable, evidence),
            },
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 700,
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{settings.deepseek_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        return _merge_judgements(
            claims,
            json.loads(content),
            [int(claim["_original_index"]) for claim in judgeable],
        )
    except Exception:
        return claims


def _split_claims(answer: str) -> list[str]:
    return [claim for claim, _ in _split_claims_with_inherited_citations(answer)]


def _split_claims_with_inherited_citations(answer: str) -> list[tuple[str, list[int]]]:
    inherited: list[tuple[str, list[int]]] = []
    for line in answer.splitlines():
        stripped = LIST_MARKER_RE.sub("", line.strip())
        if not stripped:
            continue
        line_ids = _citation_ids(stripped)
        for part in CLAIM_SPLIT_RE.split(stripped):
            part = part.strip()
            if not part:
                continue
            inherited.append((part, line_ids if not _citation_ids(part) else []))
    return inherited


def _needs_judge(claim: dict) -> bool:
    status = claim.get("status")
    score = float(claim.get("support_score") or 0.0)
    return status != "supported" or score < 0.34


def _citation_ids(text: str) -> list[int]:
    ids: list[int] = []
    for match in CITATION_RE.finditer(text):
        for value in re.split(r"\s*,\s*", match.group(1)):
            if not value.strip():
                continue
            citation_id = int(value)
            if citation_id not in ids:
                ids.append(citation_id)
    return ids


def _clean_claim(text: str) -> str:
    text = CITATION_RE.sub("", text)
    text = LIST_MARKER_RE.sub("", text)
    return clean_text(text).strip(" -:：")


def _keep_claim(claim: str, citation_ids: list[int]) -> bool:
    if not claim:
        return False
    lowered = claim.lower()
    if not citation_ids and any(lowered.startswith(prefix) for prefix in GENERIC_UNCITED_PREFIXES):
        return False
    if not citation_ids and (len(claim) < 42 or claim.endswith((":", "："))):
        return False
    return True


def _support_status(claim: str, citation_ids: list[int], evidence_by_id: dict[int, Evidence]) -> tuple[str, float]:
    if not citation_ids:
        return "missing_citation", 0.0
    if not any(citation_id in evidence_by_id for citation_id in citation_ids):
        return "unsupported", 0.0

    score = max(
        (
            _support_score(claim, evidence_by_id[citation_id])
            for citation_id in citation_ids
            if citation_id in evidence_by_id
        ),
        default=0.0,
    )
    if score >= 0.22:
        return "supported", score
    if score >= 0.08:
        return "weak", score
    return "needs_review", score


def _support_score(claim: str, evidence: Evidence) -> float:
    claim_tokens = set(_support_tokens(claim))
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(_support_tokens(f"{evidence.title} {evidence.passage}"))
    if not evidence_tokens:
        return 0.0
    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    title_tokens = set(_support_tokens(evidence.title))
    if claim_tokens & title_tokens:
        overlap += 0.04
    return min(overlap, 1.0)


def _support_tokens(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if token not in SUPPORT_STOPWORDS and not token.isdigit()
    ]


def _judge_prompt(query: str, claims: list[dict], evidence: list[Evidence]) -> str:
    evidence_by_id = {item.id: item for item in evidence}
    cited_ids = {
        citation_id
        for claim in claims
        for citation_id in claim.get("citation_ids", [])
        if citation_id in evidence_by_id
    }
    evidence_blocks = []
    for citation_id in sorted(cited_ids):
        item = evidence_by_id[citation_id]
        evidence_blocks.append(f"[{item.id}] {item.title}\nURL: {item.url}\nPASSAGE: {item.passage[:650]}")
    claim_blocks = []
    for index, claim in enumerate(claims):
        cited = [citation_id for citation_id in claim.get("citation_ids", []) if citation_id in evidence_by_id]
        claim_blocks.append(
            f"{index}. CLAIM: {claim.get('claim', '')}\nCITED_IDS: {cited}"
        )
    return (
        f"Question: {query}\n\n"
        f"Evidence:\n{chr(10).join(evidence_blocks)}\n\n"
        f"Claims:\n{chr(10).join(claim_blocks)}\n\n"
        "Return JSON like: "
        '{"claims":[{"index":0,"status":"supported","confidence":0.9,'
        '"rationale":"brief reason","supporting_quote":"short exact evidence text"}]}'
    )


def _merge_judgements(
    claims: list[dict],
    raw: dict | list,
    original_indexes: list[int] | None = None,
) -> list[dict]:
    updates = raw.get("claims", []) if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    if not isinstance(updates, list):
        return claims
    valid_statuses = {"supported", "weak", "contradicted", "insufficient"}
    by_index: dict[int, dict] = {}
    for item in updates:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if original_indexes is not None:
            if index < 0 or index >= len(original_indexes):
                continue
            index = original_indexes[index]
        by_index[index] = item

    merged: list[dict] = []
    for claim_index, claim in enumerate(claims):
        item = by_index.get(claim_index)
        if not item:
            merged.append(claim)
            continue
        status = str(item.get("status") or "").lower()
        if status not in valid_statuses:
            status = claim.get("status", "weak")
        confidence = item.get("confidence", claim.get("support_score", 0))
        try:
            score = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            score = claim.get("support_score", 0)
        merged.append(
            {
                **claim,
                "status": status,
                "support_score": round(score, 4),
                "judge_rationale": clean_text(str(item.get("rationale") or ""))[:240],
                "supporting_quote": clean_text(str(item.get("supporting_quote") or ""))[:280],
                "verifier": "deepseek",
            }
        )
    return merged
