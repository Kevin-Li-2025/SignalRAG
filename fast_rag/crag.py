from __future__ import annotations

from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

from .models import Document, Evidence
from .rank import tokenize
from .search import SearchFilters


@dataclass(frozen=True)
class RetrievalAssessment:
    status: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    corrective_queries: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    metrics: dict[str, float | int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["confidence"] = round(self.confidence, 4)
        return data


def assess_retrieval(
    query: str,
    docs: list[Document],
    evidence: list[Evidence],
    filters: SearchFilters | None = None,
) -> RetrievalAssessment:
    filters = filters or SearchFilters()
    query_tokens = set(tokenize(query))
    evidence_tokens = set(tokenize(" ".join(item.passage for item in evidence)))
    overlap = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
    domains = {_domain(item.url) for item in evidence if _domain(item.url)}
    official = sum(1 for item in evidence if item.provider == "official")
    top_score = max((item.score for item in evidence), default=0.0)
    doc_count = len(docs)
    evidence_count = len(evidence)

    confidence = 0.0
    confidence += min(overlap, 0.55) * 0.8
    confidence += min(top_score / 8.0, 0.25)
    confidence += min(evidence_count / 8.0, 0.12)
    confidence += min(len(domains) / 4.0, 0.08)
    if official:
        confidence += 0.08
    confidence = min(confidence, 1.0)

    reasons: list[str] = []
    actions: list[str] = []
    corrective_queries: list[str] = []
    if evidence_count < 3:
        reasons.append("too_few_evidence_passages")
        corrective_queries.append(f"{query} official source")
    if overlap < 0.24:
        reasons.append("low_query_coverage")
        corrective_queries.append(f"{query} exact answer")
    if len(domains) < 2 and not filters.include_domains:
        reasons.append("low_source_diversity")
        corrective_queries.append(f"{query} independent sources")
    if not official and not filters.include_domains:
        reasons.append("no_curated_official_source")
        corrective_queries.append(f"{query} official documentation")

    if confidence >= 0.58 and not reasons:
        status = "sufficient"
    elif confidence >= 0.38:
        status = "ambiguous"
        actions.append("expand_or_refine_search")
    else:
        status = "insufficient"
        actions.append("corrective_search")

    if filters.include_domains:
        actions.append("respect_domain_allowlist")
    if filters.exclude_domains:
        actions.append("respect_domain_denylist")
    corrective_queries = _dedupe(corrective_queries)[:4]
    return RetrievalAssessment(
        status=status,
        confidence=confidence,
        reasons=reasons,
        corrective_queries=corrective_queries,
        actions=actions,
        metrics={
            "query_token_coverage": round(overlap, 4),
            "domains": len(domains),
            "official_sources": official,
            "documents": doc_count,
            "evidence": evidence_count,
            "top_score": round(top_score, 4),
        },
    )


def should_correct(assessment: RetrievalAssessment, already_corrected: bool = False) -> bool:
    if already_corrected:
        return False
    if not assessment.corrective_queries:
        return False
    return assessment.status in {"insufficient", "ambiguous"} or assessment.confidence < 0.5


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        lowered = item.lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            deduped.append(item)
    return deduped
