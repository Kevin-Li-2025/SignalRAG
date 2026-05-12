from __future__ import annotations

import os
import re
from typing import Any

import httpx

from .config import settings
from .extract import clean_text
from .models import Evidence
from .rank import domain_for, source_trust_tier, tokenize


CJK_RE = re.compile(r"[\u3400-\u9fff]")
CITATION_MARK_RE = re.compile(r"\[\d+\]")
MECHANISM_TOKENS = {
    "answer",
    "answers",
    "automatic",
    "based",
    "citation",
    "citations",
    "context",
    "current",
    "distilling",
    "fine-tuned",
    "information",
    "links",
    "manual",
    "model",
    "providers",
    "query",
    "queries",
    "relevant",
    "retrieve",
    "rewrites",
    "search",
    "sources",
    "synthetic",
    "third-party",
    "timely",
    "web",
}
GENERIC_STARTS = (
    "learn how",
    "this guide",
    "an introduction",
    "a fundamental overview",
)


def _language_for(query: str) -> str:
    return "Chinese" if CJK_RE.search(query) else "the same language as the user query"


def _evidence_block(evidence: list[Evidence]) -> str:
    blocks = []
    for item in evidence:
        trust_tier = item.signals.get("trust_tier") or source_trust_tier(item.url)
        priority = "primary" if item.signals.get("primary_source") else "supporting"
        blocks.append(
            f"[{item.id}] {item.title}\n"
            f"URL: {item.url}\n"
            f"SOURCE TYPE: {trust_tier}; PRIORITY: {priority}\n"
            f"PASSAGE: {item.passage}"
        )
    return "\n\n".join(blocks)


async def generate_answer(
    query: str,
    evidence: list[Evidence],
    mode: str,
    query_plan: dict[str, Any] | None = None,
) -> tuple[str, str]:
    provider = _select_provider()
    if provider == "deepseek" and evidence:
        try:
            answer = await _deepseek_answer(query, evidence, mode, query_plan)
            answer = prefer_primary_citations(query, answer, evidence)
            if not answer.strip():
                return _extractive_answer(query, evidence), "extractive"
            if _missing_required_citations(answer):
                repaired = repair_missing_citations(query, answer, evidence)
                if not _missing_required_citations(repaired):
                    return repaired, "deepseek_repaired"
                return _extractive_answer(query, evidence), "extractive"
            return answer, "deepseek"
        except Exception as exc:
            fallback = _extractive_answer(query, evidence)
            return f"{fallback}\n\n模型生成失败，已使用抽取式答案。错误：{type(exc).__name__}", "extractive"
    if provider == "openai" and evidence:
        try:
            answer = await _openai_answer(query, evidence, mode, query_plan)
            return prefer_primary_citations(query, answer, evidence), "openai"
        except Exception as exc:
            fallback = _extractive_answer(query, evidence)
            return f"{fallback}\n\n模型生成失败，已使用抽取式答案。错误：{type(exc).__name__}", "extractive"
    return _extractive_answer(query, evidence), "extractive"


def _select_provider() -> str:
    if settings.llm_provider == "deepseek":
        return "deepseek" if settings.deepseek_api_key else "extractive"
    if settings.llm_provider == "openai":
        return "openai" if settings.openai_api_key else "extractive"
    if settings.deepseek_api_key:
        return "deepseek"
    if settings.openai_api_key:
        return "openai"
    return "extractive"


def _system_prompt(query: str, query_plan: dict[str, Any] | None = None) -> str:
    language = _language_for(query)
    prompt = (
        "You are a fast, highly accurate web-search RAG answerer. "
        "Use only the supplied evidence. Every factual claim must cite sources like [1]. "
        "When primary/official evidence supports the answer, cite it before general blogs, forums, or secondary explainers. "
        "If evidence is weak or conflicting, say that clearly. "
        "Prefer concise synthesis over long summaries. "
        "Do not mention availability, pricing, plan access, rollout status, or dates unless the user asks. "
        "When evidence contains both older launch text and newer help-center text, prefer the newer operational description. "
        f"Answer in {language}."
    )
    if query_plan:
        prompt += (
            " Internal query plan: "
            f"intent={query_plan.get('intent')}; "
            f"answer_style={query_plan.get('answer_style')}; "
            f"needs_freshness={query_plan.get('needs_freshness')}; "
            f"reasoning_effort={query_plan.get('reasoning_effort')}. "
            "Use this plan silently; do not mention it."
        )
    return prompt


def _user_prompt(query: str, evidence: list[Evidence], mode: str) -> str:
    deep_instruction = ""
    if mode == "deep":
        deep_instruction = (
            "\nDeep Research requirements: produce a structured research-grade answer with "
            "a direct conclusion, synthesized evidence, caveats or conflicts, and practical next steps when useful. "
            "Keep it compact: target 8-12 verifiable claims, every factual sentence must end with citation IDs like [1], "
            "avoid uncited background, and do not cite anything not present in the evidence.\n"
        )
    return (
        f"Question: {query}\n"
        f"Mode: {mode}\n\n"
        f"Evidence:\n{_evidence_block(evidence)}\n\n"
        f"{deep_instruction}"
        "Write a direct answer with citations. Prefer PRIMARY sources when they answer the question. "
        "For claims supported by multiple independent PRIMARY sources, cite up to two of them. "
        "Do not include a bibliography, source list, or 'Sources checked' section."
    )


async def _deepseek_answer(
    query: str,
    evidence: list[Evidence],
    mode: str,
    query_plan: dict[str, Any] | None = None,
) -> str:
    reasoning_effort = str((query_plan or {}).get("reasoning_effort") or "none").lower()
    if mode == "deep" and reasoning_effort == "none":
        reasoning_effort = "high"
    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", settings.deepseek_model),
        "messages": [
            {"role": "system", "content": _system_prompt(query, query_plan)},
            {"role": "user", "content": _user_prompt(query, evidence, mode)},
        ],
        "max_tokens": 900,
    }
    if reasoning_effort in {"high", "max"}:
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = reasoning_effort
        payload["max_tokens"] = 1600 if reasoning_effort == "high" else 3400
    else:
        payload["thinking"] = {"type": "disabled"}
        payload["temperature"] = 0.1
    timeout = 90 if reasoning_effort == "max" else 50 if reasoning_effort == "high" else 35
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{settings.deepseek_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _missing_required_citations(answer: str) -> bool:
    return not answer.strip() or not CITATION_MARK_RE.search(answer)


def repair_missing_citations(query: str, answer: str, evidence: list[Evidence]) -> str:
    if not answer.strip() or CITATION_MARK_RE.search(answer) or not evidence:
        return answer
    lines: list[str] = []
    fallback_id = evidence[0].id
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if stripped.startswith(("#", "|", "---")):
            lines.append(line)
            continue
        if len(stripped) < 24 and stripped.endswith(":"):
            lines.append(line)
            continue
        citation_id = _best_citation_id(stripped, evidence) or _best_citation_id(query, evidence) or fallback_id
        lines.append(f"{line.rstrip()} [{citation_id}]")
    return "\n".join(lines)


def prefer_primary_citations(query: str, answer: str, evidence: list[Evidence]) -> str:
    if not answer.strip() or not CITATION_MARK_RE.search(answer):
        return answer
    primary = [item for item in evidence if _is_primary(item)]
    if not primary:
        return answer

    output: list[str] = []
    in_code_block = False
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            output.append(line)
            continue
        if in_code_block or not CITATION_MARK_RE.search(line):
            output.append(line)
            continue
        cited_ids = {int(match.group(0).strip("[]")) for match in CITATION_MARK_RE.finditer(line)}
        if any(item.id in cited_ids for item in primary):
            best = _best_complementary_primary_citation_id(query, line, primary, cited_ids)
            if best is not None:
                output.append(_append_citation_to_first_group(line, best))
                continue
            output.append(line)
            continue
        best = _best_primary_citation_id(query, line, primary, cited_ids)
        if best is None:
            output.append(line)
            continue
        output.append(_append_citation_to_first_group(line, best))
    return "\n".join(output)


def _is_primary(item: Evidence) -> bool:
    return bool(item.signals.get("primary_source")) or source_trust_tier(item.url) in {
        "government",
        "academic",
        "standards",
        "official_docs",
        "medical",
        "news_wire",
    } or item.provider in {"official", "seed"}


def _best_primary_citation_id(
    query: str,
    text: str,
    primary: list[Evidence],
    existing_ids: set[int],
) -> int | None:
    text_tokens = set(tokenize(text))
    query_tokens = set(tokenize(query))
    best: tuple[float, int] | None = None
    for item in primary:
        if item.id in existing_ids:
            continue
        evidence_tokens = set(tokenize(f"{item.title} {item.passage}"))
        if not evidence_tokens:
            continue
        text_overlap = len(text_tokens & evidence_tokens) / max(len(text_tokens), 1)
        query_overlap = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
        if text_overlap < 0.10 and query_overlap < 0.38:
            continue
        tier_bonus = 0.35 if source_trust_tier(item.url) in {"government", "official_docs", "standards"} else 0.2
        score = text_overlap * 3.0 + query_overlap + tier_bonus + min(item.score, 8.0) * 0.03
        candidate = (score, item.id)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if not best or best[0] < 0.9:
        return None
    return best[1]


def _best_complementary_primary_citation_id(
    query: str,
    text: str,
    primary: list[Evidence],
    existing_ids: set[int],
) -> int | None:
    if len(existing_ids) >= 4:
        return None
    text_tokens = set(tokenize(text))
    query_tokens = set(tokenize(query))
    cited_domains = {domain_for(item.url) for item in primary if item.id in existing_ids}
    best: tuple[float, int] | None = None
    for item in primary:
        if item.id in existing_ids:
            continue
        evidence_tokens = set(tokenize(f"{item.title} {item.passage}"))
        if not evidence_tokens:
            continue
        text_overlap = len(text_tokens & evidence_tokens) / max(len(text_tokens), 1)
        query_overlap = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
        if text_overlap < 0.14 and query_overlap < 0.46:
            continue
        tier = source_trust_tier(item.url)
        tier_bonus = 0.35 if tier in {"government", "official_docs", "standards"} else 0.2
        domain_bonus = 0.22 if domain_for(item.url) not in cited_domains else -0.18
        score = text_overlap * 2.6 + query_overlap * 1.2 + tier_bonus + domain_bonus + min(item.score, 8.0) * 0.02
        candidate = (score, item.id)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if not best or best[0] < 1.15:
        return None
    return best[1]


def _append_citation_to_first_group(line: str, citation_id: int) -> str:
    def replace(match: re.Match[str]) -> str:
        return f"{match.group(0)}[{citation_id}]"

    return CITATION_MARK_RE.sub(replace, line, count=1)


def _best_citation_id(text: str, evidence: list[Evidence]) -> int | None:
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return evidence[0].id if evidence else None
    best: tuple[float, int] | None = None
    for item in evidence:
        evidence_tokens = set(tokenize(f"{item.title} {item.passage}"))
        overlap = len(text_tokens & evidence_tokens) / max(len(text_tokens), 1)
        score = overlap + min(item.score, 10) * 0.01
        if item.provider == "official":
            score += 0.03
        candidate = (score, item.id)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if not best or best[0] <= 0:
        return None
    return best[1]


async def _openai_answer(
    query: str,
    evidence: list[Evidence],
    mode: str,
    query_plan: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": os.getenv("OPENAI_MODEL", settings.openai_model),
        "instructions": _system_prompt(query, query_plan),
        "input": _user_prompt(query, evidence, mode),
        "temperature": 0.1,
    }
    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    output_text = data.get("output_text")
    if output_text:
        return output_text.strip()

    parts: list[str] = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                parts.append(content.get("text", ""))
    return "\n".join(part for part in parts if part).strip()


def _extractive_answer(query: str, evidence: list[Evidence]) -> str:
    if not evidence:
        return "没有找到足够可靠的网页证据。建议换一个更具体的问题，或配置 BRAVE_API_KEY / OPENAI_API_KEY 提升召回和综合质量。"

    patterned = _pattern_answer(query, evidence)
    if patterned:
        return patterned

    sentences = _rank_sentences(query, evidence)
    lines = ["基于当前检索到的证据，答案是："]
    for sentence, citation_id in sentences[:5]:
        lines.append(f"- {sentence} [{citation_id}]")
    return "\n".join(lines)


def _pattern_answer(query: str, evidence: list[Evidence]) -> str | None:
    lowered_query = query.lower()
    if "chatgpt" not in lowered_query or "search" not in lowered_query:
        return None

    slots = [
        ("模型层", ("search model", "fine-tuned")),
        ("检索源", ("third-party search providers", "content provided directly")),
        ("查询改写", ("rewrites your query", "targeted queries")),
        ("触发方式", ("choose to search", "manually choose")),
        ("输出形式", ("links to relevant web sources", "citations", "source links")),
    ]
    lines = ["基于当前检索到的证据，ChatGPT Search 大致是这样工作的："]
    used: set[int] = set()
    for label, needles in slots:
        match = _find_sentence(evidence, needles, used)
        if not match:
            continue
        sentence, citation_id = match
        lines.append(f"- {label}：{sentence} [{citation_id}]")
        used.add(citation_id)

    return "\n".join(lines) if len(lines) >= 4 else None


def _find_sentence(
    evidence: list[Evidence],
    needles: tuple[str, ...],
    used: set[int],
) -> tuple[str, int] | None:
    best: tuple[float, str, int] | None = None
    for item in evidence:
        for sentence in re.split(r"(?<=[.!?。！？])\s+", item.passage):
            cleaned = clean_text(sentence)
            lowered = cleaned.lower()
            if not cleaned or not any(needle in lowered for needle in needles):
                continue
            score = min(item.score, 8)
            if all(needle in lowered for needle in needles):
                score += 2.0
            if item.provider == "official":
                score += 1.0
            if item.id in used:
                score -= 0.25
            candidate = (score, cleaned[:420].rstrip(), item.id)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if not best:
        return None
    return best[1], best[2]


def _rank_sentences(query: str, evidence: list[Evidence]) -> list[tuple[str, int]]:
    query_tokens = set(tokenize(query))
    candidates: list[tuple[float, int, str]] = []
    for item in evidence:
        for sentence in re.split(r"(?<=[.!?。！？])\s+", item.passage):
            cleaned = clean_text(sentence)
            if len(cleaned) < 35:
                continue
            tokens = set(tokenize(cleaned))
            score = len(tokens & query_tokens) * 1.3
            score += len(tokens & MECHANISM_TOKENS) * 0.85
            score += min(item.score, 5) * 0.08
            if item.provider == "official":
                score += 0.4
            lowered = cleaned.lower()
            if any(lowered.startswith(prefix) for prefix in GENERIC_STARTS):
                score -= 3.0
            if "enterprise and edu" in lowered and "enterprise" not in query.lower() and "edu" not in query.lower():
                score -= 1.4
            if "deep research" in lowered and "deep" not in query.lower():
                score -= 2.2
            if "?" in cleaned:
                score -= 1.0
            candidates.append((score, item.id, cleaned))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[str, int]] = []
    seen: set[str] = set()
    citation_counts: dict[int, int] = {}
    for _, citation_id, sentence in candidates:
        fingerprint = sentence[:120].lower()
        if fingerprint in seen or citation_counts.get(citation_id, 0) >= 2:
            continue
        selected.append((sentence[:420].rstrip(), citation_id))
        seen.add(fingerprint)
        citation_counts[citation_id] = citation_counts.get(citation_id, 0) + 1
        if len(selected) >= 6:
            break

    if selected:
        return selected

    return [(_best_sentence(item.passage, query_tokens), item.id) for item in evidence[:5]]


def _best_sentence(passage: str, query_tokens: set[str]) -> str:
    best = ""
    best_score = -1.0
    for sentence in re.split(r"(?<=[.!?。！？])\s+", passage):
        cleaned = clean_text(sentence)
        if len(cleaned) < 30:
            continue
        tokens = set(tokenize(cleaned))
        score = len(tokens & query_tokens)
        if score > best_score:
            best_score = score
            best = cleaned
    if not best:
        best = clean_text(passage)
    return best[:420].rstrip()
