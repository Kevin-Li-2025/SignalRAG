from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z0-9_.%+-]+")


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    cjk = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
    return words + cjk


def lexical_score(query: str, document: str) -> float:
    q_tokens = _tokens(query)
    d_tokens = _tokens(document)
    if not q_tokens or not d_tokens:
        return 0.0

    q_counts = Counter(q_tokens)
    d_counts = Counter(d_tokens)
    overlap = sum(min(q_counts[token], d_counts.get(token, 0)) for token in q_counts)
    precision = overlap / max(1, len(d_tokens))
    recall = overlap / max(1, len(q_tokens))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    exact_bonus = 0.0
    for token in q_counts:
        if len(token) >= 2 and token in d_counts:
            exact_bonus += math.log1p(q_counts[token] + d_counts[token])
    return f1 + 0.02 * exact_bonus


def title_text(document: str, max_chars: int = 96) -> str:
    first_line = document.strip().splitlines()[0] if document.strip() else ""
    if "  " in first_line:
        first_line = first_line.split("  ", 1)[0]
    return first_line[:max_chars]


def head_text(document: str, max_chars: int = 192) -> str:
    return document.strip().replace("\n", " ")[:max_chars]


def _cjk_ngrams(text: str, n: int) -> set[str]:
    chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    if len(chars) < n:
        return set()
    return {"".join(chars[idx : idx + n]) for idx in range(len(chars) - n + 1)}


def cjk_ngram_recall(query: str, document: str, n: int) -> float:
    q_ngrams = _cjk_ngrams(query, n)
    if not q_ngrams:
        return 0.0
    d_ngrams = _cjk_ngrams(document, n)
    return len(q_ngrams & d_ngrams) / len(q_ngrams)


def lexical_feature_values(query: str, document: str) -> dict[str, float]:
    title = title_text(document)
    head = head_text(document)
    return {
        "lexical": lexical_score(query, document),
        "head_lexical": lexical_score(query, head),
        "title_lexical": lexical_score(query, title),
        "doc_bigram": cjk_ngram_recall(query, document, 2),
        "head_bigram": cjk_ngram_recall(query, head, 2),
        "title_bigram": cjk_ngram_recall(query, title, 2),
        "title_trigram": cjk_ngram_recall(query, title, 3),
    }


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-12:
        return [0.0 for _ in values]
    return [(value - mean_value) / std for value in values]


def blend_scores(model_scores: list[float], lexical_scores: list[float], alpha: float) -> list[float]:
    normalized_lexical = zscore(lexical_scores)
    normalized_model = zscore(model_scores)
    return [
        model_score + alpha * lexical_score
        for model_score, lexical_score in zip(normalized_model, normalized_lexical)
    ]


def blend_scores_by_group(
    group_ids: list[str],
    model_scores: list[float],
    lexical_scores: list[float],
    alpha: float,
) -> list[float]:
    grouped: dict[str, list[int]] = {}
    for idx, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(idx)

    blended = [0.0 for _ in model_scores]
    for indices in grouped.values():
        group_model = zscore([model_scores[idx] for idx in indices])
        group_lexical = zscore([lexical_scores[idx] for idx in indices])
        for local_idx, global_idx in enumerate(indices):
            blended[global_idx] = group_model[local_idx] + alpha * group_lexical[local_idx]
    return blended


def blend_feature_by_group(
    group_ids: list[str],
    model_scores: list[float],
    feature_scores: list[float],
    alpha: float,
) -> list[float]:
    return blend_scores_by_group(group_ids, model_scores, feature_scores, alpha)


def reciprocal_rank_feature(values: list[float], k: float = 60.0) -> list[float]:
    ranked = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    scores = [0.0 for _ in values]
    for rank, (idx, _) in enumerate(ranked, start=1):
        scores[idx] = 1.0 / (k + rank)
    return scores


def rrf_blend_by_group(
    group_ids: list[str],
    model_scores: list[float],
    feature_scores: list[float],
    alpha: float,
    k: float,
) -> list[float]:
    grouped: dict[str, list[int]] = {}
    for idx, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(idx)

    blended = [0.0 for _ in model_scores]
    for indices in grouped.values():
        model_rrf = reciprocal_rank_feature([model_scores[idx] for idx in indices], k)
        feature_rrf = reciprocal_rank_feature([feature_scores[idx] for idx in indices], k)
        for local_idx, global_idx in enumerate(indices):
            blended[global_idx] = model_rrf[local_idx] + alpha * feature_rrf[local_idx]
    return blended
