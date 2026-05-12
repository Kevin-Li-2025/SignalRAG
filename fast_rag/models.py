from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    provider: str = "web"
    rank: int = 0


@dataclass(slots=True)
class Document:
    url: str
    title: str
    text: str
    snippet: str = ""
    provider: str = "web"
    status: int | None = None
    fetched_from_cache: bool = False


@dataclass(slots=True)
class Evidence:
    id: int
    title: str
    url: str
    passage: str
    score: float
    provider: str = "web"
    signals: dict[str, float] = field(default_factory=dict)

