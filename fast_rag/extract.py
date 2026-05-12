from __future__ import annotations

import re
from html import unescape

from bs4 import BeautifulSoup


SPACE_RE = re.compile(r"\s+")
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def clean_text(text: str) -> str:
    text = unescape(text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def extract_html(html: str, fallback_title: str = "") -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "canvas", "form", "button"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "aside"]):
        tag.decompose()

    title = clean_text(soup.title.get_text(" ")) if soup.title else fallback_title
    meta = soup.find("meta", attrs={"name": "description"})
    meta_text = clean_text(meta.get("content", "")) if meta else ""

    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks: list[str] = []
    for tag in root.find_all(["h1", "h2", "h3", "p", "li"]):
        value = clean_text(tag.get_text(" "))
        if len(value) >= 40:
            blocks.append(value)

    body = clean_text(" ".join(blocks))
    if len(body) < 300:
        body = clean_text(root.get_text(" "))

    if meta_text and meta_text not in body:
        body = f"{meta_text} {body}".strip()
    return title or fallback_title, body


def split_passages(text: str, target_chars: int = 900, overlap_chars: int = 0) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    sentences = SENTENCE_RE.split(text)
    passages: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= target_chars:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            passages.append(current)
        prefix = current[-overlap_chars:] if overlap_chars and current else ""
        current = f"{prefix} {sentence}".strip()

    if current:
        passages.append(current)

    if len(passages) == 1 and len(passages[0]) > target_chars * 1.5:
        long_text = passages[0]
        passages = [
            long_text[i : i + target_chars]
            for i in range(0, len(long_text), max(1, target_chars - overlap_chars))
        ]
    return passages
