"""
Community summarizer — generates natural language descriptions.

Takes a community (cluster of code entities) and produces a human-readable
summary of what that group of code does. This is where we optionally bring
in an LLM — but only for summarization, never for entity extraction.

The summarizer supports three backends:
    1. "stub" — generates a template summary from entity names (no LLM).
    2. "openai" — uses the OpenAI API for high-quality summaries.
    3. "ollama" — uses a local Ollama instance for private, free summaries.

The stub backend is always available and is the default. This means the
core system works without any API keys or external services.
"""

from __future__ import annotations

import logging
from typing import Protocol

import networkx as nx

from codegraph.graph.community import Community
from codegraph.models import CodeEntity, EntityKind

logger = logging.getLogger(__name__)


class LLMBackend(Protocol):
    """Protocol for LLM backends used by the summarizer."""

    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""
        ...


class StubBackend:
    """
    Generates summaries without an LLM — just templates and entity names.

    This is surprisingly useful: a community containing `AuthService`,
    `validate_token`, `hash_password`, and `TokenExpiredError` is pretty
    obviously about authentication, even without asking a language model.
    """

    def generate(self, prompt: str) -> str:
        # The stub doesn't use the prompt — the summarizer constructs
        # the summary directly for this backend.
        return ""


def summarize_communities(
    graph: nx.DiGraph,
    communities: list[Community],
    backend: LLMBackend | None = None,
) -> list[Community]:
    """
    Generate natural language summaries for each community.

    Modifies the communities in-place (sets their `summary` field)
    and returns them for convenience.
    """
    if backend is None:
        backend = StubBackend()

    use_llm = not isinstance(backend, StubBackend)

    for community in communities:
        entities = _collect_entities(graph, community)
        if not entities:
            community.summary = "(empty community)"
            continue

        if use_llm:
            prompt = _build_summary_prompt(entities)
            try:
                community.summary = backend.generate(prompt)
            except Exception as e:
                logger.warning("LLM summarization failed for community %d: %s", community.id, e)
                community.summary = _build_stub_summary(entities)
        else:
            community.summary = _build_stub_summary(entities)

    return communities


def _collect_entities(graph: nx.DiGraph, community: Community) -> list[CodeEntity]:
    """Gather all CodeEntity objects for nodes in a community."""
    entities = []
    for entity_id in community.entity_ids:
        data = graph.nodes.get(entity_id)
        if data and data.get("entity"):
            entities.append(data["entity"])
    return entities


def _build_stub_summary(entities: list[CodeEntity]) -> str:
    """
    Build a summary purely from entity metadata — no LLM needed.

    Groups entities by kind and lists them. This gives a useful at-a-glance
    view of what a community contains.
    """
    by_kind: dict[EntityKind, list[str]] = {}
    files: set[str] = set()

    for entity in entities:
        by_kind.setdefault(entity.kind, []).append(entity.qualified_name)
        files.add(entity.file_path)

    parts = []
    parts.append(f"Contains {len(entities)} entities across {len(files)} file(s).")

    for kind in (EntityKind.MODULE, EntityKind.CLASS, EntityKind.FUNCTION, EntityKind.VARIABLE):
        names = by_kind.get(kind, [])
        if names:
            kind_label = kind.name.lower().replace("_", " ") + ("s" if len(names) > 1 else "")
            if len(names) <= 5:
                parts.append(f"  {kind_label}: {', '.join(names)}")
            else:
                shown = ", ".join(names[:5])
                parts.append(f"  {kind_label}: {shown}, ... (+{len(names) - 5} more)")

    return "\n".join(parts)


def _build_summary_prompt(entities: list[CodeEntity]) -> str:
    """
    Build an LLM prompt that asks for a concise community summary.

    We provide the entity names, types, docstrings, and file locations.
    The LLM's job is to synthesize this into a description of the module's
    purpose and responsibilities.
    """
    entity_descriptions = []
    for entity in entities[:30]:  # Cap at 30 to stay within context limits.
        desc = f"- {entity.kind.name} `{entity.qualified_name}` in {entity.file_path}"
        if entity.docstring:
            desc += f": {entity.docstring[:200]}"
        entity_descriptions.append(desc)

    return (
        "You are analyzing a software codebase. The following code entities "
        "were identified as a logical module (community) by graph clustering. "
        "Write a concise 2-3 sentence summary of what this module does, its "
        "purpose, and its key responsibilities.\n\n"
        "Entities in this module:\n"
        + "\n".join(entity_descriptions)
        + "\n\nSummary:"
    )
