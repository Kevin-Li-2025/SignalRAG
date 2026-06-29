"""
Hybrid retrieval — the core innovation of CodeGraph.

Traditional RAG does vector search over chunks. GraphRAG does community
summaries. We do both AND leverage the graph structure:

    1. **Semantic search** — find entities whose source/docstring is
       semantically similar to the query (vector similarity).
    2. **Graph expansion** — from those seed entities, walk N hops in the
       knowledge graph to gather structural context (callers, callees,
       parent classes, sibling methods).
    3. **Community context** — attach the community summary for each seed
       entity, giving the LLM a higher-level understanding of the module.

The result is a context window that includes both the specific code the
user asked about AND the architectural context surrounding it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from codegraph.models import CodeEntity, EntityKind

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    A ranked set of context items for answering a query.

    Each item includes the entity, its relevance score, and its structural
    context (neighbors in the graph, community summary).
    """

    items: list[ContextItem] = field(default_factory=list)


@dataclass
class ContextItem:
    """A single piece of context retrieved for a query."""

    entity_id: str
    entity: CodeEntity
    score: float
    source: str  # "semantic", "graph_expansion", or "community"
    community_summary: str = ""
    neighbors: list[str] = field(default_factory=list)


class HybridRetriever:
    """
    Combines vector search, graph traversal, and community summaries.

    Usage:
        retriever = HybridRetriever(graph, embeddings)
        results = retriever.retrieve("authentication logic", top_k=10)
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.graph = graph
        self.embeddings = embeddings or {}

        # Pre-compute entity texts for keyword fallback search.
        self._entity_texts: dict[str, str] = {}
        for node_id, data in graph.nodes(data=True):
            entity: CodeEntity | None = data.get("entity")
            if entity:
                # Combine name, docstring, and file path into a searchable text.
                parts = [
                    entity.qualified_name,
                    entity.file_path,
                    entity.docstring or "",
                    entity.kind.name.lower(),
                ]
                self._entity_texts[node_id] = " ".join(parts).lower()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        expansion_hops: int = 2,
    ) -> RetrievalResult:
        """
        Retrieve context for a natural language query.

        Steps:
            1. Find seed entities via semantic/keyword search.
            2. Expand seeds by walking the graph.
            3. Attach community summaries.
            4. Rank and deduplicate.
        """
        # Step 1: Seed discovery.
        if self.embeddings:
            seeds = self._semantic_search(query, top_k=top_k)
        else:
            seeds = self._keyword_search(query, top_k=top_k)

        # Step 2: Graph expansion.
        expanded = self._expand_seeds(seeds, hops=expansion_hops)

        # Step 3: Assemble context items.
        items: list[ContextItem] = []
        seen: set[str] = set()

        for entity_id, score in seeds:
            if entity_id in seen:
                continue
            seen.add(entity_id)
            entity = self._get_entity(entity_id)
            if not entity:
                continue

            neighbors = expanded.get(entity_id, [])
            community_summary = self.graph.nodes[entity_id].get("community_summary", "")

            items.append(ContextItem(
                entity_id=entity_id,
                entity=entity,
                score=score,
                source="semantic" if self.embeddings else "keyword",
                community_summary=community_summary,
                neighbors=neighbors,
            ))

        # Add neighbors as lower-scored context.
        for entity_id, neighbor_ids in expanded.items():
            for neighbor_id in neighbor_ids:
                if neighbor_id in seen:
                    continue
                seen.add(neighbor_id)
                entity = self._get_entity(neighbor_id)
                if not entity:
                    continue
                items.append(ContextItem(
                    entity_id=neighbor_id,
                    entity=entity,
                    score=0.3,  # Lower score for graph-expanded items.
                    source="graph_expansion",
                    community_summary=self.graph.nodes[neighbor_id].get("community_summary", ""),
                ))

        # Sort by score descending, cap at top_k * 2 (seeds + expansion).
        items.sort(key=lambda x: x.score, reverse=True)
        return RetrievalResult(items=items[: top_k * 2])

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Simple keyword-based search as a fallback when no embeddings exist.

        Scores entities by how many query terms appear in their searchable text.
        Crude but effective for structural queries like "authentication service".
        """
        query_terms = query.lower().split()
        scores: list[tuple[str, float]] = []

        for entity_id, text in self._entity_texts.items():
            # Count matching terms.
            matches = sum(1 for term in query_terms if term in text)
            if matches > 0:
                score = matches / len(query_terms)
                scores.append((entity_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _semantic_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Vector similarity search over entity embeddings.

        Requires that embeddings have been pre-computed and passed to the
        constructor. Falls back to keyword search if embeddings are empty.
        """
        # This is a placeholder — the actual implementation would need
        # a query embedding from the same model used for entity embeddings.
        # For now, we delegate to keyword search.
        logger.info("Semantic search not yet implemented, falling back to keyword search.")
        return self._keyword_search(query, top_k)

    def _expand_seeds(
        self,
        seeds: list[tuple[str, float]],
        hops: int,
    ) -> dict[str, list[str]]:
        """
        Walk the graph outward from each seed entity.

        Returns a dict of seed → list of neighbor entity IDs found within
        `hops` steps. This gives the query engine structural context beyond
        just the matching entity.
        """
        expansion: dict[str, list[str]] = {}

        for entity_id, _ in seeds:
            if not self.graph.has_node(entity_id):
                continue

            visited: set[str] = {entity_id}
            frontier = {entity_id}
            neighbors: list[str] = []

            for _ in range(hops):
                next_frontier: set[str] = set()
                for node in frontier:
                    # Walk both directions — dependencies and dependents.
                    for neighbor in list(self.graph.successors(node)) + list(self.graph.predecessors(node)):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            neighbors.append(neighbor)
                            next_frontier.add(neighbor)
                frontier = next_frontier

            expansion[entity_id] = neighbors

        return expansion

    def _get_entity(self, entity_id: str) -> CodeEntity | None:
        data = self.graph.nodes.get(entity_id)
        return data.get("entity") if data else None
