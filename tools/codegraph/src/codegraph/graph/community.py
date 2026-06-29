"""
Community detection using the Leiden algorithm.

Identifies clusters of tightly-connected code entities — these correspond
to logical modules that might not align with the physical file/directory
structure. For example, a "user authentication" community might span
files across `auth/`, `middleware/`, and `models/`.

Why Leiden over Louvain? Leiden guarantees well-connected communities
(no disconnected subsets within a community), which matters for producing
coherent summaries. Louvain can produce communities where some nodes
have no internal connections — useless for understanding code structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import igraph as ig
import leidenalg
import networkx as nx

from codegraph.models import CodeEntity

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """
    A cluster of code entities that form a logical module.

    Attributes:
        id: Unique identifier for this community.
        entity_ids: The node IDs belonging to this community.
        summary: Natural language description (populated by the summarizer).
        level: Hierarchy level (0 = top-level, higher = more granular).
    """

    id: int
    entity_ids: list[str] = field(default_factory=list)
    summary: str = ""
    level: int = 0


def detect_communities(
    graph: nx.DiGraph,
    resolution: float = 1.0,
) -> list[Community]:
    """
    Run Leiden community detection on the code knowledge graph.

    The algorithm works on undirected graphs, so we convert the directed
    edges to undirected (a call from A→B implies A and B are related,
    regardless of direction).

    We filter out unresolved phantom nodes before detection — they'd just
    create noise.

    Args:
        graph: The code knowledge graph from GraphBuilder.
        resolution: Leiden resolution parameter. Higher values produce
            more, smaller communities. Default 1.0 is a good starting point.

    Returns:
        List of Community objects, one per detected cluster.
    """
    # Filter to only real nodes (skip phantoms).
    real_nodes = [
        node_id for node_id, data in graph.nodes(data=True)
        if data.get("entity") is not None
    ]

    if len(real_nodes) < 2:
        # Degenerate case — everything is one community.
        return [Community(id=0, entity_ids=real_nodes)]

    # Build an igraph Graph from the NetworkX subgraph.
    # Leiden operates on igraph, not networkx.
    subgraph = graph.subgraph(real_nodes)
    node_list = list(subgraph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    ig_graph = ig.Graph(directed=False)
    ig_graph.add_vertices(len(node_list))

    edges = set()
    for u, v in subgraph.edges():
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        if idx_u != idx_v:  # Skip self-loops.
            edge = (min(idx_u, idx_v), max(idx_u, idx_v))
            edges.add(edge)
    ig_graph.add_edges(list(edges))

    # Run Leiden.
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
    )

    # Convert partition to Community objects.
    communities: list[Community] = []
    for community_id, members in enumerate(partition):
        entity_ids = [node_list[idx] for idx in members]
        communities.append(Community(
            id=community_id,
            entity_ids=entity_ids,
        ))

    logger.info("Detected %d communities from %d nodes.", len(communities), len(real_nodes))
    return communities


def attach_communities_to_graph(
    graph: nx.DiGraph,
    communities: list[Community],
) -> None:
    """
    Store community assignments as node attributes on the graph.

    After this, you can access `graph.nodes[node_id]['community_id']`
    for any node.
    """
    for community in communities:
        for entity_id in community.entity_ids:
            if graph.has_node(entity_id):
                graph.nodes[entity_id]["community_id"] = community.id
                graph.nodes[entity_id]["community_summary"] = community.summary
