"""
Structural graph queries — the brain of CodeGraph.

This module does what LLMs *cannot* do: deterministic, exhaustive traversal
of the actual code dependency graph. When we say "X affects Y", we can
prove it by showing the exact edge path.

Key capabilities:
    - Deep impact analysis with categorized blast radius
    - Flow tracing: follows actual call chains through the graph
    - Ancestry/descendant exploration for inheritance hierarchies
    - Cross-file dependency mapping
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

import networkx as nx

from codegraph.models import CodeEntity, EntityKind, RelationshipKind


# ---------------------------------------------------------------------------
# Data structures for rich query results
# ---------------------------------------------------------------------------

@dataclass
class ImpactResult:
    """Detailed impact analysis with categorized affected entities."""

    source_id: str
    source_name: str
    source_file: str

    # Entities that DIRECTLY call/use/inherit from the source.
    direct: list[AffectedEntity] = field(default_factory=list)

    # Entities reachable 2+ hops away.
    transitive: list[AffectedEntity] = field(default_factory=list)

    # The full chain paths for the most critical impacts.
    critical_chains: list[list[str]] = field(default_factory=list)

    max_depth: int = 0
    total_affected: int = 0


@dataclass
class AffectedEntity:
    """An entity affected by a change, with context about HOW."""

    entity_id: str
    name: str
    file_path: str
    kind: str
    relationship: str  # How it's connected: "calls", "inherits", "imports"
    depth: int  # Hops from the source


@dataclass
class FlowStep:
    """One step in a traced execution flow."""

    entity_id: str
    name: str
    file_path: str
    kind: str
    relationship: str  # How we got here from the previous step
    line: int


@dataclass
class FlowTrace:
    """A complete execution flow through the codebase."""

    description: str
    steps: list[FlowStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity lookup helpers
# ---------------------------------------------------------------------------

def get_entity(graph: nx.DiGraph, entity_id: str) -> CodeEntity | None:
    """Retrieve the CodeEntity stored on a graph node."""
    data = graph.nodes.get(entity_id)
    if data:
        return data.get("entity")
    return None


def find_entity_by_name(graph: nx.DiGraph, name: str) -> list[str]:
    """
    Find all entity IDs whose name or qualified_name matches.

    Supports partial matching: "dispatch_request" will match
    "Flask.dispatch_request" and "View.dispatch_request".
    """
    results = []
    name_lower = name.lower()
    for node_id, data in graph.nodes(data=True):
        entity: CodeEntity | None = data.get("entity")
        if not entity:
            continue
        if (entity.name.lower() == name_lower
            or entity.qualified_name.lower() == name_lower
            or node_id.lower().endswith(f"::{name_lower}")
            or name_lower in entity.qualified_name.lower()):
            results.append(node_id)
    return results


# ---------------------------------------------------------------------------
# Deep impact analysis
# ---------------------------------------------------------------------------

def deep_impact_analysis(
    graph: nx.DiGraph,
    entity_id: str,
    max_depth: int = 6,
) -> ImpactResult:
    """
    Compute the FULL blast radius of changing an entity.

    Unlike shallow impact, this:
        1. Categorizes impacts into direct vs transitive.
        2. Records the relationship type (calls/inherits/imports) for each.
        3. Traces the most critical dependency chains.
        4. Identifies user-facing endpoints in the blast radius.

    This is what makes CodeGraph superior to asking an LLM: we follow
    EVERY edge in the actual dependency graph. An LLM guesses based on
    training data; we prove based on the real code.
    """
    source_entity = get_entity(graph, entity_id)
    source_name = source_entity.qualified_name if source_entity else entity_id
    source_file = source_entity.file_path if source_entity else ""

    result = ImpactResult(
        source_id=entity_id,
        source_name=source_name,
        source_file=source_file,
    )

    visited: set[str] = {entity_id}
    # BFS with depth tracking.
    frontier: list[tuple[str, int, str]] = []  # (node_id, depth, relationship)

    # Start with direct dependents (predecessors = things that depend on us).
    for pred in graph.predecessors(entity_id):
        edge_data = graph.edges[pred, entity_id]
        rel_kind = edge_data.get("kind")
        rel_name = rel_kind.name.lower() if hasattr(rel_kind, "name") else str(rel_kind)
        frontier.append((pred, 1, rel_name))

    # Also check successors for "contains" relationships going upward
    # (if a method changes, its containing class is affected).
    for succ in graph.successors(entity_id):
        edge_data = graph.edges[entity_id, succ]
        rel_kind = edge_data.get("kind")
        if rel_kind == RelationshipKind.CALLS:
            # If we call something, changes to us might break the call.
            rel_name = "called_by"
            frontier.append((succ, 1, rel_name))

    while frontier:
        node_id, depth, rel_name = frontier.pop(0)

        if node_id in visited:
            continue
        if depth > max_depth:
            continue
        visited.add(node_id)

        entity = get_entity(graph, node_id)
        if not entity:
            continue

        affected = AffectedEntity(
            entity_id=node_id,
            name=entity.qualified_name,
            file_path=entity.file_path,
            kind=entity.kind.name,
            relationship=rel_name,
            depth=depth,
        )

        if depth == 1:
            result.direct.append(affected)
        else:
            result.transitive.append(affected)

        # Continue BFS — walk to predecessors of this node.
        for pred in graph.predecessors(node_id):
            if pred not in visited:
                edge_data = graph.edges[pred, node_id]
                rel_kind = edge_data.get("kind")
                next_rel = rel_kind.name.lower() if hasattr(rel_kind, "name") else str(rel_kind)
                frontier.append((pred, depth + 1, next_rel))

    # Find critical chains — paths from most-affected back to source.
    result.critical_chains = _find_critical_chains(graph, entity_id, visited, max_chains=5)
    result.max_depth = max(
        (e.depth for e in result.direct + result.transitive),
        default=0,
    )
    result.total_affected = len(result.direct) + len(result.transitive)

    return result


def _find_critical_chains(
    graph: nx.DiGraph,
    source_id: str,
    affected_nodes: set[str],
    max_chains: int = 5,
) -> list[list[str]]:
    """
    Find the most interesting dependency chains from affected nodes
    back to the source. Prioritizes longer chains and chains ending
    at user-facing entities (modules, top-level functions).
    """
    chains: list[list[str]] = []
    undirected = graph.to_undirected()

    # Sort affected nodes by interest: modules and top-level functions first.
    interesting = []
    for node_id in affected_nodes:
        if node_id == source_id:
            continue
        entity = get_entity(graph, node_id)
        if not entity:
            continue
        # Prioritize: modules > classes > functions.
        priority = 0
        if entity.kind == EntityKind.MODULE:
            priority = 3
        elif entity.kind == EntityKind.CLASS:
            priority = 2
        elif entity.kind == EntityKind.FUNCTION and "." not in entity.qualified_name:
            priority = 1  # Top-level function.
        interesting.append((node_id, priority))

    interesting.sort(key=lambda x: x[1], reverse=True)

    for node_id, _ in interesting[:max_chains]:
        try:
            path = nx.shortest_path(undirected, source_id, node_id)
            if len(path) > 1:
                # Resolve to readable names.
                named_path = []
                for pid in path:
                    e = get_entity(graph, pid)
                    named_path.append(e.qualified_name if e else pid)
                chains.append(named_path)
        except nx.NetworkXNoPath:
            continue

    return chains


# ---------------------------------------------------------------------------
# Flow tracing — follow actual execution paths
# ---------------------------------------------------------------------------

def trace_execution_path(
    graph: nx.DiGraph,
    start_entity_id: str,
    max_depth: int = 10,
) -> FlowTrace:
    """
    Trace the EXECUTION path from a function — what does it actually call?

    Unlike trace_flow (which picks one best neighbor), this follows ALL
    CALLS edges to build a complete picture of the execution subtree.
    The result is a depth-first traversal of the call graph.

    This answers: "When Flask.__call__ runs, what functions execute?"
    Answer: __call__ → wsgi_app → full_dispatch_request → dispatch_request

    NOTE: This is a STATIC APPROXIMATION. Python's dynamic dispatch means
    some call targets may be resolved incorrectly (e.g., overridden methods).
    We clearly mark this as static analysis, not runtime profiling.
    """
    start_entity = get_entity(graph, start_entity_id)
    start_name = start_entity.qualified_name if start_entity else start_entity_id

    trace = FlowTrace(
        description=f"Execution path from {start_name} (static approximation)",
    )

    visited: set[str] = set()

    def _dfs(node_id: str, depth: int, rel_label: str) -> None:
        if node_id in visited or depth > max_depth:
            return
        visited.add(node_id)

        entity = get_entity(graph, node_id)
        if not entity:
            return

        trace.steps.append(FlowStep(
            entity_id=node_id,
            name=entity.qualified_name,
            file_path=entity.file_path,
            kind=entity.kind.name,
            relationship=rel_label,
            line=entity.start_line,
        ))

        # Follow CALLS edges only — these represent real function invocations.
        callees: list[tuple[str, CodeEntity]] = []
        for successor in graph.successors(node_id):
            edge = graph.edges[node_id, successor]
            if edge.get("kind") != RelationshipKind.CALLS:
                continue
            callee = get_entity(graph, successor)
            if callee and callee.kind == EntityKind.FUNCTION:
                callees.append((successor, callee))

        # Sort callees by line number so the trace follows source order.
        callees.sort(key=lambda x: x[1].start_line)

        for callee_id, _ in callees:
            _dfs(callee_id, depth + 1, "calls")

    _dfs(start_entity_id, 0, "entry")
    return trace


def trace_callers_chain(
    graph: nx.DiGraph,
    entity_id: str,
    max_depth: int = 8,
) -> FlowTrace:
    """
    Trace WHO calls this entity, recursively up the call chain.

    Answers: "Who ultimately triggers SecureCookieSessionInterface.save_session?"
    """
    entity = get_entity(graph, entity_id)
    name = entity.qualified_name if entity else entity_id

    trace = FlowTrace(description=f"Callers of {name}")

    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(entity_id, 0)]

    while queue:
        node_id, depth = queue.pop(0)
        if node_id in visited or depth > max_depth:
            continue
        visited.add(node_id)

        node_entity = get_entity(graph, node_id)
        if not node_entity:
            continue

        trace.steps.append(FlowStep(
            entity_id=node_id,
            name=node_entity.qualified_name,
            file_path=node_entity.file_path,
            kind=node_entity.kind.name,
            relationship="target" if depth == 0 else "called_by",
            line=node_entity.start_line,
        ))

        # Find all functions that CALL this entity.
        for predecessor in graph.predecessors(node_id):
            edge = graph.edges[predecessor, node_id]
            if edge.get("kind") == RelationshipKind.CALLS:
                queue.append((predecessor, depth + 1))

    return trace


def trace_flow(
    graph: nx.DiGraph,
    start_entity_id: str,
    direction: str = "forward",
    max_depth: int = 10,
    relationship_filter: set[RelationshipKind] | None = None,
) -> FlowTrace:
    """
    Trace a flow through the code graph (general-purpose).

    For execution-specific tracing, prefer trace_execution_path().
    """
    if relationship_filter is None:
        # Default: CALLS only for cleaner execution traces.
        relationship_filter = {RelationshipKind.CALLS}

    start_entity = get_entity(graph, start_entity_id)
    start_name = start_entity.qualified_name if start_entity else start_entity_id

    trace = FlowTrace(
        description=f"Flow from {start_name} ({direction})",
    )

    visited: set[str] = set()
    current = start_entity_id

    for _ in range(max_depth):
        if current in visited:
            break
        visited.add(current)

        entity = get_entity(graph, current)
        if not entity:
            break

        rel_name = "start" if not trace.steps else "→"
        trace.steps.append(FlowStep(
            entity_id=current,
            name=entity.qualified_name,
            file_path=entity.file_path,
            kind=entity.kind.name,
            relationship=rel_name,
            line=entity.start_line,
        ))

        if direction == "forward":
            neighbors = list(graph.successors(current))
        else:
            neighbors = list(graph.predecessors(current))

        best_next = None
        best_priority = -1
        for neighbor in neighbors:
            if neighbor in visited:
                continue

            if direction == "forward":
                edge_data = graph.edges[current, neighbor]
            else:
                edge_data = graph.edges[neighbor, current]

            rel_kind = edge_data.get("kind")
            if rel_kind not in relationship_filter:
                continue

            n_entity = get_entity(graph, neighbor)
            if not n_entity:
                continue

            priority = 0
            if rel_kind == RelationshipKind.CALLS:
                priority += 10
            if n_entity.kind == EntityKind.FUNCTION:
                priority += 5
            elif n_entity.kind == EntityKind.CLASS:
                priority += 3

            if priority > best_priority:
                best_priority = priority
                best_next = neighbor

        if best_next is None:
            break
        current = best_next

    return trace


def trace_all_paths(
    graph: nx.DiGraph,
    start_id: str,
    end_id: str,
    max_paths: int = 5,
    max_length: int = 8,
) -> list[FlowTrace]:
    """
    Find ALL paths between two entities in the graph.

    Returns multiple FlowTrace objects, each representing a different
    route through the codebase. This is powerful for understanding
    how two seemingly unrelated components are connected.
    """
    traces: list[FlowTrace] = []
    undirected = graph.to_undirected()

    try:
        # Find all simple paths up to max_length.
        all_paths = list(nx.all_simple_paths(
            undirected, start_id, end_id, cutoff=max_length,
        ))
    except nx.NodeNotFound:
        return traces

    # Sort by length (shorter = more direct = more interesting).
    all_paths.sort(key=len)

    for path in all_paths[:max_paths]:
        start_entity = get_entity(graph, start_id)
        end_entity = get_entity(graph, end_id)
        start_name = start_entity.qualified_name if start_entity else start_id
        end_name = end_entity.qualified_name if end_entity else end_id

        trace = FlowTrace(
            description=f"Path: {start_name} → {end_name} ({len(path) - 1} hops)",
        )
        for i, node_id in enumerate(path):
            entity = get_entity(graph, node_id)
            if not entity:
                continue

            if i == 0:
                rel = "start"
            else:
                # Determine the relationship from the edge.
                prev = path[i - 1]
                if graph.has_edge(prev, node_id):
                    edge = graph.edges[prev, node_id]
                elif graph.has_edge(node_id, prev):
                    edge = graph.edges[node_id, prev]
                else:
                    edge = {}
                rel_kind = edge.get("kind")
                rel = rel_kind.name.lower() if hasattr(rel_kind, "name") else "→"

            trace.steps.append(FlowStep(
                entity_id=node_id,
                name=entity.qualified_name,
                file_path=entity.file_path,
                kind=entity.kind.name,
                relationship=rel,
                line=entity.start_line,
            ))
        traces.append(trace)

    return traces


# ---------------------------------------------------------------------------
# Dependency analysis helpers
# ---------------------------------------------------------------------------

def get_dependents(
    graph: nx.DiGraph,
    entity_id: str,
    relationship_kinds: set[RelationshipKind] | None = None,
) -> list[str]:
    """Find all entities that directly depend on the given entity."""
    dependents = []
    for predecessor in graph.predecessors(entity_id):
        edge_data = graph.edges[predecessor, entity_id]
        if relationship_kinds is None or edge_data.get("kind") in relationship_kinds:
            dependents.append(predecessor)
    return dependents


def get_dependencies(
    graph: nx.DiGraph,
    entity_id: str,
    relationship_kinds: set[RelationshipKind] | None = None,
) -> list[str]:
    """Find all entities that the given entity depends on."""
    dependencies = []
    for successor in graph.successors(entity_id):
        edge_data = graph.edges[entity_id, successor]
        if relationship_kinds is None or edge_data.get("kind") in relationship_kinds:
            dependencies.append(successor)
    return dependencies


def get_shortest_path(
    graph: nx.DiGraph,
    source_id: str,
    target_id: str,
) -> list[str] | None:
    """Find the shortest dependency path between two entities."""
    undirected = graph.to_undirected()
    try:
        return nx.shortest_path(undirected, source_id, target_id)
    except nx.NetworkXNoPath:
        return None


def get_graph_stats(graph: nx.DiGraph) -> dict[str, int]:
    """Summary statistics about the knowledge graph."""
    entity_count_by_kind: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        entity: CodeEntity | None = data.get("entity")
        if entity:
            kind = entity.kind.name
            entity_count_by_kind[kind] = entity_count_by_kind.get(kind, 0) + 1

    edge_count_by_kind: dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        kind = data.get("kind")
        if kind:
            key = kind.name if hasattr(kind, "name") else str(kind)
            edge_count_by_kind[key] = edge_count_by_kind.get(key, 0) + 1

    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        **{f"nodes_{k}": v for k, v in entity_count_by_kind.items()},
        **{f"edges_{k}": v for k, v in edge_count_by_kind.items()},
    }


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------

def impact_analysis(
    graph: nx.DiGraph,
    entity_id: str,
    max_depth: int = 5,
) -> ImpactResult:
    """Wrapper for backward compatibility. Use deep_impact_analysis instead."""
    return deep_impact_analysis(graph, entity_id, max_depth)
