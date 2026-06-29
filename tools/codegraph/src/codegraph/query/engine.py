"""
Query engine — graph reasoning, not keyword search.

The key insight: when someone asks "How does Flask handle sessions?",
they don't want a list of files with "session" in the name. They want
the actual execution flow:

    Request → Flask.wsgi_app → RequestContext → SessionInterface.open_session
    → SecureCookieSession → response.set_cookie

This module does graph-based reasoning to produce exactly that kind of
answer. It traces actual call paths, follows inheritance chains, and
maps complete execution flows.
"""

from __future__ import annotations

import logging
from enum import Enum, auto

import networkx as nx

from codegraph.graph.queries import (
    AffectedEntity,
    FlowTrace,
    deep_impact_analysis,
    find_entity_by_name,
    get_dependencies,
    get_dependents,
    get_entity,
    get_graph_stats,
    get_shortest_path,
    trace_execution_path,
    trace_callers_chain,
    trace_all_paths,
)
from codegraph.models import CodeEntity, EntityKind, RelationshipKind
from codegraph.retrieval.hybrid import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class QueryType(Enum):
    LOCAL = auto()
    GLOBAL = auto()
    IMPACT = auto()
    RELATIONSHIP = auto()
    FLOW = auto()  # NEW: trace execution flows


class QueryEngine:
    """
    Graph-reasoning query engine.

    Produces structured, PROVABLE answers by walking the actual dependency
    graph. Every claim is backed by a real edge in the code.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph
        self.retriever = HybridRetriever(graph)

    def query(self, question: str) -> str:
        query_type = self._classify(question)
        logger.info("Query classified as %s: %s", query_type.name, question)

        match query_type:
            case QueryType.IMPACT:
                return self._handle_impact(question)
            case QueryType.RELATIONSHIP:
                return self._handle_relationship(question)
            case QueryType.GLOBAL:
                return self._handle_global(question)
            case QueryType.FLOW:
                return self._handle_flow(question)
            case QueryType.LOCAL:
                return self._handle_local(question)

    # -------------------------------------------------------------------
    # Query classification
    # -------------------------------------------------------------------

    def _classify(self, question: str) -> QueryType:
        q = question.lower()

        impact_keywords = {
            "impact", "change", "break", "affect", "blast", "radius",
            "modify", "refactor", "影响", "修改", "重构", "改动",
        }
        flow_keywords = {
            "how does", "flow", "trace", "path", "happen", "process",
            "handle", "work", "step", "execute", "怎么", "流程", "处理",
        }
        relationship_keywords = {
            "connect", "between", "depend", "relationship", "关系", "路径",
        }
        global_keywords = {
            "architecture", "overview", "structure", "module", "all",
            "架构", "概览", "总体", "overall",
        }

        if any(k in q for k in impact_keywords):
            return QueryType.IMPACT
        if any(k in q for k in flow_keywords):
            return QueryType.FLOW
        if any(k in q for k in relationship_keywords):
            return QueryType.RELATIONSHIP
        if any(k in q for k in global_keywords):
            return QueryType.GLOBAL
        return QueryType.LOCAL

    # -------------------------------------------------------------------
    # FLOW handler — the star of the show
    # -------------------------------------------------------------------

    def _handle_flow(self, question: str) -> str:
        """
        Trace execution flows by following CALLS edges.

        This produces actual execution paths like:
            __call__ → wsgi_app → full_dispatch_request → dispatch_request

        NOT structural containment like:
            Flask → contains → Flask.wsgi_app

        All results are clearly labeled as STATIC APPROXIMATION.
        """
        entity_ids = self._find_mentioned_entities(question)

        if not entity_ids:
            results = self.retriever.retrieve(question, top_k=3)
            if results.items:
                entity_ids = [item.entity_id for item in results.items[:3]]

        if not entity_ids:
            return "Could not identify relevant entities to trace."

        # Deduplicate: prefer functions over classes/modules.
        seen_names: set[str] = set()
        deduped: list[str] = []
        for eid in entity_ids:
            e = get_entity(self.graph, eid)
            if e and e.qualified_name not in seen_names:
                seen_names.add(e.qualified_name)
                deduped.append(eid)

        # Prioritize functions (they have execution paths).
        funcs = [eid for eid in deduped if (e := get_entity(self.graph, eid)) and e.kind == EntityKind.FUNCTION]
        others = [eid for eid in deduped if eid not in funcs]
        ordered = (funcs + others)[:3]

        lines = ["⚠️  Note: All call chains are STATIC APPROXIMATIONS derived from AST analysis."]
        lines.append("    Dynamic dispatch, monkey-patching, and runtime imports are not captured.\n")

        for entity_id in ordered:
            entity = get_entity(self.graph, entity_id)
            if not entity:
                continue

            lines.append(f"{'━' * 60}")
            lines.append(f"  EXECUTION TRACE: {entity.qualified_name}")
            lines.append(f"  File: {entity.file_path}:{entity.start_line}")
            lines.append(f"{'━' * 60}\n")

            # Forward: what does this function call? (DFS through CALLS edges)
            exec_trace = trace_execution_path(self.graph, entity_id, max_depth=8)
            if len(exec_trace.steps) > 1:
                lines.append("  ▶ Execution call tree (functions invoked, source order):")
                for i, step in enumerate(exec_trace.steps):
                    indent = "    " + "  → " * i
                    lines.append(
                        f"{indent}{step.name} "
                        f"({step.file_path}:{step.line})"
                    )
                lines.append("")
            else:
                lines.append("  ▶ No outgoing calls detected from this entity.\n")

            # Backward: who calls this? (BFS through callers)
            callers = trace_callers_chain(self.graph, entity_id, max_depth=5)
            caller_steps = [s for s in callers.steps if s.relationship == "called_by"]
            if caller_steps:
                lines.append(f"  ◀ Called by ({len(caller_steps)} callers):")
                for step in caller_steps:
                    lines.append(
                        f"    ← {step.name} "
                        f"({step.file_path}:{step.line})"
                    )
                lines.append("")

            # Cross-entity paths if multiple entities mentioned.
            if len(ordered) >= 2:
                other_id = ordered[1] if entity_id == ordered[0] else ordered[0]
                paths = trace_all_paths(self.graph, entity_id, other_id, max_paths=3)
                if paths:
                    other_entity = get_entity(self.graph, other_id)
                    other_name = other_entity.qualified_name if other_entity else other_id
                    lines.append(f"  ⇄ Paths to {other_name}:")
                    for trace in paths:
                        path_str = " → ".join(s.name for s in trace.steps)
                        lines.append(f"    {path_str}")
                    lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # IMPACT handler — deep analysis
    # -------------------------------------------------------------------

    def _handle_impact(self, question: str) -> str:
        entity_ids = self._find_mentioned_entities(question)
        if not entity_ids:
            return "Could not identify which entity to analyze. Try: codegraph impact <name>"

        lines = []
        for entity_id in entity_ids[:2]:
            result = deep_impact_analysis(self.graph, entity_id, max_depth=6)

            lines.append(f"\n{'━' * 60}")
            lines.append(f"  💥 IMPACT ANALYSIS: {result.source_name}")
            lines.append(f"  File: {result.source_file}")
            lines.append(f"  Total blast radius: {result.total_affected} entities")
            lines.append(f"  Max propagation depth: {result.max_depth}")
            lines.append(f"{'━' * 60}\n")

            if result.direct:
                lines.append(f"  🔴 Direct dependents ({len(result.direct)}):")
                for affected in result.direct:
                    lines.append(
                        f"    [{affected.relationship}] {affected.name} "
                        f"({affected.file_path}) [{affected.kind}]"
                    )
                lines.append("")

            if result.transitive:
                lines.append(f"  🟡 Transitive impact ({len(result.transitive)}):")
                for affected in result.transitive[:15]:
                    depth_marker = "·" * affected.depth
                    lines.append(
                        f"    {depth_marker} [{affected.relationship}] {affected.name} "
                        f"({affected.file_path}) [{affected.kind}]"
                    )
                if len(result.transitive) > 15:
                    lines.append(f"    ... and {len(result.transitive) - 15} more")
                lines.append("")

            if result.critical_chains:
                lines.append("  🔗 Critical dependency chains:")
                for i, chain in enumerate(result.critical_chains):
                    chain_str = " → ".join(chain)
                    lines.append(f"    Chain {i + 1}: {chain_str}")
                lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # LOCAL handler — with structural context
    # -------------------------------------------------------------------

    def _handle_local(self, question: str) -> str:
        results: RetrievalResult = self.retriever.retrieve(question, top_k=5)

        if not results.items:
            return "No matching entities found for your query."

        lines = [f"Found {len(results.items)} relevant entities:\n"]
        for item in results.items[:8]:
            entity = item.entity
            lines.append(f"{'━' * 50}")
            lines.append(f"  {entity.kind.name}: {entity.qualified_name}")
            lines.append(f"  File: {entity.file_path}:{entity.start_line}-{entity.end_line}")
            if entity.docstring:
                doc = entity.docstring[:300]
                lines.append(f"  Doc: {doc}")

            deps = get_dependencies(self.graph, item.entity_id)
            dependents = get_dependents(self.graph, item.entity_id)
            if deps:
                dep_names = [self._short_name(d) for d in deps[:5]]
                lines.append(f"  Depends on: {', '.join(dep_names)}")
            if dependents:
                dep_names = [self._short_name(d) for d in dependents[:5]]
                lines.append(f"  Used by: {', '.join(dep_names)}")
            lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # GLOBAL handler
    # -------------------------------------------------------------------

    def _handle_global(self, question: str) -> str:
        stats = get_graph_stats(self.graph)

        lines = ["━━━ Codebase Architecture Overview ━━━\n"]
        lines.append(f"Total entities: {stats['total_nodes']}")
        lines.append(f"Total relationships: {stats['total_edges']}")
        lines.append("")

        lines.append("Entity breakdown:")
        for key, value in stats.items():
            if key.startswith("nodes_"):
                kind = key.replace("nodes_", "")
                lines.append(f"  {kind}: {value}")

        lines.append("")
        lines.append("Relationship breakdown:")
        for key, value in stats.items():
            if key.startswith("edges_"):
                kind = key.replace("edges_", "")
                lines.append(f"  {kind}: {value}")

        communities: dict[int, str] = {}
        for _, data in self.graph.nodes(data=True):
            cid = data.get("community_id")
            summary = data.get("community_summary", "")
            if cid is not None and summary and cid not in communities:
                communities[cid] = summary

        if communities:
            lines.append(f"\nDetected {len(communities)} logical modules:")
            for cid, summary in sorted(communities.items()):
                lines.append(f"\n  Module {cid}:")
                for line in summary.split("\n"):
                    lines.append(f"    {line}")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # RELATIONSHIP handler
    # -------------------------------------------------------------------

    def _handle_relationship(self, question: str) -> str:
        entity_ids = self._find_mentioned_entities(question)
        if len(entity_ids) < 2:
            return "Please mention two entities to find the relationship between them."

        source, target = entity_ids[0], entity_ids[1]
        paths = trace_all_paths(self.graph, source, target, max_paths=5)

        source_name = self._short_name(source)
        target_name = self._short_name(target)

        if not paths:
            return f"No connection found between {source_name} and {target_name}."

        lines = [f"Connections between {source_name} and {target_name}:\n"]
        for trace in paths:
            lines.append(f"  {trace.description}:")
            for step in trace.steps:
                lines.append(f"    [{step.relationship}] {step.name} ({step.file_path})")
            lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _find_mentioned_entities(self, question: str) -> list[str]:
        words = question.replace("?", "").replace(",", " ").replace(".", " ").split()
        found: list[str] = []

        skip_words = {
            "what", "how", "does", "the", "and", "for", "this", "that",
            "with", "from", "are", "was", "will", "can", "would", "could",
            "should", "about", "between", "impact", "change", "break",
            "affect", "handle", "work", "trace", "flow", "show",
        }

        for word in words:
            if len(word) <= 2 or word.lower() in skip_words:
                continue
            matches = find_entity_by_name(self.graph, word)
            found.extend(matches)

        # Also try multi-word matches (e.g., "dispatch_request").
        for i in range(len(words) - 1):
            compound = f"{words[i]}_{words[i+1]}"
            matches = find_entity_by_name(self.graph, compound)
            found.extend(matches)

        seen: set[str] = set()
        unique: list[str] = []
        for entity_id in found:
            if entity_id not in seen:
                seen.add(entity_id)
                unique.append(entity_id)

        return unique

    def _short_name(self, entity_id: str) -> str:
        entity = get_entity(self.graph, entity_id)
        if entity:
            return f"{entity.qualified_name} ({entity.file_path})"
        return entity_id
