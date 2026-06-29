"""
Graph builder — constructs a NetworkX knowledge graph from parsed code.

This is where individual ParseResults (one per file) get stitched together
into a unified, cross-file knowledge graph. The key challenge here is
**relationship resolution**: parsers produce relationships with unresolved
target names (like "UserService" or "os.path"), and the builder needs to
match those to actual entity IDs.

The resolution strategy:
    1. Exact match — target name matches an entity's qualified_name.
    2. Suffix match — target name matches the tail of an entity's qualified_name
       (handles imports like `from auth import UserService` where the parser
       records just "UserService" as the callee).
    3. Unresolved — if neither works, we still create the edge but mark it
       with a special attribute. Better to have a dangling edge than to lose
       the signal entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import networkx as nx

from codegraph.models import CodeEntity, EntityKind, ParseResult, Relationship, RelationshipKind
from codegraph.parser.registry import ParserRegistry

logger = logging.getLogger(__name__)


# Files and directories to always skip during indexing.
IGNORE_PATTERNS = {
    "__pycache__", ".git", ".svn", "node_modules", ".venv", "venv",
    "env", ".env", "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "egg-info", ".eggs",
}


class GraphBuilder:
    """
    Builds a code knowledge graph from a project directory.

    Usage:
        builder = GraphBuilder()
        graph = builder.build(Path("/path/to/project"))

    The resulting graph is a `networkx.DiGraph` where:
        - Each node has an `entity` attribute containing the full CodeEntity.
        - Each edge has `kind`, `file_path`, and `line` attributes from the Relationship.
    """

    def __init__(self, registry: ParserRegistry | None = None) -> None:
        self.registry = registry or ParserRegistry()

    def build(self, project_root: Path) -> nx.DiGraph:
        """
        Index an entire project directory and return the knowledge graph.

        This is the main entry point. It:
            1. Discovers all parseable files.
            2. Parses each file individually.
            3. Merges all entities into a single graph.
            4. Resolves cross-file relationships.
        """
        graph = nx.DiGraph()
        all_results: list[ParseResult] = []

        # Phase 1: Parse all files.
        file_count = 0
        for file_path in self._discover_files(project_root):
            parser = self.registry.get_parser(file_path)
            if not parser:
                continue
            try:
                result = parser.parse_file(file_path, project_root)
                all_results.append(result)
                file_count += 1
            except Exception as e:
                # Don't let a single broken file kill the whole index.
                logger.warning("Failed to parse %s: %s", file_path, e)

        logger.info("Parsed %d files, extracted entities from %d results.", file_count, len(all_results))

        # Phase 2: Add all entities as nodes.
        for result in all_results:
            for entity in result.entities:
                graph.add_node(entity.id, entity=entity)

        # Phase 3: Resolve and add relationships as edges.
        name_index = self._build_name_index(graph)
        unresolved_count = 0

        for result in all_results:
            for rel in result.relationships:
                source_id = self._resolve_id(rel.source_id, graph, name_index)
                target_id = self._resolve_id(rel.target_id, graph, name_index)

                if source_id and graph.has_node(source_id):
                    if target_id and graph.has_node(target_id):
                        # For CALLS edges, redirect class references to __init__.
                        # When code says Cls(), the actual call is Cls.__init__().
                        if rel.kind == RelationshipKind.CALLS:
                            init_id = self._resolve_class_to_init(target_id, graph, name_index)
                            if init_id:
                                target_id = init_id

                        graph.add_edge(
                            source_id, target_id,
                            kind=rel.kind,
                            file_path=rel.file_path,
                            line=rel.line,
                        )
                    else:
                        # Unresolved target — still add it as a "phantom" node
                        # so we don't lose the outgoing edge.
                        phantom_id = f"__unresolved__::{rel.target_id}"
                        if not graph.has_node(phantom_id):
                            graph.add_node(phantom_id, entity=None, unresolved=True)
                        graph.add_edge(
                            source_id, phantom_id,
                            kind=rel.kind,
                            file_path=rel.file_path,
                            line=rel.line,
                            unresolved=True,
                        )
                        unresolved_count += 1

        logger.info(
            "Graph built: %d nodes, %d edges (%d unresolved).",
            graph.number_of_nodes(), graph.number_of_edges(), unresolved_count,
        )
        return graph

    # -------------------------------------------------------------------
    # File discovery
    # -------------------------------------------------------------------

    def _discover_files(self, root: Path) -> Iterator[Path]:
        """
        Recursively find all parseable files under the project root.

        Skips hidden directories, common build artifacts, and files that
        don't match any registered parser's extensions.
        """
        supported = self.registry.supported_extensions

        for path in sorted(root.rglob("*")):
            # Skip ignored directories.
            if any(part in IGNORE_PATTERNS for part in path.parts):
                continue
            # Skip hidden files/dirs.
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            # Only process files with supported extensions.
            if path.is_file() and path.suffix in supported:
                yield path

    # -------------------------------------------------------------------
    # Relationship resolution
    # -------------------------------------------------------------------

    def _build_name_index(self, graph: nx.DiGraph) -> dict[str, list[str]]:
        """
        Build a reverse index: short name → list of entity IDs.

        This speeds up the suffix-matching step in resolution. For example,
        if we have an entity "src/auth/service.py::AuthService.login", this
        index will contain entries for "login", "AuthService.login", etc.
        """
        index: dict[str, list[str]] = {}
        for node_id, data in graph.nodes(data=True):
            entity: CodeEntity | None = data.get("entity")
            if not entity:
                continue

            # Index by short name.
            index.setdefault(entity.name, []).append(node_id)

            # Index by qualified name.
            if entity.qualified_name != entity.name:
                index.setdefault(entity.qualified_name, []).append(node_id)

            # Index by name without "self." prefix (common in Python call sites).
            if "self." in entity.qualified_name:
                without_self = entity.qualified_name.replace("self.", "")
                index.setdefault(without_self, []).append(node_id)

        return index

    def _resolve_id(
        self,
        raw_id: str,
        graph: nx.DiGraph,
        name_index: dict[str, list[str]],
    ) -> str | None:
        """
        Try to resolve a raw ID (which might be a name, qualified name,
        or full entity ID) to an actual node in the graph.
        """
        # Already a valid node?
        if graph.has_node(raw_id):
            return raw_id

        # Strip "self." prefix for Python method calls.
        clean = raw_id.replace("self.", "")

        # Try the name index.
        candidates = name_index.get(clean, [])
        if len(candidates) == 1:
            return candidates[0]

        # Multiple candidates — try to disambiguate by file proximity.
        # (For now, just return the first one. A smarter heuristic would
        # consider import graphs.)
        if candidates:
            return candidates[0]

        # Try matching the last component of dotted names.
        # e.g., "os.path.join" → look for "join"
        if "." in clean:
            short = clean.rsplit(".", 1)[-1]
            candidates = name_index.get(short, [])
            if len(candidates) == 1:
                return candidates[0]

        return None

    def _resolve_class_to_init(
        self,
        target_id: str,
        graph: nx.DiGraph,
        name_index: dict[str, list[str]],
    ) -> str | None:
        """
        If a CALLS edge points to a CLASS entity, redirect it to __init__.

        When Python code says `obj = MyClass()`, the actual call goes to
        `MyClass.__init__`. This method checks if the resolved target is
        a class and, if so, looks for its __init__ method.
        """
        if not graph.has_node(target_id):
            return None

        data = graph.nodes[target_id]
        entity = data.get("entity")
        if not entity or entity.kind != EntityKind.CLASS:
            return None

        # Look for ClassName.__init__
        init_name = f"{entity.qualified_name}.__init__"
        candidates = name_index.get(init_name, [])
        if candidates:
            return candidates[0]

        # Also try just the __init__ name within this class's children
        for successor in graph.successors(target_id):
            succ_data = graph.nodes.get(successor, {})
            succ_entity = succ_data.get("entity")
            if succ_entity and succ_entity.name == "__init__":
                return successor

        return None

