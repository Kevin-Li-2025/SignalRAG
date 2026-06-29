"""
Core data models for representing code structure.

These are the atoms of CodeGraph — every piece of code we analyze gets reduced
to entities (nodes) and relationships (edges) defined here. The rest of the
system operates on these types exclusively.

Design decisions:
    - Frozen dataclasses for immutability and hashability.
    - Entity IDs are deterministic: f"{file_path}::{qualified_name}" so the
      same codebase always produces the same graph.
    - Relationships are first-class objects (not just tuples) because we need
      to attach metadata like call-site line numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Entity kinds — what type of code construct are we looking at?
# ---------------------------------------------------------------------------

class EntityKind(Enum):
    """The structural role of a code entity within its language."""

    MODULE = auto()      # A file-level module (Python) or source file (TS)
    CLASS = auto()       # Class or interface definition
    FUNCTION = auto()    # Function or method definition
    VARIABLE = auto()    # Module-level constant or exported variable
    INTERFACE = auto()   # TypeScript interface (distinct from class)
    TYPE_ALIAS = auto()  # TypeScript type alias


# ---------------------------------------------------------------------------
# Relationship kinds — how do entities connect?
# ---------------------------------------------------------------------------

class RelationshipKind(Enum):
    """
    The type of dependency between two code entities.

    Each variant maps to a concrete, observable code pattern — never inferred
    by heuristics or LLMs. If we can't prove the relationship exists from the
    AST alone, we don't create it.
    """

    IMPORTS = auto()     # A imports B (Python import / TS import)
    CALLS = auto()       # A calls B (function/method invocation)
    INHERITS = auto()    # A extends/inherits from B
    IMPLEMENTS = auto()  # A implements interface B (TypeScript)
    CONTAINS = auto()    # A contains B (module contains class, class contains method)
    OVERRIDES = auto()   # A.method overrides B.method
    USES_TYPE = auto()   # A references B as a type annotation


# ---------------------------------------------------------------------------
# Code entities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodeEntity:
    """
    A single identifiable construct in source code.

    Attributes:
        id: Deterministic identifier — "{file_path}::{qualified_name}".
            Example: "src/auth/service.py::AuthService.login"
        name: Short name as written in source (e.g., "login").
        qualified_name: Dot-separated full path within the file
            (e.g., "AuthService.login").
        kind: What type of construct this is.
        file_path: Path to the source file, relative to the project root.
        start_line: First line of the definition (1-indexed).
        end_line: Last line of the definition (1-indexed).
        source_code: The raw source text of this entity.
        docstring: Extracted docstring / JSDoc, if any.
        language: Programming language ("python", "typescript").
    """

    id: str
    name: str
    qualified_name: str
    kind: EntityKind
    file_path: str
    start_line: int
    end_line: int
    source_code: str
    docstring: Optional[str] = None
    language: str = "python"


@dataclass(frozen=True)
class FunctionEntity(CodeEntity):
    """A function or method, with signature details."""

    parameters: tuple[str, ...] = ()
    return_type: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    decorators: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClassEntity(CodeEntity):
    """A class definition, with inheritance and member info."""

    bases: tuple[str, ...] = ()
    method_names: tuple[str, ...] = ()
    attribute_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModuleEntity(CodeEntity):
    """A file-level module."""

    imports: tuple[str, ...] = ()
    exports: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Relationship:
    """
    A directed edge between two code entities.

    Attributes:
        source_id: ID of the entity where the relationship originates.
        target_id: ID of the entity being referenced.
        kind: The type of relationship.
        file_path: Where in the source this relationship is expressed.
        line: Line number of the reference (e.g., the call site).
    """

    source_id: str
    target_id: str
    kind: RelationshipKind
    file_path: str
    line: int


# ---------------------------------------------------------------------------
# Container for a fully parsed file
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """
    Everything we extracted from a single source file.

    This is the output of a parser — one per file — before any cross-file
    resolution happens. The graph builder consumes these to construct the
    full knowledge graph.
    """

    file_path: str
    entities: list[CodeEntity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
