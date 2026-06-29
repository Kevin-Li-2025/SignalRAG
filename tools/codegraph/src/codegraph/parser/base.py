"""
Abstract interface for language-specific parsers.

Each supported language implements this protocol. The contract is simple:
given a file path, produce a ParseResult containing all entities and
intra-file relationships. Cross-file resolution happens later in the
graph builder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from codegraph.models import ParseResult


class LanguageParser(ABC):
    """
    Base class for all language parsers.

    Subclasses must implement `parse_file` which takes a source file and
    returns a ParseResult. The parser should extract every structurally
    significant entity (functions, classes, modules) and every relationship
    it can prove exists from the AST alone.
    """

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Human-readable name of the language (e.g., 'python')."""
        ...

    @property
    @abstractmethod
    def file_extensions(self) -> tuple[str, ...]:
        """File extensions this parser handles (e.g., ('.py',))."""
        ...

    @abstractmethod
    def parse_file(self, file_path: Path, project_root: Path) -> ParseResult:
        """
        Parse a single source file and extract its structure.

        Args:
            file_path: Absolute path to the file.
            project_root: Root directory of the project (for relative paths).

        Returns:
            A ParseResult containing all entities and relationships found.
        """
        ...
