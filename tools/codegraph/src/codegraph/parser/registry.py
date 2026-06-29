"""
Parser registry — maps file extensions to their language parsers.

Adding support for a new language is a two-step process:
    1. Write a parser class that implements `LanguageParser`.
    2. Register it here.

That's it. The rest of the system doesn't care what language it's dealing with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from codegraph.parser.base import LanguageParser
from codegraph.parser.python_parser import PythonParser
from codegraph.parser.typescript_parser import TypeScriptParser


class ParserRegistry:
    """
    Maps file extensions to language parsers.

    Lazily instantiates parsers — we don't load tree-sitter grammars
    until someone actually asks to parse a file of that type.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, LanguageParser] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the built-in parsers."""
        for parser_cls in (PythonParser, TypeScriptParser):
            parser = parser_cls()
            for ext in parser.file_extensions:
                self._parsers[ext] = parser

    def register(self, parser: LanguageParser) -> None:
        """Register a custom parser for its declared file extensions."""
        for ext in parser.file_extensions:
            self._parsers[ext] = parser

    def get_parser(self, file_path: Path) -> Optional[LanguageParser]:
        """
        Look up the parser for a given file.

        Returns None if the file extension isn't supported — the caller
        should skip unsupported files silently.
        """
        return self._parsers.get(file_path.suffix)

    @property
    def supported_extensions(self) -> set[str]:
        """All file extensions we can currently parse."""
        return set(self._parsers.keys())
