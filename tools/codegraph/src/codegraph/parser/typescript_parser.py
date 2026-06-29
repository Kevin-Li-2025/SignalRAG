"""
TypeScript / JavaScript parser using tree-sitter.

Follows the same pattern as the Python parser: deterministic AST walking
to extract typed entities and structural relationships.

TypeScript adds a few constructs Python doesn't have:
    - Interfaces (treated as a distinct EntityKind)
    - Type aliases
    - `export` declarations (tracked on the module entity)
    - Optional chaining in call expressions

The tree-sitter grammar for TypeScript is used for both .ts and .tsx files.
Plain .js and .jsx files are parsed with the TypeScript grammar too — it's
a superset, so this works correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser, Node

from codegraph.models import (
    ClassEntity,
    CodeEntity,
    EntityKind,
    FunctionEntity,
    ModuleEntity,
    ParseResult,
    Relationship,
    RelationshipKind,
)
from codegraph.parser.base import LanguageParser


_TS_LANGUAGE = Language(tstypescript.language_typescript())


def _get_text(node: Node) -> str:
    return node.text.decode("utf-8") if node.text else ""


def _build_entity_id(file_path: str, qualified_name: str) -> str:
    return f"{file_path}::{qualified_name}"


def _get_jsdoc(node: Node) -> Optional[str]:
    """
    Extract JSDoc comment preceding a declaration.

    In tree-sitter's TypeScript grammar, comments are siblings, not children.
    We look for a comment node immediately before this declaration.
    """
    prev = node.prev_sibling
    if prev and prev.type == "comment":
        text = _get_text(prev)
        # Strip /** ... */ markers.
        text = text.strip("/").strip("*").strip()
        return text if text else None
    return None


class TypeScriptParser(LanguageParser):
    """
    Extracts structural information from TypeScript / JavaScript files.

    Handles:
        - Function declarations and arrow functions assigned to variables.
        - Class declarations with heritage clauses (extends / implements).
        - Interface and type alias declarations.
        - Import / export statements.
        - Method definitions within classes.
        - Call expression detection.
    """

    @property
    def language_name(self) -> str:
        return "typescript"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".ts", ".tsx", ".js", ".jsx")

    def parse_file(self, file_path: Path, project_root: Path) -> ParseResult:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        relative_path = str(file_path.relative_to(project_root))

        parser = Parser(_TS_LANGUAGE)
        tree = parser.parse(source.encode("utf-8"))

        entities: list[CodeEntity] = []
        relationships: list[Relationship] = []

        # Module entity.
        module_name = file_path.stem
        module_id = _build_entity_id(relative_path, module_name)
        imports = self._extract_imports(tree.root_node)
        exports = self._extract_exports(tree.root_node)

        module_entity = ModuleEntity(
            id=module_id,
            name=module_name,
            qualified_name=module_name,
            kind=EntityKind.MODULE,
            file_path=relative_path,
            start_line=1,
            end_line=source.count("\n") + 1,
            source_code="",
            language="typescript",
            imports=tuple(imports),
            exports=tuple(exports),
        )
        entities.append(module_entity)

        # Walk top-level declarations.
        self._walk_scope(
            tree.root_node, "", relative_path, module_id,
            entities, relationships,
        )

        # Import relationships.
        for imp in imports:
            relationships.append(Relationship(
                source_id=module_id,
                target_id=imp,
                kind=RelationshipKind.IMPORTS,
                file_path=relative_path,
                line=1,
            ))

        # Call extraction.
        self._extract_calls(
            tree.root_node, "", relative_path,
            entities, relationships,
        )

        return ParseResult(
            file_path=relative_path,
            entities=entities,
            relationships=relationships,
        )

    # -------------------------------------------------------------------
    # Scope walking
    # -------------------------------------------------------------------

    def _walk_scope(
        self,
        node: Node,
        scope_prefix: str,
        file_path: str,
        parent_id: str,
        entities: list[CodeEntity],
        relationships: list[Relationship],
    ) -> None:
        for child in node.children:
            match child.type:
                case "function_declaration":
                    self._extract_function(
                        child, scope_prefix, file_path, parent_id,
                        entities, relationships,
                    )
                case "class_declaration":
                    self._extract_class(
                        child, scope_prefix, file_path, parent_id,
                        entities, relationships,
                    )
                case "interface_declaration":
                    self._extract_interface(
                        child, scope_prefix, file_path, parent_id,
                        entities, relationships,
                    )
                case "type_alias_declaration":
                    self._extract_type_alias(
                        child, scope_prefix, file_path, parent_id,
                        entities, relationships,
                    )
                case "lexical_declaration" | "variable_declaration":
                    # Could be `const foo = () => { ... }` — an arrow function.
                    self._extract_variable_declarations(
                        child, scope_prefix, file_path, parent_id,
                        entities, relationships,
                    )
                case "export_statement":
                    # `export function ...` / `export class ...` etc.
                    self._walk_scope(
                        child, scope_prefix, file_path, parent_id,
                        entities, relationships,
                    )

    def _extract_function(
        self,
        node: Node,
        scope_prefix: str,
        file_path: str,
        parent_id: str,
        entities: list[CodeEntity],
        relationships: list[Relationship],
        is_method: bool = False,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = _get_text(name_node)
        qualified_name = f"{scope_prefix}.{name}" if scope_prefix else name
        entity_id = _build_entity_id(file_path, qualified_name)

        params_node = node.child_by_field_name("parameters")
        parameters = self._extract_params(params_node) if params_node else ()

        return_node = node.child_by_field_name("return_type")
        return_type = _get_text(return_node).lstrip(":").strip() if return_node else None

        is_async = _get_text(node).startswith("async")
        docstring = _get_jsdoc(node)

        entity = FunctionEntity(
            id=entity_id,
            name=name,
            qualified_name=qualified_name,
            kind=EntityKind.FUNCTION,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            source_code=_get_text(node),
            docstring=docstring,
            language="typescript",
            parameters=parameters,
            return_type=return_type,
            is_async=is_async,
            is_method=is_method,
        )
        entities.append(entity)

        relationships.append(Relationship(
            source_id=parent_id,
            target_id=entity_id,
            kind=RelationshipKind.CONTAINS,
            file_path=file_path,
            line=node.start_point[0] + 1,
        ))

        # Recurse into function body.
        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk_scope(
                body_node, qualified_name, file_path, entity_id,
                entities, relationships,
            )

    def _extract_class(
        self,
        node: Node,
        scope_prefix: str,
        file_path: str,
        parent_id: str,
        entities: list[CodeEntity],
        relationships: list[Relationship],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = _get_text(name_node)
        qualified_name = f"{scope_prefix}.{name}" if scope_prefix else name
        entity_id = _build_entity_id(file_path, qualified_name)

        bases, implements = self._extract_heritage(node)
        docstring = _get_jsdoc(node)

        entity = ClassEntity(
            id=entity_id,
            name=name,
            qualified_name=qualified_name,
            kind=EntityKind.CLASS,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            source_code=_get_text(node),
            docstring=docstring,
            language="typescript",
            bases=bases,
        )
        entities.append(entity)

        relationships.append(Relationship(
            source_id=parent_id,
            target_id=entity_id,
            kind=RelationshipKind.CONTAINS,
            file_path=file_path,
            line=node.start_point[0] + 1,
        ))

        for base in bases:
            relationships.append(Relationship(
                source_id=entity_id,
                target_id=base,
                kind=RelationshipKind.INHERITS,
                file_path=file_path,
                line=node.start_point[0] + 1,
            ))
        for iface in implements:
            relationships.append(Relationship(
                source_id=entity_id,
                target_id=iface,
                kind=RelationshipKind.IMPLEMENTS,
                file_path=file_path,
                line=node.start_point[0] + 1,
            ))

        # Walk class body for methods.
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type in ("method_definition", "public_field_definition"):
                    if child.type == "method_definition":
                        self._extract_function(
                            child, qualified_name, file_path, entity_id,
                            entities, relationships, is_method=True,
                        )

    def _extract_interface(
        self,
        node: Node,
        scope_prefix: str,
        file_path: str,
        parent_id: str,
        entities: list[CodeEntity],
        relationships: list[Relationship],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = _get_text(name_node)
        qualified_name = f"{scope_prefix}.{name}" if scope_prefix else name
        entity_id = _build_entity_id(file_path, qualified_name)

        entity = CodeEntity(
            id=entity_id,
            name=name,
            qualified_name=qualified_name,
            kind=EntityKind.INTERFACE,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            source_code=_get_text(node),
            docstring=_get_jsdoc(node),
            language="typescript",
        )
        entities.append(entity)

        relationships.append(Relationship(
            source_id=parent_id,
            target_id=entity_id,
            kind=RelationshipKind.CONTAINS,
            file_path=file_path,
            line=node.start_point[0] + 1,
        ))

    def _extract_type_alias(
        self,
        node: Node,
        scope_prefix: str,
        file_path: str,
        parent_id: str,
        entities: list[CodeEntity],
        relationships: list[Relationship],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = _get_text(name_node)
        qualified_name = f"{scope_prefix}.{name}" if scope_prefix else name
        entity_id = _build_entity_id(file_path, qualified_name)

        entity = CodeEntity(
            id=entity_id,
            name=name,
            qualified_name=qualified_name,
            kind=EntityKind.TYPE_ALIAS,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            source_code=_get_text(node),
            docstring=_get_jsdoc(node),
            language="typescript",
        )
        entities.append(entity)

        relationships.append(Relationship(
            source_id=parent_id,
            target_id=entity_id,
            kind=RelationshipKind.CONTAINS,
            file_path=file_path,
            line=node.start_point[0] + 1,
        ))

    def _extract_variable_declarations(
        self,
        node: Node,
        scope_prefix: str,
        file_path: str,
        parent_id: str,
        entities: list[CodeEntity],
        relationships: list[Relationship],
    ) -> None:
        """
        Handle `const foo = () => { ... }` — arrow functions stored in variables.

        We only promote these to FunctionEntity if the initializer is an arrow
        function or function expression. Plain variable assignments become
        VARIABLE entities.
        """
        for child in node.children:
            if child.type != "variable_declarator":
                continue
            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if not name_node:
                continue

            name = _get_text(name_node)
            qualified_name = f"{scope_prefix}.{name}" if scope_prefix else name
            entity_id = _build_entity_id(file_path, qualified_name)

            if value_node and value_node.type in ("arrow_function", "function"):
                # It's an arrow function — treat as a function entity.
                params_node = value_node.child_by_field_name("parameters")
                parameters = self._extract_params(params_node) if params_node else ()

                entity = FunctionEntity(
                    id=entity_id,
                    name=name,
                    qualified_name=qualified_name,
                    kind=EntityKind.FUNCTION,
                    file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    source_code=_get_text(child),
                    docstring=_get_jsdoc(node),
                    language="typescript",
                    parameters=parameters,
                    is_async=_get_text(value_node).startswith("async"),
                )
                entities.append(entity)
            else:
                # Plain variable.
                entity = CodeEntity(
                    id=entity_id,
                    name=name,
                    qualified_name=qualified_name,
                    kind=EntityKind.VARIABLE,
                    file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    source_code=_get_text(child),
                    language="typescript",
                )
                entities.append(entity)

            relationships.append(Relationship(
                source_id=parent_id,
                target_id=entity_id,
                kind=RelationshipKind.CONTAINS,
                file_path=file_path,
                line=child.start_point[0] + 1,
            ))

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _extract_imports(self, root: Node) -> list[str]:
        imports: list[str] = []
        for child in root.children:
            if child.type == "import_statement":
                source_node = child.child_by_field_name("source")
                if source_node:
                    path = _get_text(source_node).strip("'\"")
                    imports.append(path)
        return imports

    def _extract_exports(self, root: Node) -> list[str]:
        exports: list[str] = []
        for child in root.children:
            if child.type == "export_statement":
                for sub in child.children:
                    if sub.type in ("function_declaration", "class_declaration"):
                        name_node = sub.child_by_field_name("name")
                        if name_node:
                            exports.append(_get_text(name_node))
                    elif sub.type in ("lexical_declaration", "variable_declaration"):
                        for vd in sub.children:
                            if vd.type == "variable_declarator":
                                name_node = vd.child_by_field_name("name")
                                if name_node:
                                    exports.append(_get_text(name_node))
        return exports

    def _extract_params(self, params_node: Node) -> tuple[str, ...]:
        params: list[str] = []
        for child in params_node.children:
            if child.type in ("required_parameter", "optional_parameter"):
                pattern_node = child.child_by_field_name("pattern")
                if pattern_node:
                    params.append(_get_text(pattern_node))
            elif child.type == "identifier":
                params.append(_get_text(child))
        return tuple(params)

    def _extract_heritage(self, class_node: Node) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Extract extends and implements clauses.

        Returns:
            (bases, implements) — both as tuples of names.
        """
        bases: list[str] = []
        implements: list[str] = []

        for child in class_node.children:
            if child.type == "class_heritage":
                for clause in child.children:
                    text = _get_text(clause)
                    if "extends" in text:
                        # Pull out the type names after "extends".
                        for sub in clause.children:
                            if sub.type in ("identifier", "type_identifier"):
                                bases.append(_get_text(sub))
                    elif "implements" in text:
                        for sub in clause.children:
                            if sub.type in ("identifier", "type_identifier"):
                                implements.append(_get_text(sub))

        return tuple(bases), tuple(implements)

    # -------------------------------------------------------------------
    # Call extraction
    # -------------------------------------------------------------------

    def _extract_calls(
        self,
        node: Node,
        scope_prefix: str,
        file_path: str,
        entities: list[CodeEntity],
        relationships: list[Relationship],
    ) -> None:
        scope_to_id: dict[str, str] = {}
        for entity in entities:
            if entity.kind == EntityKind.FUNCTION:
                scope_to_id[entity.qualified_name] = entity.id

        self._walk_calls(node, scope_prefix, file_path, scope_to_id, relationships)

    def _walk_calls(
        self,
        node: Node,
        current_scope: str,
        file_path: str,
        scope_to_id: dict[str, str],
        relationships: list[Relationship],
    ) -> None:
        if node.type in ("function_declaration", "method_definition", "arrow_function"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _get_text(name_node)
                current_scope = f"{current_scope}.{name}" if current_scope else name

        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                callee = _get_text(func_node)
                caller_id = scope_to_id.get(current_scope)
                if callee and caller_id:
                    relationships.append(Relationship(
                        source_id=caller_id,
                        target_id=callee,
                        kind=RelationshipKind.CALLS,
                        file_path=file_path,
                        line=node.start_point[0] + 1,
                    ))

        for child in node.children:
            self._walk_calls(child, current_scope, file_path, scope_to_id, relationships)
