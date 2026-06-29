"""
Python source code parser using tree-sitter.

Walks the AST to extract functions, classes, imports, and their relationships.
This is deterministic — given the same source file, it always produces the
exact same entities and relationships. No LLM involved, no guessing.

The tree-sitter queries here are intentionally explicit rather than using the
query DSL, because we need fine-grained control over how nested structures
(methods inside classes, classes inside functions) get their qualified names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import tree_sitter_python as tspython
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


# Cache the language object — loading the grammar is expensive.
_PY_LANGUAGE = Language(tspython.language())


def _get_text(node: Node) -> str:
    """Extract the UTF-8 text content of a tree-sitter node."""
    return node.text.decode("utf-8") if node.text else ""


def _get_docstring(body_node: Node) -> Optional[str]:
    """
    Extract the docstring from a function or class body.

    In Python, a docstring is the first statement in the body if it's a
    string expression. We look for `expression_statement > string`.
    """
    if body_node.type != "block":
        return None

    for child in body_node.children:
        # Skip comments and whitespace-only nodes.
        if child.type in ("comment", "newline", "indent", "dedent"):
            continue
        # First real statement — is it a docstring?
        if child.type == "expression_statement" and child.child_count > 0:
            expr = child.children[0]
            if expr.type in ("string", "concatenated_string"):
                raw = _get_text(expr)
                # Strip the quote characters.
                return raw.strip('"""').strip("'''").strip('"').strip("'").strip()
        break  # First non-comment statement isn't a string → no docstring.

    return None


def _build_entity_id(file_path: str, qualified_name: str) -> str:
    """Deterministic entity ID: file path + qualified name."""
    return f"{file_path}::{qualified_name}"


class PythonParser(LanguageParser):
    """
    Extracts structural information from Python source files.

    Handles:
        - Module-level structure (imports, top-level functions/classes).
        - Class bodies (methods, attributes, base classes).
        - Nested functions and closures (with correct qualified names).
        - Decorator extraction.
        - Type annotation extraction (best-effort, from AST text).
        - Call-site detection for function/method invocations.
    """

    @property
    def language_name(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".py",)

    def parse_file(self, file_path: Path, project_root: Path) -> ParseResult:
        """Parse a Python file into entities and relationships."""
        source = file_path.read_text(encoding="utf-8", errors="replace")
        relative_path = str(file_path.relative_to(project_root))

        parser = Parser(_PY_LANGUAGE)
        tree = parser.parse(source.encode("utf-8"))

        entities: list[CodeEntity] = []
        relationships: list[Relationship] = []

        # The module itself is an entity.
        module_name = file_path.stem
        module_id = _build_entity_id(relative_path, module_name)
        imports = self._extract_imports(tree.root_node)

        module_entity = ModuleEntity(
            id=module_id,
            name=module_name,
            qualified_name=module_name,
            kind=EntityKind.MODULE,
            file_path=relative_path,
            start_line=1,
            end_line=source.count("\n") + 1,
            source_code="",  # Don't store the full file — too large.
            language="python",
            imports=tuple(imports),
        )
        entities.append(module_entity)

        # Walk top-level children to find classes, functions, assignments.
        self._walk_scope(
            node=tree.root_node,
            scope_prefix="",
            file_path=relative_path,
            module_id=module_id,
            entities=entities,
            relationships=relationships,
            source=source,
        )

        # Extract call-site relationships.
        self._extract_calls(
            node=tree.root_node,
            scope_prefix=module_name,
            file_path=relative_path,
            entities=entities,
            relationships=relationships,
        )

        # Create import relationships (module → module).
        for imp in imports:
            relationships.append(Relationship(
                source_id=module_id,
                target_id=imp,  # Will be resolved to actual entity ID later.
                kind=RelationshipKind.IMPORTS,
                file_path=relative_path,
                line=1,  # Simplified — real line tracked below if needed.
            ))

        return ParseResult(
            file_path=relative_path,
            entities=entities,
            relationships=relationships,
        )

    # -------------------------------------------------------------------
    # Private: Scope walking
    # -------------------------------------------------------------------

    def _walk_scope(
        self,
        node: Node,
        scope_prefix: str,
        file_path: str,
        module_id: str,
        entities: list[CodeEntity],
        relationships: list[Relationship],
        source: str,
    ) -> None:
        """
        Recursively walk a scope (module body, class body, function body)
        and extract all definitions.
        """
        for child in node.children:
            if child.type == "function_definition":
                self._extract_function(
                    child, scope_prefix, file_path, module_id,
                    entities, relationships, source, is_method=bool(scope_prefix),
                )
            elif child.type == "class_definition":
                self._extract_class(
                    child, scope_prefix, file_path, module_id,
                    entities, relationships, source,
                )
            elif child.type == "decorated_definition":
                # Decorated functions/classes — unwrap the decoration.
                for sub in child.children:
                    if sub.type == "function_definition":
                        decorators = self._extract_decorators(child)
                        self._extract_function(
                            sub, scope_prefix, file_path, module_id,
                            entities, relationships, source,
                            is_method=bool(scope_prefix),
                            extra_decorators=decorators,
                        )
                    elif sub.type == "class_definition":
                        self._extract_class(
                            sub, scope_prefix, file_path, module_id,
                            entities, relationships, source,
                        )

    def _extract_function(
        self,
        node: Node,
        scope_prefix: str,
        file_path: str,
        parent_id: str,
        entities: list[CodeEntity],
        relationships: list[Relationship],
        source: str,
        is_method: bool = False,
        extra_decorators: tuple[str, ...] = (),
    ) -> None:
        """Extract a function/method definition from the AST."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = _get_text(name_node)

        qualified_name = f"{scope_prefix}.{name}" if scope_prefix else name
        entity_id = _build_entity_id(file_path, qualified_name)

        # Parameters.
        params_node = node.child_by_field_name("parameters")
        parameters = self._extract_parameters(params_node) if params_node else ()

        # Return type annotation.
        return_node = node.child_by_field_name("return_type")
        return_type = _get_text(return_node) if return_node else None

        # Is it async?
        is_async = any(c.type == "async" for c in node.parent.children) if node.parent else False

        # Body for docstring.
        body_node = node.child_by_field_name("body")
        docstring = _get_docstring(body_node) if body_node else None

        # Source code of just this function.
        func_source = _get_text(node)

        entity = FunctionEntity(
            id=entity_id,
            name=name,
            qualified_name=qualified_name,
            kind=EntityKind.FUNCTION,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            source_code=func_source,
            docstring=docstring,
            language="python",
            parameters=parameters,
            return_type=return_type,
            is_async=is_async,
            is_method=is_method,
            decorators=extra_decorators or self._extract_decorators_from_siblings(node),
        )
        entities.append(entity)

        # Containment relationship: parent contains this function.
        relationships.append(Relationship(
            source_id=parent_id,
            target_id=entity_id,
            kind=RelationshipKind.CONTAINS,
            file_path=file_path,
            line=node.start_point[0] + 1,
        ))

        # Recurse into the function body for nested definitions.
        if body_node:
            self._walk_scope(
                body_node, qualified_name, file_path, entity_id,
                entities, relationships, source,
            )

    def _extract_class(
        self,
        node: Node,
        scope_prefix: str,
        file_path: str,
        parent_id: str,
        entities: list[CodeEntity],
        relationships: list[Relationship],
        source: str,
    ) -> None:
        """Extract a class definition and its members."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = _get_text(name_node)

        qualified_name = f"{scope_prefix}.{name}" if scope_prefix else name
        entity_id = _build_entity_id(file_path, qualified_name)

        # Base classes.
        bases = self._extract_bases(node)

        # Body.
        body_node = node.child_by_field_name("body")
        docstring = _get_docstring(body_node) if body_node else None
        class_source = _get_text(node)

        entity = ClassEntity(
            id=entity_id,
            name=name,
            qualified_name=qualified_name,
            kind=EntityKind.CLASS,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            source_code=class_source,
            docstring=docstring,
            language="python",
            bases=bases,
        )
        entities.append(entity)

        # Containment: parent → class.
        relationships.append(Relationship(
            source_id=parent_id,
            target_id=entity_id,
            kind=RelationshipKind.CONTAINS,
            file_path=file_path,
            line=node.start_point[0] + 1,
        ))

        # Inheritance relationships.
        for base in bases:
            relationships.append(Relationship(
                source_id=entity_id,
                target_id=base,  # Unresolved — will be matched to entity IDs later.
                kind=RelationshipKind.INHERITS,
                file_path=file_path,
                line=node.start_point[0] + 1,
            ))

        # Recurse into class body to extract methods and nested classes.
        if body_node:
            self._walk_scope(
                body_node, qualified_name, file_path, entity_id,
                entities, relationships, source,
            )

    # -------------------------------------------------------------------
    # Private: Extraction helpers
    # -------------------------------------------------------------------

    def _extract_imports(self, root: Node) -> list[str]:
        """
        Walk the module root and collect all import targets.

        Returns module names, not entity IDs — the graph builder resolves
        these to actual entities later.
        """
        imports: list[str] = []
        for child in root.children:
            if child.type == "import_statement":
                # `import foo, bar`
                for name_node in child.children:
                    if name_node.type == "dotted_name":
                        imports.append(_get_text(name_node))
            elif child.type == "import_from_statement":
                # `from foo.bar import baz`
                module_node = child.child_by_field_name("module_name")
                if module_node:
                    imports.append(_get_text(module_node))
        return imports

    def _extract_parameters(self, params_node: Node) -> tuple[str, ...]:
        """Extract parameter names from a function's parameter list."""
        params: list[str] = []
        for child in params_node.children:
            if child.type == "identifier":
                params.append(_get_text(child))
            elif child.type in ("default_parameter", "typed_parameter", "typed_default_parameter"):
                # The parameter name is the first identifier child — tree-sitter
                # doesn't expose it via a named field in all grammar versions.
                name_node = child.child_by_field_name("name")
                if name_node:
                    params.append(_get_text(name_node))
                else:
                    # Fallback: grab the first identifier child directly.
                    for sub in child.children:
                        if sub.type == "identifier":
                            params.append(_get_text(sub))
                            break
            elif child.type == "dictionary_splat_pattern":
                params.append("**" + _get_text(child.children[-1]) if child.children else "**kwargs")
            elif child.type == "list_splat_pattern":
                params.append("*" + _get_text(child.children[-1]) if child.children else "*args")
        return tuple(params)

    def _extract_bases(self, class_node: Node) -> tuple[str, ...]:
        """Extract base class names from the superclass list."""
        bases: list[str] = []
        superclass_node = class_node.child_by_field_name("superclasses")
        if not superclass_node:
            return ()
        for child in superclass_node.children:
            if child.type in ("identifier", "dotted_name", "attribute"):
                bases.append(_get_text(child))
        return tuple(bases)

    def _extract_decorators(self, decorated_node: Node) -> tuple[str, ...]:
        """Extract decorator names from a decorated_definition node."""
        decorators: list[str] = []
        for child in decorated_node.children:
            if child.type == "decorator":
                # The decorator node's first identifier-like child is the name.
                for sub in child.children:
                    if sub.type in ("identifier", "dotted_name", "attribute", "call"):
                        text = _get_text(sub)
                        # For @decorator(args), just grab the name part.
                        if "(" in text:
                            text = text[: text.index("(")]
                        decorators.append(text)
                        break
        return tuple(decorators)

    def _extract_decorators_from_siblings(self, func_node: Node) -> tuple[str, ...]:
        """
        If the function is inside a decorated_definition, extract its decorators.
        Fallback when we reach the function node directly (not via decorated_definition).
        """
        if func_node.parent and func_node.parent.type == "decorated_definition":
            return self._extract_decorators(func_node.parent)
        return ()

    # -------------------------------------------------------------------
    # Private: Call-site extraction
    # -------------------------------------------------------------------

    def _extract_calls(
        self,
        node: Node,
        scope_prefix: str,
        file_path: str,
        entities: list[CodeEntity],
        relationships: list[Relationship],
    ) -> None:
        """
        Walk the AST and find all function/method call sites.

        For each call, we create a CALLS relationship from the enclosing
        scope (function, class, or module) to the callee. The callee is
        identified by name — the graph builder resolves it to an entity
        ID during cross-file linking.
        """
        # Build a lookup of scope → entity ID for all scope-creating entities.
        # This includes modules (for top-level calls), classes (for class-body
        # calls), and functions (for calls inside function bodies).
        scope_to_id: dict[str, str] = {}
        for entity in entities:
            if entity.kind in (EntityKind.FUNCTION, EntityKind.MODULE, EntityKind.CLASS):
                scope_to_id[entity.qualified_name] = entity.id

        # _walk_calls builds scope names by concatenation starting from
        # the module name. So at module scope, current_scope = "main".
        # Inside a class method, current_scope = "main.MyClass.__init__".
        #
        # But entity qualified_names don't include the module prefix:
        # "MyClass.__init__", not "main.MyClass.__init__".
        #
        # To handle both, we index entities under BOTH their raw
        # qualified_name AND the module-prefixed version.
        module_qname = ""
        for entity in entities:
            if entity.kind == EntityKind.MODULE:
                module_qname = entity.qualified_name
                break

        for entity in entities:
            if entity.kind in (EntityKind.FUNCTION, EntityKind.CLASS):
                # Module-prefixed version (what _walk_calls will construct).
                prefixed = f"{module_qname}.{entity.qualified_name}" if module_qname else entity.qualified_name
                scope_to_id[prefixed] = entity.id

        self._walk_calls(
            node, module_qname, file_path,
            scope_to_id, relationships,
        )

    def _walk_calls(
        self,
        node: Node,
        current_scope: str,
        file_path: str,
        scope_to_id: dict[str, str],
        relationships: list[Relationship],
    ) -> None:
        """Recursively walk the AST looking for call expressions."""
        # Update scope when entering a function or class definition.
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _get_text(name_node)
                current_scope = f"{current_scope}.{name}" if current_scope else name
        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _get_text(name_node)
                current_scope = f"{current_scope}.{name}" if current_scope else name

        # Found a call — record it.
        if node.type == "call":
            callee_name = self._resolve_callee(node)
            caller_id = scope_to_id.get(current_scope)

            if callee_name and caller_id:
                relationships.append(Relationship(
                    source_id=caller_id,
                    target_id=callee_name,  # Unresolved — just the name.
                    kind=RelationshipKind.CALLS,
                    file_path=file_path,
                    line=node.start_point[0] + 1,
                ))

        # Recurse into children.
        for child in node.children:
            self._walk_calls(child, current_scope, file_path, scope_to_id, relationships)

    def _resolve_callee(self, call_node: Node) -> Optional[str]:
        """
        Extract the callee name from a call expression.

        Handles:
            - Simple calls:        `foo()`         → "foo"
            - Attribute calls:     `self.bar()`    → "bar"
            - Chained calls:       `a.b.c()`       → "a.b.c"
        """
        func_node = call_node.child_by_field_name("function")
        if not func_node:
            return None

        if func_node.type == "identifier":
            return _get_text(func_node)
        elif func_node.type == "attribute":
            return _get_text(func_node)
        elif func_node.type == "dotted_name":
            return _get_text(func_node)

        return None
