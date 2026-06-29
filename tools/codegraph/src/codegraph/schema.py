"""
Graph schema constants.

Defines which node types and edge types exist in our knowledge graph, and
which combinations are valid. This acts as a contract — the graph builder
uses it as a validation layer to catch parser bugs early.
"""

from codegraph.models import EntityKind, RelationshipKind


# ---------------------------------------------------------------------------
# Valid edge types between entity kinds
# ---------------------------------------------------------------------------

# Maps (source_kind, target_kind) → set of allowed relationship kinds.
# If a parser produces a relationship not in this table, it's a bug.

VALID_RELATIONSHIPS: dict[tuple[EntityKind, EntityKind], set[RelationshipKind]] = {
    #
    # Module-level relationships
    #
    (EntityKind.MODULE, EntityKind.MODULE): {
        RelationshipKind.IMPORTS,
    },
    (EntityKind.MODULE, EntityKind.CLASS): {
        RelationshipKind.CONTAINS,
        RelationshipKind.IMPORTS,
    },
    (EntityKind.MODULE, EntityKind.FUNCTION): {
        RelationshipKind.CONTAINS,
        RelationshipKind.IMPORTS,
    },
    (EntityKind.MODULE, EntityKind.VARIABLE): {
        RelationshipKind.CONTAINS,
    },
    #
    # Class-level relationships
    #
    (EntityKind.CLASS, EntityKind.CLASS): {
        RelationshipKind.INHERITS,
    },
    (EntityKind.CLASS, EntityKind.INTERFACE): {
        RelationshipKind.IMPLEMENTS,
    },
    (EntityKind.CLASS, EntityKind.FUNCTION): {
        RelationshipKind.CONTAINS,
    },
    (EntityKind.CLASS, EntityKind.VARIABLE): {
        RelationshipKind.CONTAINS,
    },
    #
    # Function-level relationships
    #
    (EntityKind.FUNCTION, EntityKind.FUNCTION): {
        RelationshipKind.CALLS,
        RelationshipKind.OVERRIDES,
    },
    (EntityKind.FUNCTION, EntityKind.CLASS): {
        RelationshipKind.CALLS,      # instantiation is a call
        RelationshipKind.USES_TYPE,
    },
    (EntityKind.FUNCTION, EntityKind.VARIABLE): {
        RelationshipKind.CALLS,      # calling a callable variable
    },
    (EntityKind.FUNCTION, EntityKind.INTERFACE): {
        RelationshipKind.USES_TYPE,
    },
    (EntityKind.FUNCTION, EntityKind.TYPE_ALIAS): {
        RelationshipKind.USES_TYPE,
    },
}


def is_valid_relationship(
    source_kind: EntityKind,
    target_kind: EntityKind,
    relationship_kind: RelationshipKind,
) -> bool:
    """Check whether a relationship is structurally valid per our schema."""
    allowed = VALID_RELATIONSHIPS.get((source_kind, target_kind), set())
    return relationship_kind in allowed
