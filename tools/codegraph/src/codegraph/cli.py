"""
CLI interface for CodeGraph.

Commands:
    codegraph index <path>       — Parse + build graph + detect communities.
    codegraph query "<text>"     — Ask a question (with graph reasoning).
    codegraph impact <name>      — Deep blast radius analysis.
    codegraph compare "<text>"   — Compare CodeGraph vs LLM+RAG side-by-side.
    codegraph stats              — Print graph statistics.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text

from codegraph.graph.builder import GraphBuilder
from codegraph.graph.community import Community, attach_communities_to_graph, detect_communities
from codegraph.graph.queries import (
    deep_impact_analysis,
    find_entity_by_name,
    get_entity,
    get_graph_stats,
)
from codegraph.graph.summarizer import summarize_communities
from codegraph.query.engine import QueryEngine

console = Console()

INDEX_DIR = Path(".codegraph")
GRAPH_FILE = INDEX_DIR / "graph.pkl"
COMMUNITIES_FILE = INDEX_DIR / "communities.json"


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging.")
def main(debug: bool) -> None:
    """CodeGraph — Code-aware GraphRAG for complex codebases."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(name)s | %(message)s")


@main.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--resolution", default=1.0, help="Leiden resolution parameter.")
def index(path: str, resolution: float) -> None:
    """Parse a project and build the knowledge graph."""
    project_root = Path(path).resolve()
    console.print(f"\n[bold]Indexing[/bold] {project_root}\n")

    with console.status("[bold green]Parsing source files..."):
        builder = GraphBuilder()
        graph = builder.build(project_root)

    stats = get_graph_stats(graph)
    _print_stats(stats)

    with console.status("[bold green]Detecting communities (Leiden)..."):
        communities = detect_communities(graph, resolution=resolution)

    with console.status("[bold green]Generating community summaries..."):
        summarize_communities(graph, communities)
        attach_communities_to_graph(graph, communities)

    console.print(f"\n[green]✓[/green] Detected {len(communities)} logical modules.\n")

    for community in communities[:10]:
        entity_count = len(community.entity_ids)
        console.print(Panel(
            community.summary or "(no summary)",
            title=f"Module {community.id} ({entity_count} entities)",
            border_style="dim",
        ))

    INDEX_DIR.mkdir(exist_ok=True)
    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(graph, f)

    community_data = [
        {"id": c.id, "entity_count": len(c.entity_ids), "summary": c.summary}
        for c in communities
    ]
    with open(COMMUNITIES_FILE, "w") as f:
        json.dump(community_data, f, indent=2)

    console.print(f"\n[green]✓[/green] Index saved to {INDEX_DIR}/")


@main.command()
@click.argument("question")
def query(question: str) -> None:
    """Ask a question — answered via graph reasoning, not keyword search."""
    graph = _load_graph()
    engine = QueryEngine(graph)
    answer = engine.query(question)
    console.print(f"\n{answer}\n")


@main.command()
@click.argument("entity_name")
@click.option("--depth", default=6, help="Maximum traversal depth.")
def impact(entity_name: str, depth: int) -> None:
    """Deep blast radius analysis for an entity."""
    graph = _load_graph()
    matches = find_entity_by_name(graph, entity_name)

    if not matches:
        console.print(f"[red]✗[/red] No entity found matching '{entity_name}'.")
        return

    for entity_id in matches[:3]:
        result = deep_impact_analysis(graph, entity_id, max_depth=depth)

        console.print(Panel(
            f"[bold]{result.source_name}[/bold]\n"
            f"File: {result.source_file}\n"
            f"Total blast radius: [bold red]{result.total_affected}[/bold red] entities\n"
            f"Max propagation depth: {result.max_depth}",
            title="💥 Impact Analysis",
            border_style="red",
        ))

        if result.direct:
            table = Table(title=f"🔴 Direct Dependents ({len(result.direct)})")
            table.add_column("Relationship", style="red")
            table.add_column("Entity", style="bold")
            table.add_column("File", style="cyan")
            table.add_column("Kind", style="dim")
            for a in result.direct:
                table.add_row(a.relationship, a.name, a.file_path, a.kind)
            console.print(table)

        if result.transitive:
            table = Table(title=f"🟡 Transitive Impact ({len(result.transitive)})")
            table.add_column("Depth", style="yellow", justify="center")
            table.add_column("Relationship", style="yellow")
            table.add_column("Entity", style="bold")
            table.add_column("File", style="cyan")
            for a in result.transitive[:20]:
                table.add_row(str(a.depth), a.relationship, a.name, a.file_path)
            if len(result.transitive) > 20:
                console.print(f"  ... and {len(result.transitive) - 20} more")
            console.print(table)

        if result.critical_chains:
            console.print("\n[bold]🔗 Critical Dependency Chains:[/bold]")
            for i, chain in enumerate(result.critical_chains):
                chain_str = " → ".join(chain)
                console.print(f"  Chain {i + 1}: {chain_str}")

        console.print("")


@main.command()
@click.argument("question")
@click.option("--api-key", envvar="OPENROUTER_API_KEY", help="OpenRouter API key.")
@click.option("--model", default="openai/gpt-4o", help="LLM model to compare against.")
def compare(question: str, api_key: str, model: str) -> None:
    """Compare CodeGraph vs LLM+RAG — FAIR side-by-side comparison."""
    if not api_key:
        console.print("[red]✗[/red] Set OPENROUTER_API_KEY or pass --api-key.")
        return

    graph = _load_graph()
    engine = QueryEngine(graph)

    console.print(f"\n[bold]Question:[/bold] {question}\n")

    with console.status("[bold green]Running CodeGraph graph reasoning..."):
        cg_answer = engine.query(question)

    with console.status(f"[bold blue]Querying {model} with RAG context (code chunks)..."):
        from codegraph.comparison import run_comparison
        result = run_comparison(question, cg_answer, api_key, graph, model)

    # Display side-by-side.
    console.print(Panel(
        result.codegraph_answer,
        title="🧠 CodeGraph (Graph Traversal)",
        border_style="green",
    ))

    console.print(Panel(
        f"[dim]Context given: {result.context_given_to_llm}[/dim]\n\n"
        + result.llm_answer,
        title=f"🤖 {result.llm_model} (LLM + RAG chunks)",
        border_style="blue",
    ))

    console.print(Panel(
        result.analysis.to_text(),
        title="⚖️ Structured Analysis",
        border_style="yellow",
    ))


@main.command()
def stats() -> None:
    """Print statistics about the indexed codebase."""
    graph = _load_graph()
    stats = get_graph_stats(graph)
    _print_stats(stats)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _load_graph():
    if not GRAPH_FILE.exists():
        console.print("[red]✗[/red] No index found. Run `codegraph index <path>` first.")
        raise SystemExit(1)
    with open(GRAPH_FILE, "rb") as f:
        return pickle.load(f)


def _print_stats(stats: dict[str, int]) -> None:
    table = Table(title="Knowledge Graph Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right", style="cyan")

    for key, value in stats.items():
        label = key.replace("_", " ").title()
        table.add_row(label, str(value))

    console.print(table)
