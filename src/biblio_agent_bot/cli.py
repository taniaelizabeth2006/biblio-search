from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from biblio_agent_bot.agents import BiblioAgentBot
from biblio_agent_bot.config import load_config
from biblio_agent_bot.llm import provider_from_name
from biblio_agent_bot.report import write_outputs

app = typer.Typer(help="Agentic bibliographic review bot for health research.")
console = Console()


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, readable=True, help="YAML topic config."),
    output: Path = typer.Option(Path("runs/latest"), "--output", "-o", help="Output directory."),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="none, gemini, claude-cli, or codex-cli. Overrides YAML.",
    ),
    model_name: str | None = typer.Option(None, "--model", help="Model name override (e.g. gpt-4o, gemini-3-pro-preview, claude-opus-4-5)."),
    max_records_per_source: int | None = typer.Option(
        None,
        "--max-records-per-source",
        min=1,
        help="Override YAML max_records_per_source for PubMed/EuropePMC/Crossref.",
    ),
    embedding_provider: str | None = typer.Option(
        None,
        "--embedding-provider",
        help="none or openrouter. Uses embeddings to prefilter records before LLM screening.",
    ),
    embedding_model: str | None = typer.Option(
        None,
        "--embedding-model",
        help="Embedding model override, e.g. google/gemini-embedding-2-preview.",
    ),
    embedding_top_k: int | None = typer.Option(
        None,
        "--embedding-top-k",
        min=1,
        help="Number of embedding-ranked records to send to LLM screening.",
    ),
    embedding_min_score: float | None = typer.Option(
        None,
        "--embedding-min-score",
        help="Optional cosine-similarity threshold for embedding prefilter.",
    ),
) -> None:
    topic = load_config(config)
    if provider:
        topic.provider = provider
    if model_name:
        topic.model_name = model_name
    if max_records_per_source is not None:
        topic.max_records_per_source = max_records_per_source
    if embedding_provider:
        topic.embedding_provider = embedding_provider
    if embedding_model:
        topic.embedding_model = embedding_model
    if embedding_top_k is not None:
        topic.embedding_top_k = embedding_top_k
    if embedding_min_score is not None:
        topic.embedding_min_score = embedding_min_score

    progress = lambda message: console.log(message)
    llm = provider_from_name(topic.provider, model_name=topic.model_name, progress=progress)
    console.print(f"[bold]Running review[/bold]: {topic.project_slug}")
    console.print(f"Provider: {llm.name}")
    console.print(
        f"Embedding prefilter: {topic.embedding_provider}"
        + (
            f" ({topic.embedding_model}, top_k={topic.embedding_top_k})"
            if topic.embedding_provider != "none"
            else ""
        )
    )
    console.print(f"Sources: {', '.join(topic.sources)}")

    bot = BiblioAgentBot(llm, progress=progress)
    review = bot.run(topic)
    console.log(f"Writing outputs to {output}...")
    write_outputs(review, output)
    console.log("Outputs written.")

    table = Table(title="Bibliographic Review Summary")
    table.add_column("Metric")
    table.add_column("Value")
    llm_screened_count = (
        len(review.embedding_selected_records)
        if topic.embedding_provider != "none"
        else len(review.deduped_records)
    )
    table.add_row("Raw records", str(len(review.raw_records)))
    table.add_row("Unique records", str(len(review.deduped_records)))
    table.add_row("Embedding-ranked records", str(len(review.embedding_ranks)))
    table.add_row("LLM-screened records", str(llm_screened_count))
    table.add_row("Selected records", str(len(review.selected_records)))
    table.add_row("Report", str(output / "report.md"))
    table.add_row("Matrix", str(output / "screening_matrix.csv"))
    table.add_row("Audit", str(output / "audit.json"))
    console.print(table)


@app.command()
def providers() -> None:
    table = Table(title="Supported LLM Providers")
    table.add_column("Provider")
    table.add_column("Use case")
    table.add_column("Requirements")
    table.add_row("none", "Deterministic search plan and heuristic screening", "No LLM")
    table.add_row("gemini", "Best structured extraction/synthesis", "GEMINI_API_KEY")
    table.add_row("claude-cli", "Use local Claude Code auth/session", "claude command")
    table.add_row("codex-cli", "Use local Codex auth/session", "codex command")
    table.add_row(
        "openrouter embeddings",
        "Prefilter thousands of abstracts before LLM screening",
        "OPENROUTER_API_KEY",
    )
    console.print(table)


if __name__ == "__main__":
    app()
