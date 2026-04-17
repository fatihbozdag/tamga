"""`tamga ingest <path>` — parse a corpus and populate the DocBin cache."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from tamga.io import load_corpus
from tamga.preprocess.pipeline import SpacyPipeline

console = Console()


def ingest_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path | None = typer.Option(  # noqa: B008
        None,
        "--metadata",
        "-m",
        exists=True,
        dir_okay=False,
        help="TSV file mapping filename to metadata fields.",
    ),
    strict: bool = typer.Option(
        True, "--strict/--no-strict", help="Every file must have metadata."
    ),
    cache_dir: Path = typer.Option(  # noqa: B008
        Path(".tamga/cache"), "--cache-dir", help="Directory for the DocBin cache."
    ),
    spacy_model: str = typer.Option("en_core_web_trf", "--spacy-model", help="spaCy model name."),
    exclude: list[str] | None = typer.Option(  # noqa: B008
        None, "--exclude", help="spaCy pipeline components to skip."
    ),
) -> None:
    """Parse a corpus directory and cache spaCy parses."""
    corpus = load_corpus(path, metadata=metadata, strict=strict)
    console.print(f"[green]loaded[/green] {len(corpus)} documents from {path}")

    pipe = SpacyPipeline(
        model=spacy_model,
        cache_dir=cache_dir / "docbin",
        exclude=exclude or [],
    )

    docbin_before = set(pipe.cache.keys())
    pipe.parse(corpus)
    docbin_after = set(pipe.cache.keys())
    newly_parsed = len(docbin_after - docbin_before)
    cached_hits = len(corpus) - newly_parsed

    console.print(
        f"[green]parsed[/green] {len(corpus)} documents"
        f" ({cached_hits} cached, {newly_parsed} newly parsed)"
    )
    console.print(f"  cache: {cache_dir / 'docbin'} ({pipe.cache.size_bytes()} bytes)")
