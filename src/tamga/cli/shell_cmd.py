"""`tamga shell [<corpus>]` — guided wizard over the analytical methods."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import IntPrompt, Prompt

from tamga.io import load_corpus

console = Console()


def shell_command(
    corpus_path: Path | None = typer.Argument(None, exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path | None = typer.Option(None, "--metadata", "-m"),  # noqa: B008
) -> None:
    """Launch the interactive wizard."""
    console.rule("[bold cyan]tamga shell[/bold cyan]")

    if corpus_path is None:
        corpus_path_str = Prompt.ask("Corpus path")
        corpus_path = Path(corpus_path_str)
    console.print(f"[green]Corpus:[/green] {corpus_path}")

    corpus = load_corpus(corpus_path, metadata=metadata)
    console.print(f"[green]Loaded {len(corpus)} documents[/green]")

    menu = [
        "Inspect corpus",
        "Run Delta attribution (tamga delta)",
        "Run Zeta comparison (tamga zeta)",
        "Cluster & visualise (tamga cluster)",
        "Classify (tamga classify)",
        "Reduce & plot (tamga reduce)",
        "Quit",
    ]
    for i, item in enumerate(menu, start=1):
        console.print(f"  [cyan]{i}[/cyan]. {item}")
    choice = IntPrompt.ask("Choose", default=1, choices=[str(i) for i in range(1, len(menu) + 1)])

    if choice == 1:
        console.print(f"Documents: {len(corpus)}")
        meta_keys: set[str] = set().union(*(d.metadata.keys() for d in corpus.documents))
        console.print(f"Metadata fields: {sorted(meta_keys)}")
    elif choice == len(menu):
        console.print("[dim]bye[/dim]")
    else:
        method = menu[choice - 1]
        console.print(
            f"Would run: [dim]tamga {method.split('(')[1].rstrip(')').split()[1]}[/dim]"
            f" — run the CLI directly for the full flow."
        )
