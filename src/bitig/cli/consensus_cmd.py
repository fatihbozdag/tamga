"""`bitig consensus <corpus>` — bootstrap consensus tree over MFW bands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from bitig.io import load_corpus
from bitig.methods.consensus import BootstrapConsensus

console = Console()


def consensus_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path | None = typer.Option(None, "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    bands: str = typer.Option(
        "100,200,300,400,500", "--bands", help="Comma-separated MFW band sizes"
    ),
    replicates: int = typer.Option(100, "--replicates"),
    subsample: float = typer.Option(0.8, "--subsample"),
    support_threshold: float = typer.Option(0.5, "--support-threshold"),
    seed: int = typer.Option(42, "--seed"),
    output: Path = typer.Option(Path("consensus.nwk"), "--output", "-o"),  # noqa: B008
) -> None:
    """Build a bootstrap consensus tree and write Newick to disk."""
    corpus = load_corpus(path, metadata=metadata)
    bands_list = [int(b) for b in bands.split(",")]
    result = BootstrapConsensus(
        mfw_bands=bands_list,
        replicates=replicates,
        subsample=subsample,
        support_threshold=support_threshold,
        seed=seed,
    ).fit_transform(corpus)
    output.write_text(result.values["newick"], encoding="utf-8")
    console.print(
        f"[green]wrote[/green] {output} (based on {result.values['total_dendrograms']} dendrograms)"
    )
