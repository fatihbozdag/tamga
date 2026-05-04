"""`bitig run <study.yaml>` — execute a full declarative study."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from bitig.runner import run_study

console = Console()


def run_command(
    config: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),  # noqa: B008
    output: Path | None = typer.Option(None, "--output", "-o"),  # noqa: B008
    name: str | None = typer.Option(
        None, "--name", help="Override the default timestamp run-directory name"
    ),
) -> None:
    """Execute a full declarative study and save results to `results/<run>/`."""
    run_dir = run_study(config, output_dir=output, run_name=name)
    console.print(f"[green]run complete[/green] {run_dir}")
