"""`tamga plot <result-dir>` — render figures from a saved Result directory."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from tamga.viz.style import apply_publication_style

console = Console()


def plot_command(
    result_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    format: str = typer.Option("png", "--format", help="png | pdf | svg"),
    dpi: int = typer.Option(300, "--dpi"),
) -> None:
    """Re-render figures for a saved Result directory (idempotent)."""
    apply_publication_style(dpi=dpi)
    rj = result_dir / "result.json"
    if not rj.is_file():
        console.print(f"[red]error:[/red] {rj} not found")
        raise typer.Exit(code=1)
    data = json.loads(rj.read_text(encoding="utf-8"))
    method = data.get("method_name", "?")
    console.print(f"(plot rendering for {method} — stub; full plot renderer wiring is in Phase 6)")
    console.print(
        "[yellow]note:[/yellow] to render actual figures, use the tamga.viz.plot_* "
        "functions directly from Python for now"
    )
