"""`bitig report <result-dir>` — generate an HTML/MD report from a run directory."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from bitig.report import build_report

console = Console()


def report_command(
    result_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    output: Path = typer.Option(Path("report.html"), "--output", "-o"),  # noqa: B008
    format: str = typer.Option("html", "--format", help="html | md"),
    title: str = typer.Option("bitig study", "--title"),
) -> None:
    """Generate an HTML or Markdown report from a bitig run directory."""
    if format not in ("html", "md"):
        console.print(f"[red]error:[/red] format must be 'html' or 'md', got {format!r}")
        raise typer.Exit(code=1)
    out = build_report(result_dir, output=output, format=format, title=title)  # type: ignore[arg-type]
    console.print(f"[green]wrote[/green] {out}")
