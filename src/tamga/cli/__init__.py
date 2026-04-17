"""Typer CLI entry point."""

from __future__ import annotations

import typer
from rich.console import Console

from tamga._version import __version__

console = Console()
app = typer.Typer(
    name="tamga",
    help="tamga — computational stylometry (next-generation Python replacement for R's Stylo).",
    no_args_is_help=True,
    add_completion=True,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"tamga {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """tamga — computational stylometry."""
