"""`tamga info` — show versions and environment."""

from __future__ import annotations

import platform

import spacy
from rich.console import Console
from rich.table import Table

from tamga._version import __version__

console = Console()


def info_command() -> None:
    """Print versions, paths, and runtime information."""
    table = Table(title="tamga environment", show_header=False)
    table.add_column("key", style="cyan")
    table.add_column("value")
    table.add_row("tamga", __version__)
    table.add_row("python", platform.python_version())
    table.add_row("platform", platform.platform())
    table.add_row("spacy", spacy.__version__)
    console.print(table)
