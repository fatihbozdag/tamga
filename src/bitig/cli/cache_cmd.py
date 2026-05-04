"""`bitig cache [size|list|clear]` — inspect and manage the DocBin cache."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from bitig.preprocess.cache import DocBinCache

console = Console()

cache_app = typer.Typer(
    name="cache",
    help="Inspect and manage the DocBin cache.",
    no_args_is_help=True,
)


@cache_app.command("size")
def cache_size(
    cache_dir: Path = typer.Option(Path(".bitig/cache"), "--cache-dir"),  # noqa: B008
) -> None:
    """Show total bytes stored in the DocBin cache."""
    cache = DocBinCache(cache_dir / "docbin")
    console.print(f"{cache.size_bytes()} bytes across {len(cache.keys())} entries")


@cache_app.command("list")
def cache_list(
    cache_dir: Path = typer.Option(Path(".bitig/cache"), "--cache-dir"),  # noqa: B008
) -> None:
    """List cache keys."""
    cache = DocBinCache(cache_dir / "docbin")
    for key in cache.keys():  # noqa: SIM118  # DocBinCache.keys() is a method returning list[str]
        console.print(key)


@cache_app.command("clear")
def cache_clear(
    cache_dir: Path = typer.Option(Path(".bitig/cache"), "--cache-dir"),  # noqa: B008
) -> None:
    """Delete every entry from the DocBin cache."""
    cache = DocBinCache(cache_dir / "docbin")
    n = len(cache.keys())
    cache.clear()
    console.print(f"cleared {n} entries from {cache_dir / 'docbin'}")
