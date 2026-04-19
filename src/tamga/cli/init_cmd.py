"""`tamga init <name>` — scaffold a new project."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from tamga.scaffold import scaffold_project

console = Console()


def init_command(
    name: str = typer.Argument(..., help="Project name; used as directory name and in study.yaml."),
    target: Path | None = typer.Option(  # noqa: B008
        None, "--target", "-t", help="Directory to create (default: ./<name>)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Fill in missing files even if directory is non-empty."
    ),
    language: str = typer.Option(
        "en",
        "--language",
        "-l",
        help="Project language code (en, tr, de, es, fr). Default: en.",
    ),
) -> None:
    """Scaffold a new tamga project directory."""
    dest = target if target is not None else Path.cwd() / name
    try:
        created = scaffold_project(name=name, target=dest, force=force, language=language)
    except FileExistsError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        # Unknown language code — surface the registry's helpful message.
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]created project[/green] {created} (language={language})")
    console.print(f"  cd {created}")
    console.print("  # edit study.yaml; drop .txt files in corpus/; then run:")
    console.print("  tamga run study.yaml")
