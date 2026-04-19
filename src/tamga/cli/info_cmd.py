"""`tamga info` — show versions and environment."""

from __future__ import annotations

import platform
from pathlib import Path

import spacy
from rich.console import Console
from rich.table import Table

from tamga._version import __version__

console = Console()


def info_command() -> None:
    """Print versions, paths, and runtime information.

    When a `study.yaml` is present in the current working directory, the configured language
    is additionally surfaced so operators can confirm the active pipeline at a glance.
    """
    table = Table(title="tamga environment", show_header=False)
    table.add_column("key", style="cyan")
    table.add_column("value")
    table.add_row("tamga", __version__)
    table.add_row("python", platform.python_version())
    table.add_row("platform", platform.platform())
    table.add_row("spacy", spacy.__version__)

    study_yaml = Path.cwd() / "study.yaml"
    if study_yaml.is_file():
        language = _read_study_language(study_yaml)
        if language is not None:
            table.add_row("language", language)

    console.print(table)


def _read_study_language(path: Path) -> str | None:
    """Return the language stamped in `study.yaml`, or None if unreadable/absent.

    Falls back to ``"en"`` when the file is valid YAML but does not declare a language.
    Any parse/validation error leaves the info output silent rather than raising.
    """
    try:
        from tamga.config import load_config

        cfg = load_config(path)
        return cfg.preprocess.language
    except Exception:
        return None
