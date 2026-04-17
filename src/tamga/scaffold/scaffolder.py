"""Scaffold a new tamga project directory."""

from __future__ import annotations

from datetime import datetime
from importlib import resources
from pathlib import Path

from jinja2 import Environment

from tamga._version import __version__

_TEMPLATE_PKG = "tamga.scaffold.templates"


def scaffold_project(name: str, target: Path, *, force: bool = False) -> Path:
    """Create a new tamga project at `target`.

    Refuses to create on top of an existing non-empty directory unless `force=True`. When `force`
    is True, existing files are left alone and only missing scaffold files are written.
    """
    target = Path(target)
    if target.exists():
        if any(target.iterdir()) and not force:
            raise FileExistsError(f"{target} exists and is not empty (use force=True to fill in)")
    else:
        target.mkdir(parents=True)

    (target / "corpus").mkdir(exist_ok=True)
    (target / ".tamga" / "cache").mkdir(parents=True, exist_ok=True)
    (target / "results").mkdir(exist_ok=True)
    (target / "reports").mkdir(exist_ok=True)

    env = Environment(trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True)
    ctx: dict[str, object] = {
        "name": name,
        "tamga_version": __version__,
        "created_on": datetime.now().strftime("%Y-%m-%d"),
    }

    _render(env, "study.yaml.j2", target / "study.yaml", ctx)
    _render(env, "README.md.j2", target / "README.md", ctx)
    _copy("gitignore.tmpl", target / ".gitignore")
    return target


def _render(env: Environment, template_name: str, dest: Path, ctx: dict[str, object]) -> None:
    if dest.exists():
        return
    src = resources.files(_TEMPLATE_PKG) / template_name
    template = env.from_string(src.read_text(encoding="utf-8"))
    dest.write_text(template.render(**ctx), encoding="utf-8")


def _copy(template_name: str, dest: Path) -> None:
    if dest.exists():
        return
    src = resources.files(_TEMPLATE_PKG) / template_name
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
