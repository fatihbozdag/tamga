"""Build an HTML or Markdown report from a directory containing saved Result artefacts."""

from __future__ import annotations

import json
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment

from tamga._version import __version__

Format = Literal["html", "md"]

_TEMPLATE_PKG = "tamga.report.templates"


def build_report(
    result_dir: str | Path,
    *,
    output: str | Path,
    format: Format = "html",
    title: str = "tamga study",
    corpus_summary: dict[str, Any] | None = None,
) -> Path:
    """Assemble `result.json` + figures in `result_dir` into a single HTML/MD file at `output`.

    `result_dir` may be either a single Result directory (one method) or a parent directory
    containing per-method subdirectories (produced by `tamga run`).
    """
    result_dir = Path(result_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    results = list(_collect_results(result_dir))
    provenance = _first_provenance(results) or {}

    template_name = f"report.{format}.j2"
    env = Environment(keep_trailing_newline=True)
    template = env.from_string(
        (resources.files(_TEMPLATE_PKG) / template_name).read_text(encoding="utf-8")
    )

    rendered = template.render(
        title=title,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        tamga_version=__version__,
        corpus_summary=corpus_summary,
        resolved_config=json.dumps(provenance.get("resolved_config", {}), indent=2),
        results=results,
        provenance=json.dumps(provenance, indent=2, default=str),
    )
    output.write_text(rendered, encoding="utf-8")
    return output


def _collect_results(result_dir: Path) -> Any:
    if (result_dir / "result.json").is_file():
        yield _load_one(result_dir)
        return
    for sub in sorted(result_dir.iterdir()):
        if sub.is_dir() and (sub / "result.json").is_file():
            yield _load_one(sub)


def _load_one(result_subdir: Path) -> dict[str, Any]:
    data = json.loads((result_subdir / "result.json").read_text(encoding="utf-8"))
    figures = [str(p.resolve()) for p in sorted(result_subdir.glob("*.png"))]
    return {
        "method_name": data.get("method_name", "unknown"),
        "params": json.dumps(data.get("params", {}), indent=2),
        "figures": figures,
        "tables": [],  # Tables are exported as parquet; embedding in report deferred to Phase 6.
        "summary": None,
        "provenance": data.get("provenance"),
    }


def _first_provenance(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    for r in results:
        if r.get("provenance"):
            return r["provenance"]  # type: ignore[no-any-return]
    return None
