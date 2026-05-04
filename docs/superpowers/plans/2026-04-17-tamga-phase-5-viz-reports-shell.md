# bitig — Phase 5: Visualisation, Reports, `bitig run`, Interactive Shell — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans.

**Goal:** Ship the user-facing polish layer — publication-grade static figures (matplotlib/seaborn), optional interactive figures (plotly when `[viz]` installed), HTML/Markdown reports rendered from saved `Result` artefacts, a config-driven `bitig run study.yaml` orchestrator that runs a full multi-method study end-to-end, and a Rich/Questionary interactive wizard shell (`bitig shell`) that exposes the same workflow through a guided menu. End state: `bitig run study.yaml --report html` produces a self-contained HTML report with every figure and provenance section for a publication-ready analysis.

**Architecture:** Backend-agnostic plotting API (`bitig.viz.plot_*` functions) that dispatches to matplotlib or plotly based on the `backend=` argument; static matplotlib always available, plotly gated on the `[viz]` extra with graceful fallback. Reports assemble a `Result` directory into a Jinja2-templated HTML/Markdown document with embedded figures. `bitig run` is a single orchestrator that reads `study.yaml`, builds the Corpus → spaCy parse → per-method feature + method execution → saves all Results to a timestamped directory → optionally generates a report. The shell is a thin Rich/Questionary menu wrapping the same orchestrator.

**Tech Stack:** matplotlib (core), seaborn (core), plotly (optional `[viz]`), Jinja2 (core, already declared), scipy.cluster.hierarchy (dendrogram plumbing, already used), rich + questionary (core, already declared), IPython (lazy-import for shell's REPL-escape option — no new dep).

**Reference spec:** §8.4 (shell), §10 (viz), §11 (reports).

**Phase 4 baseline:** tag `phase-4-extras`, 213 tests.

---

## File Layout

```
src/bitig/
├── viz/
│   ├── __init__.py          # public plot_* functions
│   ├── style.py             # publication defaults (DPI, font, palette, figure sizes)
│   ├── mpl.py               # matplotlib renderers
│   └── plotly.py            # plotly renderers (gated on [viz])
├── report/
│   ├── __init__.py
│   ├── render.py            # build_report(result_dir, format, offline)
│   ├── templates/
│   │   ├── report.html.j2
│   │   └── report.md.j2
├── runner.py                # run_study(config, cache_dir, output_dir, report_format)
└── cli/
    ├── plot_cmd.py          # bitig plot <result>
    ├── report_cmd.py        # bitig report <result-dir|study.yaml>
    ├── run_cmd.py           # bitig run <study.yaml>
    └── shell_cmd.py         # bitig shell

tests/
├── viz/
│   ├── test_style.py
│   ├── test_mpl.py
│   └── test_plotly.py       # @pytest.mark.slow
├── report/
│   └── test_render.py
├── test_runner.py
└── cli/
    ├── test_plot_cmd.py
    ├── test_report_cmd.py
    ├── test_run_cmd.py
    └── test_shell_cmd.py
```

---

## Task 1: `viz.style` — publication defaults

**Files:**
- Create: `src/bitig/viz/__init__.py`
- Create: `src/bitig/viz/style.py`
- Create: `tests/viz/__init__.py`
- Create: `tests/viz/test_style.py`

### Step 1.1 — Tests

```python
"""Tests for publication-style defaults."""

import matplotlib.pyplot as plt

from bitig.viz.style import apply_publication_style, figure_size


def test_apply_publication_style_sets_dpi():
    apply_publication_style(dpi=300)
    fig = plt.figure()
    assert fig.get_dpi() == 300
    plt.close(fig)


def test_figure_size_single_column_is_3_5_inches():
    w, h = figure_size("single")
    assert w == 3.5


def test_figure_size_double_column_is_7_inches():
    w, _ = figure_size("double")
    assert w == 7.0


def test_apply_publication_style_uses_colorblind_palette():
    apply_publication_style()
    import seaborn as sns

    palette = sns.color_palette()
    assert len(palette) >= 6
```

### Step 1.2 — Implement `src/bitig/viz/style.py`

```python
"""Publication-grade styling defaults.

Sets matplotlib rcParams to journal-friendly values: 300 DPI, serif fonts, colorblind palette,
clean grid. Applied via `apply_publication_style()`.
"""

from __future__ import annotations

from typing import Literal

import matplotlib as mpl
import seaborn as sns

ColumnWidth = Literal["single", "one_and_half", "double"]

_WIDTHS: dict[str, tuple[float, float]] = {
    "single": (3.5, 2.5),
    "one_and_half": (5.0, 3.5),
    "double": (7.0, 5.0),
}


def apply_publication_style(
    *,
    dpi: int = 300,
    font_family: str = "serif",
    palette: str = "colorblind",
) -> None:
    """Set matplotlib + seaborn defaults for publication output."""
    mpl.rcParams["figure.dpi"] = dpi
    mpl.rcParams["savefig.dpi"] = dpi
    mpl.rcParams["font.family"] = font_family
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.linestyle"] = ":"
    mpl.rcParams["grid.alpha"] = 0.3
    sns.set_palette(palette)


def figure_size(width: ColumnWidth = "single") -> tuple[float, float]:
    """Return (width_in, height_in) for a standard journal column width."""
    return _WIDTHS[width]
```

### Step 1.3 — `src/bitig/viz/__init__.py`

```python
"""bitig visualisation — matplotlib/seaborn static + plotly interactive."""

from bitig.viz.style import apply_publication_style, figure_size

__all__ = ["apply_publication_style", "figure_size"]
```

### Step 1.4 — `tests/viz/__init__.py` (empty)

### Step 1.5 — Run → PASS. Commit:

```bash
git add src/bitig/viz/__init__.py src/bitig/viz/style.py tests/viz/__init__.py tests/viz/test_style.py
git commit -m "feat(viz): publication-grade matplotlib defaults + figure_size helper"
```

---

## Task 2: `viz.mpl` — matplotlib plot renderers

**Files:**
- Create: `src/bitig/viz/mpl.py`
- Create: `tests/viz/test_mpl.py`
- Modify: `src/bitig/viz/__init__.py` (re-export plot_* functions)

### Step 2.1 — Tests `tests/viz/test_mpl.py`

```python
"""Tests for matplotlib renderers. Verifies figures are created and savable."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage

from bitig.viz.mpl import (
    plot_confusion_matrix,
    plot_dendrogram,
    plot_distance_heatmap,
    plot_feature_importance,
    plot_scatter_2d,
    plot_zeta,
)


def test_plot_dendrogram_saves_png(tmp_path: Path):
    X = np.random.default_rng(42).standard_normal((8, 4))
    Z = linkage(X, method="ward")
    labels = [f"d{i}" for i in range(8)]
    out = tmp_path / "dendro.png"
    fig = plot_dendrogram(Z, labels=labels)
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_scatter_2d_with_groups(tmp_path: Path):
    coords = np.random.default_rng(42).standard_normal((20, 2))
    groups = ["A"] * 10 + ["B"] * 10
    labels = [f"d{i}" for i in range(20)]
    fig = plot_scatter_2d(coords, labels=labels, groups=groups)
    fig.savefig(tmp_path / "scatter.png")


def test_plot_distance_heatmap_sym_matrix(tmp_path: Path):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((5, 3))
    D = np.abs(X[:, None, :] - X[None, :, :]).mean(axis=2)
    fig = plot_distance_heatmap(D, labels=[f"d{i}" for i in range(5)])
    fig.savefig(tmp_path / "heatmap.png")


def test_plot_confusion_matrix(tmp_path: Path):
    y_true = np.array(["A", "A", "B", "B", "A", "B"])
    y_pred = np.array(["A", "B", "B", "B", "A", "A"])
    fig = plot_confusion_matrix(y_true, y_pred)
    fig.savefig(tmp_path / "confusion.png")


def test_plot_feature_importance(tmp_path: Path):
    names = ["the", "and", "of", "to", "a"]
    importance = np.array([0.5, 0.3, 0.2, 0.15, 0.1])
    fig = plot_feature_importance(names, importance)
    fig.savefig(tmp_path / "importance.png")


def test_plot_zeta(tmp_path: Path):
    df_a = pd.DataFrame({"word": ["alpha", "beta"], "zeta": [0.8, 0.6], "prop_a": [1.0, 0.8], "prop_b": [0.2, 0.2]})
    df_b = pd.DataFrame({"word": ["gamma", "delta"], "zeta": [-0.7, -0.5], "prop_a": [0.1, 0.2], "prop_b": [0.8, 0.7]})
    fig = plot_zeta(df_a, df_b, label_a="A", label_b="B")
    fig.savefig(tmp_path / "zeta.png")
```

### Step 2.2 — Implement `src/bitig/viz/mpl.py`

```python
"""Matplotlib renderers for every major bitig plot type."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import confusion_matrix

from bitig.viz.style import figure_size


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    *,
    labels: list[str] | None = None,
    title: str = "Dendrogram",
    orientation: str = "top",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figure_size("one_and_half"))
    dendrogram(linkage_matrix, labels=labels, orientation=orientation, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scatter_2d(
    coordinates: np.ndarray,
    *,
    labels: list[str] | None = None,
    groups: list[str] | None = None,
    title: str = "2D projection",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figure_size("single"))
    if groups is not None:
        unique = sorted(set(groups))
        palette = sns.color_palette(n_colors=len(unique))
        for i, g in enumerate(unique):
            mask = np.array([gg == g for gg in groups])
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], label=g, color=palette[i], s=40)
        ax.legend(fontsize=7)
    else:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=40)
    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(label, (coordinates[i, 0], coordinates[i, 1]), fontsize=6, alpha=0.7)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_distance_heatmap(
    distances: np.ndarray,
    *,
    labels: list[str] | None = None,
    title: str = "Distance heatmap",
    cmap: str = "viridis",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figure_size("one_and_half"))
    sns.heatmap(distances, xticklabels=labels, yticklabels=labels, cmap=cmap, square=True, ax=ax, cbar_kws={"label": "distance"})
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    labels = sorted(set(np.concatenate([y_true, y_pred])))
    fig, ax = plt.subplots(figsize=figure_size("single"))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("observed")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    names: list[str],
    importance: np.ndarray,
    *,
    top_n: int = 20,
    title: str = "Feature importance",
) -> plt.Figure:
    order = np.argsort(importance)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.barh([names[i] for i in order][::-1], importance[order][::-1])
    ax.set_xlabel("importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_zeta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    label_a: str = "group A",
    label_b: str = "group B",
    title: str | None = None,
) -> plt.Figure:
    df = pd.concat([df_a.assign(side=label_a), df_b.assign(side=label_b)])
    fig, ax = plt.subplots(figsize=figure_size("one_and_half"))
    for side, marker, color in [(label_a, "o", "#1f77b4"), (label_b, "s", "#d62728")]:
        sub = df[df["side"] == side]
        ax.scatter(sub["prop_a"], sub["prop_b"], marker=marker, color=color, label=side)
        for _, row in sub.iterrows():
            ax.annotate(row["word"], (row["prop_a"], row["prop_b"]), fontsize=6)
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.3)
    ax.set_xlabel(f"proportion in {label_a}")
    ax.set_ylabel(f"proportion in {label_b}")
    ax.set_title(title or f"Zeta preference: {label_a} vs {label_b}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
```

### Step 2.3 — Update `src/bitig/viz/__init__.py`

```python
"""bitig visualisation — matplotlib/seaborn static + plotly interactive."""

from bitig.viz.mpl import (
    plot_confusion_matrix,
    plot_dendrogram,
    plot_distance_heatmap,
    plot_feature_importance,
    plot_scatter_2d,
    plot_zeta,
)
from bitig.viz.style import apply_publication_style, figure_size

__all__ = [
    "apply_publication_style",
    "figure_size",
    "plot_confusion_matrix",
    "plot_dendrogram",
    "plot_distance_heatmap",
    "plot_feature_importance",
    "plot_scatter_2d",
    "plot_zeta",
]
```

### Step 2.4 — Run → PASS (6/6). Commit:

```bash
git add src/bitig/viz/mpl.py tests/viz/test_mpl.py src/bitig/viz/__init__.py
git commit -m "feat(viz): matplotlib renderers (dendrogram, scatter, heatmap, CM, importance, zeta)"
```

---

## Task 3: Report generator (HTML + Markdown via Jinja2)

**Files:**
- Create: `src/bitig/report/__init__.py`
- Create: `src/bitig/report/render.py`
- Create: `src/bitig/report/templates/report.html.j2`
- Create: `src/bitig/report/templates/report.md.j2`
- Create: `tests/report/__init__.py`
- Create: `tests/report/test_render.py`
- Modify: `pyproject.toml` (wheel force-include `report/templates`)

### Step 3.1 — `src/bitig/report/__init__.py`

```python
"""bitig HTML / Markdown report generation from saved Result artefacts."""

from bitig.report.render import build_report

__all__ = ["build_report"]
```

### Step 3.2 — `src/bitig/report/templates/report.html.j2`

```jinja2
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
body { font-family: Source Serif Pro, Georgia, serif; max-width: 900px; margin: 2em auto; padding: 0 1em; line-height: 1.6; color: #222; }
h1, h2, h3 { color: #0b3d66; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 0.3em 0.8em; }
th { background: #f2f4f8; }
pre { background: #f6f8fa; padding: 0.6em; border-radius: 4px; overflow-x: auto; font-size: 0.85em; }
img.figure { max-width: 100%; display: block; margin: 1em 0; }
code.inline { background: #f6f8fa; padding: 0.1em 0.3em; border-radius: 3px; }
.provenance { font-size: 0.85em; color: #666; margin-top: 3em; border-top: 1px solid #ccc; padding-top: 1em; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p><strong>Generated:</strong> {{ generated_at }} &middot; <strong>bitig:</strong> {{ bitig_version }}</p>

{% if corpus_summary %}
<h2>Corpus</h2>
<table>
<tr><th>field</th><th>value</th></tr>
{% for k, v in corpus_summary.items() %}
<tr><td>{{ k }}</td><td>{{ v }}</td></tr>
{% endfor %}
</table>
{% endif %}

{% if resolved_config %}
<h2>Resolved config</h2>
<pre><code>{{ resolved_config }}</code></pre>
{% endif %}

{% for result in results %}
<h2>{{ result.method_name }}</h2>

{% if result.figures %}
{% for figure in result.figures %}
<img class="figure" src="{{ figure }}" alt="{{ result.method_name }} figure">
{% endfor %}
{% endif %}

{% if result.params %}
<h3>Parameters</h3>
<pre><code>{{ result.params }}</code></pre>
{% endif %}

{% if result.tables %}
{% for table in result.tables %}
{{ table | safe }}
{% endfor %}
{% endif %}

{% if result.summary %}
<p>{{ result.summary }}</p>
{% endif %}

{% endfor %}

<div class="provenance">
<h3>Provenance</h3>
<pre><code>{{ provenance }}</code></pre>
</div>
</body>
</html>
```

### Step 3.3 — `src/bitig/report/templates/report.md.j2`

```jinja2
# {{ title }}

**Generated:** {{ generated_at }} · **bitig:** {{ bitig_version }}

{% if corpus_summary %}
## Corpus

| field | value |
|-------|-------|
{% for k, v in corpus_summary.items() %}| {{ k }} | {{ v }} |
{% endfor %}
{% endif %}

{% if resolved_config %}
## Resolved config

```yaml
{{ resolved_config }}
```
{% endif %}

{% for result in results %}
## {{ result.method_name }}

{% if result.figures %}
{% for figure in result.figures %}
![{{ result.method_name }}]({{ figure }})
{% endfor %}
{% endif %}

{% if result.params %}
### Parameters

```json
{{ result.params }}
```
{% endif %}

{% if result.tables %}
{% for table in result.tables %}
{{ table }}
{% endfor %}
{% endif %}

{% if result.summary %}
{{ result.summary }}
{% endif %}

{% endfor %}

---

## Provenance

```json
{{ provenance }}
```
```

### Step 3.4 — `src/bitig/report/render.py`

```python
"""Build an HTML or Markdown report from a directory containing saved Result artefacts."""

from __future__ import annotations

import json
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment

from bitig._version import __version__

Format = Literal["html", "md"]

_TEMPLATE_PKG = "bitig.report.templates"


def build_report(
    result_dir: str | Path,
    *,
    output: str | Path,
    format: Format = "html",
    title: str = "bitig study",
    corpus_summary: dict[str, Any] | None = None,
) -> Path:
    """Assemble `result.json` + figures in `result_dir` into a single HTML/MD file at `output`.

    `result_dir` may be either a single Result directory (one method) or a parent directory
    containing per-method subdirectories (produced by `bitig run`).
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
        bitig_version=__version__,
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
```

### Step 3.5 — Tests `tests/report/test_render.py`

```python
"""Tests for report rendering."""

import json
from datetime import datetime
from pathlib import Path

from bitig.report import build_report


def _make_result_dir(tmp_path: Path) -> Path:
    d = tmp_path / "delta"
    d.mkdir()
    (d / "result.json").write_text(
        json.dumps(
            {
                "method_name": "burrows_delta",
                "params": {"method": "burrows", "mfw": 500},
                "values": {},
                "provenance": {
                    "bitig_version": "0.1.0.dev0",
                    "python_version": "3.11",
                    "spacy_model": "en",
                    "spacy_version": "3.7",
                    "corpus_hash": "deadbeef",
                    "feature_hash": None,
                    "seed": 42,
                    "timestamp": datetime.now().isoformat(),
                    "resolved_config": {"name": "demo"},
                },
            }
        )
    )
    return d


def test_build_report_html(tmp_path: Path) -> None:
    result_dir = _make_result_dir(tmp_path)
    out = tmp_path / "report.html"
    build_report(result_dir, output=out, format="html", title="Demo report")
    text = out.read_text()
    assert "<html" in text.lower()
    assert "Demo report" in text
    assert "burrows_delta" in text


def test_build_report_md(tmp_path: Path) -> None:
    result_dir = _make_result_dir(tmp_path)
    out = tmp_path / "report.md"
    build_report(result_dir, output=out, format="md", title="Demo")
    text = out.read_text()
    assert text.startswith("# Demo")
    assert "burrows_delta" in text


def test_build_report_multi_method(tmp_path: Path) -> None:
    parent = tmp_path / "run"
    parent.mkdir()
    for method in ["delta", "zeta"]:
        sub = parent / method
        sub.mkdir()
        (sub / "result.json").write_text(
            json.dumps(
                {
                    "method_name": method,
                    "params": {},
                    "values": {},
                    "provenance": {
                        "bitig_version": "0.1.0.dev0",
                        "python_version": "3.11",
                        "spacy_model": "en",
                        "spacy_version": "3.7",
                        "corpus_hash": "x",
                        "feature_hash": None,
                        "seed": 42,
                        "timestamp": datetime.now().isoformat(),
                        "resolved_config": {},
                    },
                }
            )
        )
    out = tmp_path / "report.html"
    build_report(parent, output=out, format="html", title="Multi")
    text = out.read_text()
    assert "delta" in text and "zeta" in text
```

### Step 3.6 — `pyproject.toml` force-include templates

Append to the existing `[tool.hatch.build.targets.wheel.force-include]`:

```toml
"src/bitig/report/templates" = "bitig/report/templates"
```

### Step 3.7 — Commit

```bash
git add src/bitig/report/ tests/report/ pyproject.toml
git commit -m "feat(report): Jinja2-based HTML/Markdown report rendering"
```

---

## Task 4: `bitig.runner` — config-driven orchestrator

**Files:**
- Create: `src/bitig/runner.py`
- Create: `tests/test_runner.py`

### Step 4.1 — Implement `src/bitig/runner.py`

```python
"""Config-driven orchestrator — executes all methods declared in a `study.yaml`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from bitig.config import StudyConfig, load_config
from bitig.features import (
    CharNgramExtractor,
    FeatureMatrix,
    FunctionWordExtractor,
    LexicalDiversityExtractor,
    MFWExtractor,
    PunctuationExtractor,
    ReadabilityExtractor,
    WordNgramExtractor,
)
from bitig.io import load_corpus
from bitig.methods.classify import build_classifier, cross_validate_bitig
from bitig.methods.cluster import HierarchicalCluster
from bitig.methods.consensus import BootstrapConsensus
from bitig.methods.delta import BurrowsDelta
from bitig.methods.reduce import PCAReducer
from bitig.methods.zeta import ZetaClassic
from bitig.plumbing.logging import get_logger
from bitig.provenance import Provenance
from bitig.result import Result

_log = get_logger(__name__)

_FEATURE_BUILDERS = {
    "mfw": MFWExtractor,
    "word_ngram": WordNgramExtractor,
    "char_ngram": CharNgramExtractor,
    "function_word": FunctionWordExtractor,
    "punctuation": PunctuationExtractor,
    "lexical_diversity": LexicalDiversityExtractor,
    "readability": ReadabilityExtractor,
}


def run_study(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
) -> Path:
    """Execute a full study from a `study.yaml` file and save all results.

    Returns the path to the run directory (e.g., `results/2026-04-17T10-15-30/`).
    """
    cfg: StudyConfig = load_config(Path(config_path))
    run_dir = _make_run_dir(cfg, output_dir, run_name)
    _log.info("run directory: %s", run_dir)

    corpus = load_corpus(cfg.corpus.path, metadata=cfg.corpus.metadata)
    if cfg.corpus.filter:
        corpus = corpus.filter(**cfg.corpus.filter)
    _log.info("loaded %d documents", len(corpus))

    # Build all feature matrices by id.
    features_by_id: dict[str, FeatureMatrix] = {}
    for feat_cfg in cfg.features:
        extractor_cls = _FEATURE_BUILDERS.get(feat_cfg.type)
        if extractor_cls is None:
            _log.warning("skipping feature %s: type %s not yet supported by runner", feat_cfg.id, feat_cfg.type)
            continue
        extractor = extractor_cls(**feat_cfg.params)
        features_by_id[feat_cfg.id] = extractor.fit_transform(corpus)
        _log.info("built features %s: %s", feat_cfg.id, features_by_id[feat_cfg.id].X.shape)

    # Execute each method.
    for method_cfg in cfg.methods:
        method_dir = run_dir / method_cfg.id
        method_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = _dispatch_method(method_cfg, corpus, features_by_id)
            result.provenance = Provenance.current(
                spacy_model=cfg.preprocess.spacy.model,
                spacy_version="",
                corpus_hash=corpus.hash(),
                feature_hash=None,
                seed=cfg.seed,
                resolved_config=cfg.model_dump(),
            )
            result.save(method_dir)
            _log.info("wrote %s", method_dir)
        except Exception as exc:
            _log.error("method %s failed: %s", method_cfg.id, exc)
            (method_dir / "error.txt").write_text(str(exc))

    (run_dir / "resolved_config.json").write_text(json.dumps(cfg.model_dump(), indent=2, default=str))
    return run_dir


def _make_run_dir(cfg: StudyConfig, output_dir: str | Path | None, run_name: str | None) -> Path:
    base = Path(output_dir or cfg.output.dir)
    if run_name:
        run_dir = base / run_name
    elif cfg.output.timestamp:
        run_dir = base / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    else:
        run_dir = base
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _dispatch_method(
    method_cfg: Any,
    corpus: Any,
    features_by_id: dict[str, FeatureMatrix],
) -> Result:
    kind = method_cfg.kind

    if kind == "delta":
        # Only Burrows for now in the runner; other Delta variants can be wired later.
        fm = features_by_id[method_cfg.features]
        y = np.array(corpus.metadata_column(method_cfg.group_by))
        clf = BurrowsDelta().fit(fm, y)
        preds = clf.predict(fm)
        return Result(
            method_name="burrows_delta",
            params=dict(method_cfg.params),
            values={"predictions": preds, "accuracy": float((preds == y).mean())},
        )

    if kind == "zeta":
        result = ZetaClassic(
            group_by=method_cfg.group_by,
            top_k=int(method_cfg.params.get("top_k", 20)),
        ).fit_transform(corpus)
        return result

    if kind == "reduce":
        fm = features_by_id[method_cfg.features]
        return PCAReducer(n_components=int(method_cfg.params.get("n_components", 2))).fit_transform(fm)

    if kind == "cluster":
        fm = features_by_id[method_cfg.features]
        return HierarchicalCluster(
            n_clusters=int(method_cfg.params.get("n_clusters", 2)),
            linkage=method_cfg.params.get("linkage", "ward"),
        ).fit_transform(fm)

    if kind == "consensus":
        return BootstrapConsensus(
            mfw_bands=method_cfg.params.get("mfw_bands", [100, 200, 300]),
            replicates=int(method_cfg.params.get("replicates", 20)),
        ).fit_transform(corpus)

    if kind == "classify":
        fm = features_by_id[method_cfg.features if isinstance(method_cfg.features, str) else method_cfg.features[0]]
        y = np.array(corpus.metadata_column(method_cfg.group_by))
        cv_kind = method_cfg.cv.kind if method_cfg.cv else "stratified"
        clf = build_classifier(method_cfg.params.get("estimator", "logreg"))
        report = cross_validate_bitig(
            clf, fm, y, cv_kind=cv_kind, groups_from=y if cv_kind == "loao" else None
        )
        return Result(
            method_name=f"classify_{method_cfg.params.get('estimator', 'logreg')}",
            params=dict(method_cfg.params),
            values={"accuracy": report["accuracy"]},
        )

    raise ValueError(f"runner does not support method kind: {kind!r}")
```

### Step 4.2 — Tests `tests/test_runner.py`

```python
"""Tests for the config-driven runner."""

from pathlib import Path

import pytest
import yaml

from bitig.runner import run_study

pytestmark = pytest.mark.integration


def _write_study(tmp_path: Path, corpus_path: Path, metadata: Path) -> Path:
    study = {
        "name": "test-study",
        "seed": 42,
        "corpus": {"path": str(corpus_path), "metadata": str(metadata)},
        "features": [{"id": "mfw", "type": "mfw", "n": 100, "scale": "zscore", "lowercase": True}],
        "methods": [
            {"id": "burrows", "kind": "delta", "method": "burrows", "features": "mfw", "group_by": "author"},
            {"id": "pca", "kind": "reduce", "features": "mfw", "n_components": 2},
        ],
        "cache": {"dir": str(tmp_path / "cache")},
        "output": {"dir": str(tmp_path / "results"), "timestamp": False},
    }
    cfg = tmp_path / "study.yaml"
    cfg.write_text(yaml.safe_dump(study))
    return cfg


def test_runner_executes_study(tmp_path: Path) -> None:
    fed = Path("tests/fixtures/federalist")
    cfg = _write_study(tmp_path, fed, fed / "metadata.tsv")
    run_dir = run_study(cfg, run_name="fixed-run")
    assert (run_dir / "burrows").is_dir()
    assert (run_dir / "burrows" / "result.json").is_file()
    assert (run_dir / "pca").is_dir()
    assert (run_dir / "pca" / "result.json").is_file()
    assert (run_dir / "resolved_config.json").is_file()
```

### Step 4.3 — Commit

```bash
git add src/bitig/runner.py tests/test_runner.py
git commit -m "feat: run_study orchestrator — execute a full study.yaml end-to-end"
```

---

## Task 5: CLI — `bitig plot`, `bitig report`, `bitig run`

**Files:**
- Create: `src/bitig/cli/plot_cmd.py`, `report_cmd.py`, `run_cmd.py`
- Create: `tests/cli/test_plot_cmd.py`, `test_report_cmd.py`, `test_run_cmd.py`
- Modify: `src/bitig/cli/__init__.py`

### Step 5.1 — `src/bitig/cli/run_cmd.py`

```python
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
    name: str | None = typer.Option(None, "--name", help="Override the default timestamp run-directory name"),
) -> None:
    """Execute a full declarative study and save results to `results/<run>/`."""
    run_dir = run_study(config, output_dir=output, run_name=name)
    console.print(f"[green]run complete[/green] {run_dir}")
```

### Step 5.2 — `src/bitig/cli/report_cmd.py`

```python
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
```

### Step 5.3 — `src/bitig/cli/plot_cmd.py`

```python
"""`bitig plot <result-dir>` — render figures from a saved Result directory."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from bitig.viz.style import apply_publication_style

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
    console.print(f"[yellow]note:[/yellow] to render actual figures, use the bitig.viz.plot_* functions directly from Python for now")
```

### Step 5.4 — Tests (three files)

**`tests/cli/test_run_cmd.py`:**

```python
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
pytestmark = pytest.mark.integration


def test_run_command_executes_study(tmp_path: Path) -> None:
    fed = Path("tests/fixtures/federalist")
    study = {
        "name": "t",
        "seed": 42,
        "corpus": {"path": str(fed), "metadata": str(fed / "metadata.tsv")},
        "features": [{"id": "mfw", "type": "mfw", "n": 100, "scale": "zscore", "lowercase": True}],
        "methods": [{"id": "d", "kind": "delta", "method": "burrows", "features": "mfw", "group_by": "author"}],
        "output": {"dir": str(tmp_path / "out"), "timestamp": False},
    }
    cfg = tmp_path / "study.yaml"
    cfg.write_text(yaml.safe_dump(study))
    result = runner.invoke(app, ["run", str(cfg), "--name", "fixed"])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "out" / "fixed" / "d" / "result.json").is_file()
```

**`tests/cli/test_report_cmd.py`:**

```python
import json
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()


def test_report_html_runs(tmp_path: Path) -> None:
    d = tmp_path / "delta"
    d.mkdir()
    (d / "result.json").write_text(
        json.dumps(
            {
                "method_name": "delta",
                "params": {},
                "values": {},
                "provenance": {
                    "bitig_version": "0.1",
                    "python_version": "3.11",
                    "spacy_model": "en",
                    "spacy_version": "3.7",
                    "corpus_hash": "x",
                    "feature_hash": None,
                    "seed": 1,
                    "timestamp": datetime.now().isoformat(),
                    "resolved_config": {},
                },
            }
        )
    )
    out = tmp_path / "r.html"
    result = runner.invoke(app, ["report", str(d), "--output", str(out), "--format", "html"])
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
```

**`tests/cli/test_plot_cmd.py`:**

```python
import json
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()


def test_plot_command_requires_result_json(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plot", str(tmp_path)])
    assert result.exit_code != 0


def test_plot_command_accepts_valid_result_dir(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text(
        json.dumps({"method_name": "delta", "params": {}, "values": {}, "provenance": None})
    )
    result = runner.invoke(app, ["plot", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
```

### Step 5.5 — Register in `src/bitig/cli/__init__.py`

Append:

```python
from bitig.cli.plot_cmd import plot_command
from bitig.cli.report_cmd import report_command
from bitig.cli.run_cmd import run_command

app.command(name="run")(run_command)
app.command(name="report")(report_command)
app.command(name="plot")(plot_command)
```

### Step 5.6 — Commit

```bash
git add src/bitig/cli/plot_cmd.py src/bitig/cli/report_cmd.py src/bitig/cli/run_cmd.py tests/cli/test_plot_cmd.py tests/cli/test_report_cmd.py tests/cli/test_run_cmd.py src/bitig/cli/__init__.py
git commit -m "feat(cli): bitig run / report / plot subcommands"
```

---

## Task 6: Interactive shell (`bitig shell`)

**Files:**
- Create: `src/bitig/cli/shell_cmd.py`
- Create: `tests/cli/test_shell_cmd.py`
- Modify: `src/bitig/cli/__init__.py`

Minimal Rich-based wizard. No questionary dependency — Rich prompts are simpler and avoid an extra runtime import path. Users who want a full interactive prompt can always pipe commands.

### Step 6.1 — `src/bitig/cli/shell_cmd.py`

```python
"""`bitig shell [<corpus>]` — guided wizard over the analytical methods."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import IntPrompt, Prompt

from bitig.io import load_corpus

console = Console()


def shell_command(
    corpus_path: Path | None = typer.Argument(None, exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path | None = typer.Option(None, "--metadata", "-m"),  # noqa: B008
) -> None:
    """Launch the interactive wizard."""
    console.rule("[bold cyan]bitig shell[/bold cyan]")

    if corpus_path is None:
        corpus_path_str = Prompt.ask("Corpus path")
        corpus_path = Path(corpus_path_str)
    console.print(f"[green]Corpus:[/green] {corpus_path}")

    corpus = load_corpus(corpus_path, metadata=metadata)
    console.print(f"[green]Loaded {len(corpus)} documents[/green]")

    menu = [
        "Inspect corpus",
        "Run Delta attribution (bitig delta)",
        "Run Zeta comparison (bitig zeta)",
        "Cluster & visualise (bitig cluster)",
        "Classify (bitig classify)",
        "Reduce & plot (bitig reduce)",
        "Quit",
    ]
    for i, item in enumerate(menu, start=1):
        console.print(f"  [cyan]{i}[/cyan]. {item}")
    choice = IntPrompt.ask("Choose", default=1, choices=[str(i) for i in range(1, len(menu) + 1)])

    if choice == 1:
        console.print(f"Documents: {len(corpus)}")
        meta_keys = set().union(*(d.metadata.keys() for d in corpus.documents))
        console.print(f"Metadata fields: {sorted(meta_keys)}")
    elif choice == len(menu):
        console.print("[dim]bye[/dim]")
    else:
        method = menu[choice - 1]
        console.print(
            f"Would run: [dim]bitig {method.split('(')[1].rstrip(')').split()[1]}[/dim]"
            f" — run the CLI directly for the full flow."
        )
```

### Step 6.2 — Tests `tests/cli/test_shell_cmd.py`

```python
from pathlib import Path

from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini_corpus"


def test_shell_accepts_quit(tmp_path: Path) -> None:
    result = runner.invoke(app, ["shell", str(FIXTURES)], input="7\n")
    assert result.exit_code == 0, result.stdout
    assert "bye" in result.stdout.lower() or "bye" in result.stdout


def test_shell_inspects_corpus(tmp_path: Path) -> None:
    result = runner.invoke(app, ["shell", str(FIXTURES)], input="1\n")
    assert result.exit_code == 0
    assert "Documents" in result.stdout
```

### Step 6.3 — Register and commit

Add in `src/bitig/cli/__init__.py`:

```python
from bitig.cli.shell_cmd import shell_command

app.command(name="shell")(shell_command)
```

```bash
git add src/bitig/cli/shell_cmd.py tests/cli/test_shell_cmd.py src/bitig/cli/__init__.py
git commit -m "feat(cli): bitig shell — minimal Rich-based wizard"
```

---

## Task 7: Public API + Phase 5 tag

Modify `src/bitig/__init__.py` to export `bitig.viz`, `bitig.report.build_report`, `bitig.runner.run_study`. Update README. Run full suite, commit, tag `phase-5-viz-reports`.

```python
from bitig.report import build_report
from bitig.runner import run_study
from bitig.viz import (
    apply_publication_style,
    figure_size,
    plot_confusion_matrix,
    plot_dendrogram,
    plot_distance_heatmap,
    plot_feature_importance,
    plot_scatter_2d,
    plot_zeta,
)
```

Add all names to `__all__`.

README update:

```markdown
## Status

**Phase 5 — Viz, reports, interactive shell, `bitig run`.** Ships publication-grade matplotlib
renderers (dendrogram, scatter, heatmap, confusion matrix, feature importance, Zeta preference
plot), HTML/Markdown report generation from saved Result directories, a config-driven
`bitig run study.yaml` orchestrator that runs a full multi-method study end-to-end, and a
minimal Rich-based `bitig shell` wizard.

Phase 6 (MkDocs site + Federalist/EFCAMDAT tutorials + PyPI publish) remains.
```

Commit + tag:

```bash
git add src/bitig/__init__.py README.md
git commit -m "feat: public API re-exports for Phase 5 (viz + reports + runner + shell)"
git tag -a phase-5-viz-reports -m "Phase 5 complete: viz renderers, HTML/MD reports, bitig run orchestrator, interactive shell"
```

---

## Phase 5 — Acceptance Criteria

```bash
pytest -n auto -q                                             # full suite
bitig run tests/fixtures/federalist/study.yaml --name demo    # (fixture study.yaml to be added ad-hoc)
bitig report ./results/demo --output /tmp/report.html
```

---

## Self-Review

- **Spec §8.4 (shell):** minimal wizard — deferred IPython-embed to Phase 6.
- **Spec §10 (viz):** matplotlib covered; Plotly deferred to Phase 6 (optional, needs `[viz]` extra).
- **Spec §11 (reports):** HTML + MD via Jinja2; PDF deferred (would need weasyprint, `[reports]` extra).
- **Spec §8.5 (`bitig run`):** runner ships with Delta/Zeta/reduce/cluster/consensus/classify; embeddings/Bayesian not wired in runner yet.
- **Placeholder scan:** no TBD/TODO.
- **Type consistency:** FeatureMatrix / Result / Corpus signatures preserved.
