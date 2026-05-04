"""Results page -- inspect the contents of the run output directory.

Supports a static / interactive (plotly) toggle: when interactive is selected and
the optional `bitig[interactive]` extra is installed, supported method types are
re-rendered as plotly HTML; unsupported methods or import failures fall back to
the static PNGs the runner already wrote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

from bitig.gui.layout import page_shell
from bitig.gui.state import PlotFormat, get_state
from bitig.result import Result


def _as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value)
    except (TypeError, ValueError):
        return None


def _plotly_figures_for(result: Result) -> list[tuple[str, Any]]:
    """Return [(title, plotly.Figure), ...] for the supported method types.

    Empty list means the method has no interactive equivalent and the caller
    should fall back to static PNGs.
    """
    import bitig.viz.plotly as bplotly

    name = result.method_name
    values = result.values
    out: list[tuple[str, Any]] = []

    if name in {"pca", "mds", "tsne", "umap"}:
        coords = _as_array(values.get("coordinates"))
        if coords is not None and coords.ndim == 2 and coords.shape[1] >= 2:
            doc_ids = values.get("document_ids")
            labels = list(doc_ids) if isinstance(doc_ids, list) else None
            fig = bplotly.plot_scatter_2d(coords, labels=labels, title=f"{name.upper()} projection")
            out.append((f"{name}_scatter", fig))
            if name == "pca":
                loadings = _as_array(values.get("loadings"))
                feature_names = values.get("feature_names")
                if (
                    loadings is not None
                    and isinstance(feature_names, list)
                    and loadings.ndim == 2
                    and loadings.shape[0] >= 2
                ):
                    evr = _as_array(values.get("explained_variance_ratio"))
                    biplot = bplotly.plot_pca_biplot(
                        coords,
                        loadings,
                        list(feature_names),
                        labels=labels,
                        explained_variance_ratio=evr,
                    )
                    out.append(("pca_biplot", biplot))
        return out

    if name == "hierarchical":
        linkage = _as_array(values.get("linkage"))
        if linkage is not None and linkage.ndim == 2 and linkage.shape[1] == 4:
            doc_ids = values.get("document_ids")
            labels = list(doc_ids) if isinstance(doc_ids, list) else None
            fig = bplotly.plot_dendrogram(linkage, labels=labels)
            out.append(("dendrogram", fig))
        return out

    if name == "bayesian_authorship":
        proba = _as_array(values.get("proba"))
        classes = values.get("classes")
        doc_ids = values.get("document_ids")
        if (
            proba is not None
            and isinstance(classes, list)
            and isinstance(doc_ids, list)
            and proba.ndim == 2
        ):
            fig = bplotly.plot_posterior_heatmap(proba, list(doc_ids), list(classes))
            out.append(("posterior", fig))
        return out

    if name.startswith("classify_"):
        y_true = _as_array(values.get("y_true"))
        preds = _as_array(values.get("predictions"))
        if y_true is not None and preds is not None and y_true.shape == preds.shape:
            fig = bplotly.plot_confusion_matrix(y_true, preds)
            out.append(("confusion_matrix", fig))
        return out

    return out


def _try_render_plotly(method_dir: Path) -> bool:
    """Attempt to render the method via plotly. Return True on success, False to fall back."""
    result_file = method_dir / "result.json"
    if not result_file.is_file():
        return False
    try:
        result = Result.from_json(result_file)
    except (OSError, json.JSONDecodeError, KeyError):
        return False
    try:
        figures = _plotly_figures_for(result)
    except ImportError as exc:
        ui.markdown(
            f"_Interactive plots require `bitig[interactive]` -- {exc}. Showing static PNGs._"
        )
        return False
    except Exception as exc:
        ui.markdown(f"_Interactive render failed for {method_dir.name} ({exc}); showing PNG._")
        return False
    if not figures:
        return False
    for _title, fig in figures:
        ui.html(fig.to_html(include_plotlyjs="cdn", full_html=False)).classes("w-full")
    return True


def _render_method_dir(method_dir: Path, plot_format: PlotFormat) -> None:
    result_file = method_dir / "result.json"
    if not result_file.is_file():
        ui.markdown(f"_{method_dir.name}: no result.json_")
        return
    try:
        payload = json.loads(result_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        ui.markdown(f"**{method_dir.name}:** failed to read result.json -- {exc}")
        return

    with ui.expansion(method_dir.name, icon="science").classes("w-full"):
        method_name = payload.get("method_name", method_dir.name)
        ui.markdown(f"**method:** `{method_name}`")
        params = payload.get("params") or {}
        if params:
            ui.markdown("**params:**")
            ui.json_editor({"content": {"json": params}}).classes("w-full").props(
                "read-only mode=text"
            )
        values = payload.get("values") or {}
        scalar_values = {k: v for k, v in values.items() if isinstance(v, (int, float, str, bool))}
        if scalar_values:
            rows = [{"metric": k, "value": v} for k, v in scalar_values.items()]
            ui.table(
                columns=[
                    {"name": "metric", "label": "metric", "field": "metric"},
                    {"name": "value", "label": "value", "field": "value"},
                ],
                rows=rows,
            ).classes("w-full")
        for parquet in sorted(method_dir.glob("table_*.parquet")):
            try:
                df = pd.read_parquet(parquet)
            except Exception as exc:
                ui.markdown(f"_could not read {parquet.name}: {exc}_")
                continue
            ui.markdown(f"**{parquet.stem}** ({len(df)} rows)")
            preview = df.head(100)
            ui.table(
                columns=[{"name": c, "label": c, "field": c} for c in preview.columns],
                rows=preview.to_dict(orient="records"),
            ).classes("w-full")

        rendered_interactive = False
        if plot_format == "interactive":
            rendered_interactive = _try_render_plotly(method_dir)
        if not rendered_interactive:
            for png in sorted(method_dir.glob("*.png")):
                ui.image(str(png)).classes("w-full max-w-xl")


def _set_format_and_refresh(value: PlotFormat) -> None:
    state = get_state()
    state.plot_format = value
    ui.navigate.reload()


@ui.page("/results")
def results_page() -> None:
    state = get_state()

    with page_shell("Results"):
        if state.run_dir is None or not state.run_dir.is_dir():
            ui.markdown("_No run yet._ Go to **Run** and execute a study first.")
            ui.button("← Run", on_click=lambda: ui.navigate.to("/run")).props("flat")
            return

        ui.markdown(f"**Run directory:** `{state.run_dir}`")
        with ui.row().classes("items-center gap-2"):
            ui.markdown("**Plot format:**")
            ui.toggle(
                {"static": "Static (PNG)", "interactive": "Interactive (plotly)"},
                value=state.plot_format,
                on_change=lambda e: _set_format_and_refresh(e.value),
            )
        method_dirs = sorted(d for d in state.run_dir.iterdir() if d.is_dir())
        if not method_dirs:
            ui.markdown("_Run directory contains no method subdirectories yet._")
            return
        for method_dir in method_dirs:
            _render_method_dir(method_dir, state.plot_format)
