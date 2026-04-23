"""Results page — inspect the contents of the run output directory."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from nicegui import ui

from tamga.gui.layout import page_shell
from tamga.gui.state import get_state


def _render_method_dir(method_dir: Path) -> None:
    result_file = method_dir / "result.json"
    if not result_file.is_file():
        ui.markdown(f"_{method_dir.name}: no result.json_")
        return
    try:
        payload = json.loads(result_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        ui.markdown(f"**{method_dir.name}:** failed to read result.json — {exc}")
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
        for png in sorted(method_dir.glob("*.png")):
            ui.image(str(png)).classes("w-full max-w-xl")


@ui.page("/results")
def results_page() -> None:
    state = get_state()

    with page_shell("Results"):
        if state.run_dir is None or not state.run_dir.is_dir():
            ui.markdown("_No run yet._ Go to **Run** and execute a study first.")
            ui.button("← Run", on_click=lambda: ui.navigate.to("/run")).props("flat")
            return

        ui.markdown(f"**Run directory:** `{state.run_dir}`")
        method_dirs = sorted(d for d in state.run_dir.iterdir() if d.is_dir())
        if not method_dirs:
            ui.markdown("_Run directory contains no method subdirectories yet._")
            return
        for method_dir in method_dirs:
            _render_method_dir(method_dir)
