"""Study page — choose a feature + method and write a minimal study.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from nicegui import ui

from tamga.gui.layout import page_shell
from tamga.gui.state import get_state

_METHODS = {
    "delta": "Burrows Delta — nearest-author-centroid (small, known-author corpora)",
    "cluster": "Hierarchical cluster — unsupervised grouping",
    "consensus": "Bootstrap consensus — stability across many MFW bands",
    "classify": "Classify (logreg) — cross-validated supervised",
}
_FEATURES = {
    "mfw": "Most-frequent words",
    "char_ngram": "Character n-grams",
    "word_ngram": "Word n-grams",
    "function_word": "Function words",
}


@ui.page("/study")
def study_page() -> None:
    state = get_state()

    with page_shell("Study"):
        if state.corpus_path is None:
            ui.markdown("_No corpus loaded yet._ Go to **Ingest** and load a corpus first.")
            ui.button("← Ingest", on_click=lambda: ui.navigate.to("/ingest")).props("flat")
            return

        ui.markdown(
            f"**Corpus:** `{state.corpus_path}` ({state.corpus_doc_count} documents, "
            f"language: {state.language})"
        )

        ui.markdown("### 2. Configure a study")
        method_select = ui.select(options=_METHODS, value="delta", label="Method").classes("w-full")
        feature_select = ui.select(options=_FEATURES, value="mfw", label="Feature").classes(
            "w-full"
        )

        with ui.row().classes("gap-4"):
            top_n_input = ui.number(label="Feature top_n", value=200, min=50, max=2000).classes(
                "w-40"
            )
            ngram_n_input = ui.number(label="n (for n-grams)", value=3, min=1, max=6).classes(
                "w-32"
            )

        group_by_input = ui.input(
            label="Group-by metadata column (for delta / classify)",
            value="author" if "author" in state.corpus_metadata_cols else "",
        ).classes("w-64")

        default_path = (
            (state.corpus_path.parent / "study.yaml") if state.corpus_path else Path("study.yaml")
        )
        out_path_input = ui.input(
            label="Save study.yaml to",
            value=str(state.study_path or default_path),
        ).classes("w-full")

        status = ui.markdown("")

        def save() -> None:
            feat_id = feature_select.value
            feat_type = feat_id
            feat_params: dict[str, object] = {"top_n": int(top_n_input.value)}
            if feat_type in ("char_ngram", "word_ngram"):
                feat_params["n"] = int(ngram_n_input.value)

            method_kind = method_select.value
            method_params: dict[str, object] = {}
            method_cfg: dict[str, object] = {
                "id": f"{method_kind}_1",
                "kind": method_kind,
                "params": method_params,
            }
            if method_kind != "consensus":
                method_cfg["features"] = feat_id
            if method_kind in ("delta", "classify"):
                if not group_by_input.value.strip():
                    status.set_content("**error:** `group_by` is required for delta/classify.")
                    return
                method_cfg["group_by"] = group_by_input.value.strip()
            if method_kind == "classify":
                method_params["estimator"] = "logreg"

            study = {
                "seed": 42,
                "preprocess": {"language": state.language, "spacy": {}},
                "corpus": {
                    "path": str(state.corpus_path),
                    **({"metadata": str(state.metadata_path)} if state.metadata_path else {}),
                },
                "features": [{"id": feat_id, "type": feat_type, "params": feat_params}],
                "methods": [method_cfg],
                "output": {"dir": "runs", "timestamp": True},
            }

            out_path = Path(out_path_input.value.strip()).expanduser()
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(yaml.safe_dump(study, sort_keys=False))
            except OSError as exc:
                status.set_content(f"**error:** {exc}")
                return
            state.study_path = out_path
            status.set_content(f"**saved:** `{out_path}`")

        with ui.row().classes("gap-2 mt-2"):
            ui.button("Save study.yaml", on_click=save).props("color=primary")
            ui.button(
                "Next: Run →",
                on_click=lambda: ui.navigate.to("/run"),
            ).props("flat")
