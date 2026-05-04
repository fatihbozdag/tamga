"""Ingest page — load a corpus directory + optional metadata TSV."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from bitig.gui.filepicker import is_native_available, pick_file, pick_folder
from bitig.gui.layout import page_shell
from bitig.gui.state import get_state
from bitig.io import load_corpus
from bitig.languages import get_language

_LANGUAGE_CHOICES = {
    "en": "English",
    "tr": "Türkçe",
    "de": "Deutsch",
    "es": "Español",
    "fr": "Français",
}

_METADATA_CANDIDATES = ("metadata.tsv", "metadata.csv", "metadata.txt")


def _detect_metadata(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    for name in _METADATA_CANDIDATES:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


@ui.page("/")
@ui.page("/ingest")
def ingest_page() -> None:
    state = get_state()

    with page_shell("Ingest"):
        ui.markdown("### 1. Point bitig at a corpus directory")
        ui.markdown(
            "A corpus is a directory of plain-text documents. Each ``.txt`` file is one "
            "document; filenames become document ids. If you have a metadata TSV "
            "(columns: ``id``, ``author``, ``year``, …), pass it too."
        )

        native = is_native_available()
        browse_disabled_hint = (
            "Browse works only in native mode (without --no-native). "
            "Type or paste the absolute path instead."
        )

        with ui.row().classes("w-full items-end gap-2"):
            corpus_input = ui.input(
                label="Corpus directory",
                placeholder="/path/to/corpus",
                value=str(state.corpus_path or ""),
            ).classes("flex-1")

            def _autofill_metadata_from_corpus() -> None:
                folder_str = (corpus_input.value or "").strip()
                if not folder_str or (metadata_input.value or "").strip():
                    return
                detected = _detect_metadata(Path(folder_str).expanduser())
                if detected is not None:
                    metadata_input.value = str(detected)
                    ui.notify(f"auto-detected metadata: {detected.name}", type="positive")

            async def browse_corpus() -> None:
                chosen = await pick_folder("Select corpus folder")
                if chosen:
                    corpus_input.value = chosen
                    _autofill_metadata_from_corpus()

            browse_corpus_btn = ui.button(
                "Browse folder", icon="folder_open", on_click=browse_corpus
            ).props("outline")
            browse_corpus_btn.set_enabled(native)
            if not native:
                browse_corpus_btn.tooltip(browse_disabled_hint)

        with ui.row().classes("w-full items-end gap-2"):
            metadata_input = ui.input(
                label="Metadata TSV (optional)",
                placeholder="/path/to/metadata.tsv",
                value=str(state.metadata_path or ""),
            ).classes("flex-1")

            async def browse_metadata() -> None:
                chosen = await pick_file(
                    "Select metadata TSV",
                    file_types=("TSV files (*.tsv;*.txt)", "All files (*.*)"),
                )
                if chosen:
                    metadata_input.value = chosen

            browse_metadata_btn = ui.button(
                "Browse file", icon="description", on_click=browse_metadata
            ).props("outline")
            browse_metadata_btn.set_enabled(native)
            if not native:
                browse_metadata_btn.tooltip(browse_disabled_hint)

        corpus_input.on_value_change(lambda _e: _autofill_metadata_from_corpus())

        language_select = ui.select(
            options=_LANGUAGE_CHOICES,
            value=state.language,
            label="Language",
        ).classes("w-64")

        status = ui.markdown("")

        def load() -> None:
            corpus_path_str = corpus_input.value.strip()
            if not corpus_path_str:
                status.set_content("**error:** please give a corpus directory path.")
                return
            corpus_path = Path(corpus_path_str).expanduser()
            if not corpus_path.is_dir():
                status.set_content(f"**error:** `{corpus_path}` is not a directory.")
                return
            try:
                get_language(language_select.value)
            except ValueError as exc:
                status.set_content(f"**error:** {exc}")
                return

            metadata_path: Path | None = None
            metadata_path_str = metadata_input.value.strip()
            if metadata_path_str:
                metadata_path = Path(metadata_path_str).expanduser()
                if not metadata_path.is_file():
                    status.set_content(f"**error:** metadata file `{metadata_path}` not found.")
                    return

            try:
                corpus = load_corpus(
                    corpus_path,
                    metadata=metadata_path,
                    language=language_select.value,
                )
            except Exception as exc:
                status.set_content(f"**error:** {type(exc).__name__}: {exc}")
                return

            state.corpus_path = corpus_path
            state.metadata_path = metadata_path
            state.language = language_select.value
            state.corpus_doc_count = len(corpus)
            meta_cols: set[str] = set()
            for doc in corpus.documents:
                meta_cols.update(doc.metadata.keys())
            state.corpus_metadata_cols = sorted(meta_cols)

            cols_str = ", ".join(f"`{c}`" for c in state.corpus_metadata_cols) or "_(none)_"
            status.set_content(
                f"**loaded:** {state.corpus_doc_count} documents "
                f"(language: {state.language}); metadata columns: {cols_str}"
            )

        with ui.row().classes("gap-2 mt-2"):
            ui.button("Load corpus", on_click=load).props("color=primary")
            ui.button(
                "Next: Study →",
                on_click=lambda: ui.navigate.to("/study"),
            ).props("flat")
