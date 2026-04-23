"""Ingest page — load a corpus directory + optional metadata TSV."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from tamga.gui.layout import page_shell
from tamga.gui.state import get_state
from tamga.io import load_corpus
from tamga.languages import get_language

_LANGUAGE_CHOICES = {
    "en": "English",
    "tr": "Türkçe",
    "de": "Deutsch",
    "es": "Español",
    "fr": "Français",
}


@ui.page("/")
@ui.page("/ingest")
def ingest_page() -> None:
    state = get_state()

    with page_shell("Ingest"):
        ui.markdown("### 1. Point tamga at a corpus directory")
        ui.markdown(
            "A corpus is a directory of plain-text documents. Each ``.txt`` file is one "
            "document; filenames become document ids. If you have a metadata TSV "
            "(columns: ``id``, ``author``, ``year``, …), pass it too."
        )

        corpus_input = ui.input(
            label="Corpus directory",
            placeholder="/path/to/corpus",
            value=str(state.corpus_path or ""),
        ).classes("w-full")

        metadata_input = ui.input(
            label="Metadata TSV (optional)",
            placeholder="/path/to/metadata.tsv",
            value=str(state.metadata_path or ""),
        ).classes("w-full")

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
