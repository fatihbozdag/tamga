"""State module is the only non-UI-bound piece of the GUI; test it directly."""

from __future__ import annotations

from pathlib import Path

from bitig.gui.state import GuiState, get_state, reset_state


def test_get_state_returns_singleton() -> None:
    reset_state()
    a = get_state()
    b = get_state()
    assert a is b


def test_default_state_is_empty() -> None:
    reset_state()
    s = get_state()
    assert s.corpus_path is None
    assert s.metadata_path is None
    assert s.language == "en"
    assert s.corpus_doc_count == 0
    assert s.corpus_metadata_cols == []
    assert s.study_path is None
    assert s.run_dir is None


def test_state_mutation_persists_across_get(tmp_path: Path) -> None:
    reset_state()
    s = get_state()
    s.corpus_path = tmp_path / "corpus"
    s.language = "tr"
    s2 = get_state()
    assert s2.corpus_path == tmp_path / "corpus"
    assert s2.language == "tr"


def test_reset_state_clears_values() -> None:
    s = get_state()
    s.language = "de"
    reset_state()
    assert get_state().language == "en"


def test_dataclass_construction() -> None:
    s = GuiState(language="fr", corpus_doc_count=42)
    assert s.language == "fr"
    assert s.corpus_doc_count == 42
    assert s.corpus_metadata_cols == []
