"""Tests for corpus ingestion from a filesystem directory + metadata TSV."""

from pathlib import Path

import pytest

from tamga.corpus import Corpus
from tamga.io import load_corpus, load_metadata

FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini_corpus"


def test_load_metadata_parses_tsv():
    rows = load_metadata(FIXTURES / "metadata.tsv")
    assert len(rows) == 4
    assert rows["alice_one.txt"] == {"author": "Alice", "group": "native", "year": "2019"}


def test_load_metadata_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_metadata(FIXTURES / "does_not_exist.tsv")


def test_load_corpus_from_directory_returns_corpus():
    corpus = load_corpus(FIXTURES)
    assert isinstance(corpus, Corpus)
    assert len(corpus) == 4


def test_load_corpus_attaches_metadata():
    corpus = load_corpus(FIXTURES, metadata=FIXTURES / "metadata.tsv")
    alice_docs = [d for d in corpus if d.metadata.get("author") == "Alice"]
    assert len(alice_docs) == 2


def test_load_corpus_without_metadata_still_works():
    corpus = load_corpus(FIXTURES)
    assert len(corpus) == 4
    for d in corpus:
        assert d.metadata == {}


def test_load_corpus_id_is_filename_stem():
    corpus = load_corpus(FIXTURES)
    ids = {d.id for d in corpus}
    assert ids == {"alice_one", "alice_two", "bob_one", "bob_two"}


def test_load_corpus_sorts_documents_deterministically():
    # Loading the same directory twice yields Documents in the same order → stable corpus hash.
    c1 = load_corpus(FIXTURES)
    c2 = load_corpus(FIXTURES)
    assert [d.id for d in c1] == [d.id for d in c2]


def test_load_corpus_raises_on_missing_metadata_row(tmp_path: Path):
    # Fixture a corpus where metadata does not cover every file.
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    (tmp_path / "meta.tsv").write_text("filename\tauthor\na.txt\tAlice\n")
    with pytest.raises(ValueError, match="missing metadata"):
        load_corpus(tmp_path, metadata=tmp_path / "meta.tsv", strict=True)


def test_load_corpus_non_strict_allows_missing_metadata(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    (tmp_path / "meta.tsv").write_text("filename\tauthor\na.txt\tAlice\n")
    corpus = load_corpus(tmp_path, metadata=tmp_path / "meta.tsv", strict=False)
    assert len(corpus) == 2


def test_load_corpus_stamps_language_argument(tmp_path) -> None:
    from tamga.io import load_corpus

    (tmp_path / "d.txt").write_text("hola mundo")
    corpus = load_corpus(tmp_path, language="es", strict=False)
    assert corpus.language == "es"


def test_load_corpus_defaults_to_english(tmp_path) -> None:
    from tamga.io import load_corpus

    (tmp_path / "d.txt").write_text("hello")
    corpus = load_corpus(tmp_path, strict=False)
    assert corpus.language == "en"


def test_load_corpus_rejects_unknown_language_code(tmp_path) -> None:
    import pytest

    from tamga.io import load_corpus

    (tmp_path / "d.txt").write_text("x")
    with pytest.raises(ValueError, match="Unknown language code"):
        load_corpus(tmp_path, language="xx", strict=False)
