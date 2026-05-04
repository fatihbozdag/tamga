"""Tests for `bitig ingest`."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()

FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini_corpus"


pytestmark = pytest.mark.spacy


def test_ingest_parses_corpus_without_metadata(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        app,
        [
            "ingest",
            str(FIXTURES),
            "--cache-dir",
            str(cache_dir),
            "--spacy-model",
            "en_core_web_sm",
        ],
    )
    assert result.exit_code == 0, result.stdout
    docbin_dir = cache_dir / "docbin"
    assert len(list(docbin_dir.glob("*.docbin"))) == 4


def test_ingest_uses_metadata_when_provided(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "ingest",
            str(FIXTURES),
            "--metadata",
            str(FIXTURES / "metadata.tsv"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--spacy-model",
            "en_core_web_sm",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "4 documents" in result.stdout


def test_ingest_reports_cache_hits_on_rerun(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    first = runner.invoke(
        app,
        ["ingest", str(FIXTURES), "--cache-dir", str(cache_dir), "--spacy-model", "en_core_web_sm"],
    )
    assert first.exit_code == 0

    second = runner.invoke(
        app,
        ["ingest", str(FIXTURES), "--cache-dir", str(cache_dir), "--spacy-model", "en_core_web_sm"],
    )
    assert second.exit_code == 0
    assert "cached" in second.stdout.lower() or "cache" in second.stdout.lower()


def test_ingest_cli_accepts_language_flag(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from bitig.cli import app

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("merhaba dünya")

    # Stub out SpacyPipeline to avoid model loads during unit tests.
    recorded: dict[str, object] = {}
    from bitig.cli import ingest_cmd

    class _StubPipe:
        def __init__(self, **kwargs: object) -> None:
            recorded["pipe_kwargs"] = kwargs
            self.cache = type("C", (), {"keys": lambda self: [], "size_bytes": lambda self: 0})()

        def parse(self, corpus) -> None:
            recorded["corpus_language"] = corpus.language

    monkeypatch.setattr(ingest_cmd, "SpacyPipeline", _StubPipe)

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", str(corpus_dir), "--language", "tr", "--no-strict"])
    assert result.exit_code == 0, result.stdout
    assert recorded["corpus_language"] == "tr"
