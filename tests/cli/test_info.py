"""Tests for `tamga info`."""

from pathlib import Path

from typer.testing import CliRunner

from tamga import __version__
from tamga.cli import app

runner = CliRunner()


def test_info_reports_version() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_info_reports_spacy_version() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "spacy" in result.stdout.lower()


def test_info_displays_corpus_language(tmp_path: Path, monkeypatch) -> None:
    """When a `study.yaml` sits in the cwd, `tamga info` surfaces its language."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("merhaba")
    (tmp_path / "study.yaml").write_text(
        "name: t\ncorpus:\n  path: corpus\npreprocess:\n  language: tr\n"
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.stdout
    assert "language" in result.stdout.lower()
    assert "tr" in result.stdout


def test_info_no_study_yaml_still_works(tmp_path: Path, monkeypatch) -> None:
    """When no `study.yaml` exists, `tamga info` still prints tamga/spacy versions."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.stdout
    assert __version__ in result.stdout
