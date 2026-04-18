"""Tests for `tamga init`."""

from pathlib import Path

from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()


def test_init_creates_project(tmp_path: Path) -> None:
    target = tmp_path / "my-study"
    result = runner.invoke(app, ["init", "my-study", "--target", str(target)])
    assert result.exit_code == 0, result.stdout
    assert (target / "study.yaml").is_file()
    assert (target / "corpus").is_dir()


def test_init_refuses_to_clobber(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    (target / "file").write_text("hi")
    result = runner.invoke(app, ["init", "existing", "--target", str(target)])
    assert result.exit_code != 0


def test_init_force_fills_in(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    (target / "file").write_text("hi")
    result = runner.invoke(app, ["init", "existing", "--target", str(target), "--force"])
    assert result.exit_code == 0
    assert (target / "study.yaml").is_file()
    assert (target / "file").read_text() == "hi"


def test_init_writes_language_into_study_yaml(tmp_path: Path) -> None:
    target = tmp_path / "mystudy"
    result = runner.invoke(app, ["init", "mystudy", "--target", str(target), "--language", "tr"])
    assert result.exit_code == 0, result.stdout
    study_yaml = (target / "study.yaml").read_text()
    assert "language: tr" in study_yaml


def test_init_default_language_is_english(tmp_path: Path) -> None:
    target = tmp_path / "default-lang"
    result = runner.invoke(app, ["init", "default-lang", "--target", str(target)])
    assert result.exit_code == 0, result.stdout
    study_yaml = (target / "study.yaml").read_text()
    assert "language: en" in study_yaml


def test_init_rejects_unknown_language(tmp_path: Path) -> None:
    target = tmp_path / "badlang"
    result = runner.invoke(app, ["init", "badlang", "--target", str(target), "--language", "xx"])
    assert result.exit_code != 0
