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
