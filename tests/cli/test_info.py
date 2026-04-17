"""Tests for `tamga info`."""

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
