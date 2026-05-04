"""Tests for the CLI skeleton."""

from typer.testing import CliRunner

from bitig import __version__
from bitig.cli import app

runner = CliRunner()


def test_cli_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "bitig" in result.stdout.lower()


def test_cli_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # Typer exits with 0 (help) or 2 (usage-error), both acceptable — check output instead.
    assert "Usage" in result.stdout or "usage" in result.stdout
