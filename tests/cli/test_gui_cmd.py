"""Smoke tests for ``tamga gui`` — the full GUI is not launched here."""

from __future__ import annotations

from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()


def test_gui_subcommand_appears_in_top_level_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "gui" in result.stdout


def test_gui_help_describes_native_and_no_native() -> None:
    result = runner.invoke(app, ["gui", "--help"])
    assert result.exit_code == 0
    assert "--no-native" in result.stdout
    assert "--host" in result.stdout
    assert "--port" in result.stdout


def test_gui_fails_cleanly_without_extra(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """If nicegui is not installed, the command exits 1 with a helpful error."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "nicegui" or name.startswith("nicegui."):
            raise ImportError("No module named 'nicegui'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = runner.invoke(app, ["gui"])
    assert result.exit_code == 1
    assert "gui" in result.stdout.lower()
