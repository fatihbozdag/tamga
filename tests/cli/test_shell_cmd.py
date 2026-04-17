from pathlib import Path

from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini_corpus"


def test_shell_accepts_quit(tmp_path: Path) -> None:
    result = runner.invoke(app, ["shell", str(FIXTURES)], input="7\n")
    assert result.exit_code == 0, result.stdout
    assert "bye" in result.stdout.lower() or "bye" in result.stdout


def test_shell_inspects_corpus(tmp_path: Path) -> None:
    result = runner.invoke(app, ["shell", str(FIXTURES)], input="1\n")
    assert result.exit_code == 0
    assert "Documents" in result.stdout
