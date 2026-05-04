import json
from pathlib import Path

from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()


def test_plot_command_requires_result_json(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plot", str(tmp_path)])
    assert result.exit_code != 0


def test_plot_command_accepts_valid_result_dir(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text(
        json.dumps({"method_name": "delta", "params": {}, "values": {}, "provenance": None})
    )
    result = runner.invoke(app, ["plot", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
