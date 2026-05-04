from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
pytestmark = pytest.mark.integration


def test_run_command_executes_study(tmp_path: Path) -> None:
    fed = Path("tests/fixtures/federalist")
    study = {
        "name": "t",
        "seed": 42,
        "corpus": {"path": str(fed), "metadata": str(fed / "metadata.tsv")},
        "features": [{"id": "mfw", "type": "mfw", "n": 100, "scale": "zscore", "lowercase": True}],
        "methods": [
            {
                "id": "d",
                "kind": "delta",
                "method": "burrows",
                "features": "mfw",
                "group_by": "author",
            }
        ],
        "output": {"dir": str(tmp_path / "out"), "timestamp": False},
    }
    cfg = tmp_path / "study.yaml"
    cfg.write_text(yaml.safe_dump(study))
    result = runner.invoke(app, ["run", str(cfg), "--name", "fixed"])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "out" / "fixed" / "d" / "result.json").is_file()
