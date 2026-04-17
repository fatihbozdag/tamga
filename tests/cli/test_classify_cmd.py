from pathlib import Path

import pytest
from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"
pytestmark = pytest.mark.integration


def test_classify_logreg_loao_runs() -> None:
    result = runner.invoke(
        app,
        [
            "classify",
            str(FED),
            "--metadata",
            str(FED / "metadata.tsv"),
            "--estimator",
            "logreg",
            "--group-by",
            "author",
            "--cv-kind",
            "loao",
            "--mfw",
            "200",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "accuracy" in result.stdout.lower()
