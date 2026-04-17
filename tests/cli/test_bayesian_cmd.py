from pathlib import Path

import pytest
from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_bayesian_attributes_fed_49_to_madison() -> None:
    result = runner.invoke(
        app,
        [
            "bayesian",
            str(FED),
            "--metadata",
            str(FED / "metadata.tsv"),
            "--group-by",
            "author",
            "--test-filter",
            "role=test",
            "--mfw",
            "500",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "fed_49" in result.stdout
    assert "Madison" in result.stdout
