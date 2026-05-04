from pathlib import Path

import pytest
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"
pytestmark = pytest.mark.integration


def test_zeta_classic_runs_on_federalist() -> None:
    result = runner.invoke(
        app,
        [
            "zeta",
            str(FED),
            "--metadata",
            str(FED / "metadata.tsv"),
            "--group-by",
            "author",
            "--variant",
            "classic",
            "--top-k",
            "5",
            "--group-a",
            "Hamilton",
            "--group-b",
            "Madison",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "preferred in Hamilton" in result.stdout
    assert "preferred in Madison" in result.stdout


def test_zeta_rejects_unknown_variant() -> None:
    result = runner.invoke(
        app,
        ["zeta", str(FED), "--metadata", str(FED / "metadata.tsv"), "--variant", "bogus"],
    )
    assert result.exit_code != 0
