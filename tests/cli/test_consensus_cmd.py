from pathlib import Path

import pytest
from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"
pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_consensus_writes_newick(tmp_path: Path) -> None:
    out = tmp_path / "consensus.nwk"
    result = runner.invoke(
        app,
        [
            "consensus",
            str(FED),
            "--metadata",
            str(FED / "metadata.tsv"),
            "--bands",
            "100,200",
            "--replicates",
            "3",
            "--seed",
            "42",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    text = out.read_text()
    assert text.endswith(";")
    assert len(text) > 10
