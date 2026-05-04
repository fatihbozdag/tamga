from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"
pytestmark = pytest.mark.integration


def test_reduce_pca_writes_parquet(tmp_path: Path) -> None:
    out = tmp_path / "r.parquet"
    result = runner.invoke(
        app,
        [
            "reduce",
            str(FED),
            "--metadata",
            str(FED / "metadata.tsv"),
            "--method",
            "pca",
            "--n-components",
            "2",
            "--mfw",
            "200",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pd.read_parquet(out)
    assert df.shape == (13, 2)


def test_reduce_rejects_unknown_method(tmp_path: Path) -> None:
    out = tmp_path / "r.parquet"
    result = runner.invoke(
        app,
        [
            "reduce",
            str(FED),
            "--metadata",
            str(FED / "metadata.tsv"),
            "--method",
            "bogus",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code != 0
