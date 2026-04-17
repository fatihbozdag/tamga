from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"
pytestmark = pytest.mark.integration


def test_cluster_hierarchical_writes_parquet(tmp_path: Path) -> None:
    out = tmp_path / "c.parquet"
    result = runner.invoke(
        app,
        [
            "cluster",
            str(FED),
            "--metadata",
            str(FED / "metadata.tsv"),
            "--method",
            "hierarchical",
            "--n-clusters",
            "3",
            "--mfw",
            "200",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pd.read_parquet(out)
    assert df.shape == (13, 2)
    assert set(df.columns) == {"document_id", "cluster"}


def test_cluster_kmeans_runs(tmp_path: Path) -> None:
    out = tmp_path / "k.parquet"
    result = runner.invoke(
        app,
        [
            "cluster",
            str(FED),
            "--metadata",
            str(FED / "metadata.tsv"),
            "--method",
            "kmeans",
            "--n-clusters",
            "3",
            "--mfw",
            "200",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
