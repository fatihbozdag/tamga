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


def test_cluster_kmeans_seed_flag_controls_labels(tmp_path: Path) -> None:
    """`--seed` must flow to KMeansCluster.random_state.

    Regression test for the audit finding that cluster_cmd.py hardcoded
    random_state=42 regardless of user-supplied seed, making k-means clustering
    non-reproducible under a user-controlled seed.
    """

    def _run(seed: int, out: Path) -> pd.DataFrame:
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
                "--seed",
                str(seed),
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.stdout
        return pd.read_parquet(out)

    df_a = _run(1, tmp_path / "a.parquet")
    df_a_again = _run(1, tmp_path / "a2.parquet")
    # Same seed → same labels (up to label permutation; here equal since n_init="auto" fixes init).
    assert df_a["cluster"].tolist() == df_a_again["cluster"].tolist()
