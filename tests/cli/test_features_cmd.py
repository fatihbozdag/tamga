"""Tests for `bitig features`."""

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini_corpus"


def test_features_mfw_writes_parquet(tmp_path: Path) -> None:
    out = tmp_path / "feats.parquet"
    result = runner.invoke(
        app,
        [
            "features",
            str(FIXTURES),
            "--type",
            "mfw",
            "--n",
            "20",
            "--scale",
            "none",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pd.read_parquet(out)
    assert df.shape[0] == 4
    assert df.shape[1] > 0


def test_features_rejects_unknown_type() -> None:
    result = runner.invoke(app, ["features", str(FIXTURES), "--type", "nonsense"])
    assert result.exit_code != 0
    assert "unknown" in result.stdout.lower()


def test_features_char_ngram_works(tmp_path: Path) -> None:
    out = tmp_path / "char.parquet"
    result = runner.invoke(
        app,
        [
            "features",
            str(FIXTURES),
            "--type",
            "char_ngram",
            "--n",
            "3",
            "--scale",
            "none",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
