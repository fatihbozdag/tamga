"""Tests for `bitig delta`."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"


pytestmark = pytest.mark.integration


def test_delta_burrows_on_federalist() -> None:
    result = runner.invoke(
        app,
        [
            "delta",
            str(FED),
            "--method",
            "burrows",
            "--mfw",
            "500",
            "--metadata",
            str(FED / "metadata.tsv"),
            "--group-by",
            "author",
            "--test-filter",
            "role=test",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # Fed 49 → Madison.
    assert "Madison" in result.stdout
    assert "fed_49" in result.stdout


def test_delta_rejects_unknown_method() -> None:
    result = runner.invoke(
        app, ["delta", str(FED), "--method", "bogus", "--metadata", str(FED / "metadata.tsv")]
    )
    assert result.exit_code != 0
