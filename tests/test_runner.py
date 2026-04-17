"""Tests for the config-driven runner."""

from pathlib import Path

import pytest
import yaml

from tamga.runner import run_study

pytestmark = pytest.mark.integration


def _write_study(tmp_path: Path, corpus_path: Path, metadata: Path) -> Path:
    study = {
        "name": "test-study",
        "seed": 42,
        "corpus": {"path": str(corpus_path), "metadata": str(metadata)},
        "features": [{"id": "mfw", "type": "mfw", "n": 100, "scale": "zscore", "lowercase": True}],
        "methods": [
            {
                "id": "burrows",
                "kind": "delta",
                "method": "burrows",
                "features": "mfw",
                "group_by": "author",
            },
            {
                "id": "pca",
                "kind": "reduce",
                "features": "mfw",
                "n_components": 2,
            },
        ],
        "cache": {"dir": str(tmp_path / "cache")},
        "output": {"dir": str(tmp_path / "results"), "timestamp": False},
    }
    cfg = tmp_path / "study.yaml"
    cfg.write_text(yaml.safe_dump(study))
    return cfg


def test_runner_executes_study(tmp_path: Path) -> None:
    fed = Path("tests/fixtures/federalist")
    cfg = _write_study(tmp_path, fed, fed / "metadata.tsv")
    run_dir = run_study(cfg, run_name="fixed-run")
    assert (run_dir / "burrows").is_dir()
    assert (run_dir / "burrows" / "result.json").is_file()
    assert (run_dir / "pca").is_dir()
    assert (run_dir / "pca" / "result.json").is_file()
    assert (run_dir / "resolved_config.json").is_file()
