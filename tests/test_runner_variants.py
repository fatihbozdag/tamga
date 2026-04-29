"""Round-trip tests for each runner-dispatchable method variant on mini_corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml


def _study_yaml(
    tmp_path: Path,
    corpus_dir: Path,
    *,
    kind: str,
    feat_type: str = "mfw",
    feat_params: dict[str, Any] | None = None,
    method_params: dict[str, Any] | None = None,
    group_by: str | None = None,
) -> Path:
    method_cfg: dict[str, Any] = {
        "id": f"{kind}_1",
        "kind": kind,
        "params": dict(method_params or {}),
    }
    if kind != "consensus":
        method_cfg["features"] = "feat1"
    if group_by:
        method_cfg["group_by"] = group_by
    study = {
        "seed": 42,
        "preprocess": {"language": "en", "spacy": {}},
        "corpus": {
            "path": str(corpus_dir),
            "metadata": str(corpus_dir / "metadata.tsv"),
        },
        "features": [{"id": "feat1", "type": feat_type, "params": dict(feat_params or {"n": 50})}],
        "methods": [method_cfg],
        "output": {"dir": str(tmp_path / "runs"), "timestamp": False},
    }
    cfg_path = tmp_path / "study.yaml"
    cfg_path.write_text(yaml.safe_dump(study, sort_keys=False))
    return cfg_path


@pytest.mark.parametrize(
    "variant",
    ["burrows", "cosine", "argamon_linear", "quadratic", "eder", "eder_simple"],
)
def test_delta_variant_runs(tmp_path: Path, mini_corpus_dir: Path, variant: str) -> None:
    from tamga.runner import run_study

    cfg = _study_yaml(
        tmp_path,
        mini_corpus_dir,
        kind="delta",
        method_params={"variant": variant},
        group_by="author",
    )
    run_dir = run_study(cfg, output_dir=tmp_path / "runs", run_name="r")
    result_path = run_dir / "delta_1" / "result.json"
    assert result_path.exists(), f"no result.json for variant={variant}"
    payload = json.loads(result_path.read_text())
    assert payload["method_name"] == f"delta_{variant}"
    assert "accuracy" in payload["values"]


@pytest.mark.parametrize(
    ("variant", "extra"),
    [
        ("pca", {}),
        ("mds", {"random_state": 0}),
        # mini_corpus is 4 docs → perplexity must be < n_samples for t-SNE
        # and n_neighbors < n_samples for UMAP.
        ("tsne", {"perplexity": 2.0, "random_state": 0}),
        ("umap", {"n_neighbors": 2, "random_state": 0}),
    ],
)
def test_reduce_variant_runs(
    tmp_path: Path,
    mini_corpus_dir: Path,
    variant: str,
    extra: dict[str, Any],
) -> None:
    from tamga.runner import run_study

    cfg = _study_yaml(
        tmp_path,
        mini_corpus_dir,
        kind="reduce",
        method_params={"variant": variant, "n_components": 2, **extra},
    )
    run_dir = run_study(cfg, output_dir=tmp_path / "runs", run_name="r")
    result_path = run_dir / "reduce_1" / "result.json"
    assert result_path.exists(), f"no result.json for variant={variant}"
    payload = json.loads(result_path.read_text())
    assert payload["method_name"] == variant
    assert "coordinates" in payload["values"]


@pytest.mark.parametrize(
    ("variant", "extra"),
    [
        ("hierarchical", {"n_clusters": 2, "linkage": "ward"}),
        ("kmeans", {"n_clusters": 2, "random_state": 0}),
        # 4 docs → min_cluster_size must drop below default 5 to produce non-noise.
        ("hdbscan", {"min_cluster_size": 2}),
    ],
)
def test_cluster_variant_runs(
    tmp_path: Path,
    mini_corpus_dir: Path,
    variant: str,
    extra: dict[str, Any],
) -> None:
    from tamga.runner import run_study

    cfg = _study_yaml(
        tmp_path,
        mini_corpus_dir,
        kind="cluster",
        method_params={"variant": variant, **extra},
    )
    run_dir = run_study(cfg, output_dir=tmp_path / "runs", run_name="r")
    result_path = run_dir / "cluster_1" / "result.json"
    assert result_path.exists(), f"no result.json for variant={variant}"
    payload = json.loads(result_path.read_text())
    assert payload["method_name"] == variant
    assert "labels" in payload["values"]
