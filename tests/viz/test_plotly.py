"""Smoke tests for the plotly renderers. Skipped if `plotly` is not installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage

pytest.importorskip("plotly")  # everything below requires the optional `bitig[interactive]` extra.

import pandas as pd

from bitig.viz.plotly import (
    plot_confusion_matrix,
    plot_dendrogram,
    plot_distance_heatmap,
    plot_imposters_scores,
    plot_pca_biplot,
    plot_posterior_heatmap,
    plot_reliability_diagram,
    plot_rolling_delta,
    plot_scatter_2d,
)


def test_plot_dendrogram_writes_html(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    Z = linkage(rng.standard_normal((6, 4)), method="ward")
    fig = plot_dendrogram(Z, labels=[f"d{i}" for i in range(6)])
    out = tmp_path / "dendro.html"
    fig.write_html(out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_scatter_2d_with_groups(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((20, 2))
    fig = plot_scatter_2d(
        coords,
        labels=[f"d{i}" for i in range(20)],
        groups=["A"] * 10 + ["B"] * 10,
    )
    fig.write_html(tmp_path / "scatter.html")


def test_plot_distance_heatmap(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3))
    D = np.abs(X[:, None, :] - X[None, :, :]).mean(axis=2)
    fig = plot_distance_heatmap(D, labels=[f"d{i}" for i in range(5)])
    fig.write_html(tmp_path / "heatmap.html")


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    y_true = np.array(["A", "A", "B", "B", "A", "B"])
    y_pred = np.array(["A", "B", "B", "B", "A", "A"])
    fig = plot_confusion_matrix(y_true, y_pred)
    fig.write_html(tmp_path / "cm.html")


def test_plot_pca_biplot(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((20, 2))
    loadings = rng.standard_normal((2, 8))
    names = ["the", "and", "of", "to", "a", "in", "that", "is"]
    fig = plot_pca_biplot(
        coords,
        loadings,
        names,
        labels=[f"d{i}" for i in range(20)],
        groups=["A"] * 10 + ["B"] * 10,
        explained_variance_ratio=np.array([0.4, 0.25]),
        top_n=4,
    )
    fig.write_html(tmp_path / "biplot.html")


def test_plot_pca_biplot_validates_shapes() -> None:
    coords = np.random.default_rng(0).standard_normal((10, 2))
    with pytest.raises(ValueError, match="loadings columns"):
        plot_pca_biplot(coords, np.zeros((2, 5)), ["only", "three"])
    with pytest.raises(ValueError, match="coordinates must be"):
        plot_pca_biplot(np.zeros((10, 1)), np.zeros((2, 3)), ["a", "b", "c"])


def test_plot_posterior_heatmap(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(size=(6, 3))
    proba = raw / raw.sum(axis=1, keepdims=True)
    fig = plot_posterior_heatmap(
        proba,
        document_ids=[f"d{i}" for i in range(6)],
        classes=["Hamilton", "Madison", "Jay"],
    )
    fig.write_html(tmp_path / "posterior.html")


def test_plot_posterior_heatmap_validates_shapes() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(size=(4, 2))
    proba = raw / raw.sum(axis=1, keepdims=True)
    with pytest.raises(ValueError, match="proba rows"):
        plot_posterior_heatmap(proba, document_ids=["a", "b"], classes=["x", "y"])
    with pytest.raises(ValueError, match="proba columns"):
        plot_posterior_heatmap(proba, document_ids=["a", "b", "c", "d"], classes=["x"])
    with pytest.raises(ValueError, match="proba must be 2-D"):
        plot_posterior_heatmap(np.zeros(4), document_ids=["a"], classes=["x"])


def test_plot_rolling_delta(tmp_path: Path) -> None:
    rows = []
    for w in range(5):
        rows.extend(
            [
                {
                    "doc_id": "doc1",
                    "window_idx": w,
                    "window_start_token": w * 100,
                    "window_end_token": (w + 1) * 100,
                    "nearest_author": "alice" if w % 2 == 0 else "bob",
                    "distance_alice": 0.4 + 0.05 * w,
                    "distance_bob": 0.5 - 0.04 * w,
                }
            ]
        )
    table = pd.DataFrame(rows)
    fig = plot_rolling_delta(table)
    fig.write_html(tmp_path / "rolling.html")


def test_plot_rolling_delta_validates() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        plot_rolling_delta(pd.DataFrame())
    with pytest.raises(ValueError, match="x_axis"):
        plot_rolling_delta(
            pd.DataFrame([{"doc_id": "d", "distance_alice": 0.1}]),
            x_axis="missing_column",
        )
    with pytest.raises(ValueError, match="distance_<author>"):
        plot_rolling_delta(
            pd.DataFrame([{"doc_id": "d", "window_start_token": 0, "nearest_author": "alice"}])
        )


def test_plot_imposters_scores(tmp_path: Path) -> None:
    table = pd.DataFrame(
        [
            {"target_id": "doc1", "candidate": "hamilton", "score": 0.82, "verified": True},
            {"target_id": "doc2", "candidate": "hamilton", "score": 0.31, "verified": False},
            {"target_id": "doc3", "candidate": "hamilton", "score": 0.55, "verified": True},
        ]
    )
    fig = plot_imposters_scores(table, threshold=0.5)
    fig.write_html(tmp_path / "imposters.html")


def test_plot_imposters_scores_validates() -> None:
    with pytest.raises(ValueError, match="at least one target"):
        plot_imposters_scores(pd.DataFrame())
    with pytest.raises(ValueError, match="missing required columns"):
        plot_imposters_scores(pd.DataFrame([{"target_id": "d"}]))


def test_plot_reliability_diagram(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 60
    classes = np.array(["alice", "bob"])
    y_true = rng.choice(classes, size=n)
    raw = rng.uniform(size=(n, 2))
    y_proba = raw / raw.sum(axis=1, keepdims=True)
    fig = plot_reliability_diagram(y_true, y_proba, classes=classes, n_bins=5)
    fig.write_html(tmp_path / "reliability.html")
