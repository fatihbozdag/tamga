"""Tests for matplotlib renderers. Verifies figures are created and savable."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from tamga.viz.mpl import (
    plot_bootstrap_consensus_tree,
    plot_confusion_matrix,
    plot_dendrogram,
    plot_distance_heatmap,
    plot_feature_importance,
    plot_imposters_scores,
    plot_pca_biplot,
    plot_reliability_diagram,
    plot_rolling_delta,
    plot_scatter_2d,
    plot_zeta,
)


def test_plot_dendrogram_saves_png(tmp_path: Path):
    X = np.random.default_rng(42).standard_normal((8, 4))
    Z = linkage(X, method="ward")
    labels = [f"d{i}" for i in range(8)]
    out = tmp_path / "dendro.png"
    fig = plot_dendrogram(Z, labels=labels)
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_scatter_2d_with_groups(tmp_path: Path):
    coords = np.random.default_rng(42).standard_normal((20, 2))
    groups = ["A"] * 10 + ["B"] * 10
    labels = [f"d{i}" for i in range(20)]
    fig = plot_scatter_2d(coords, labels=labels, groups=groups)
    fig.savefig(tmp_path / "scatter.png")


def test_plot_distance_heatmap_sym_matrix(tmp_path: Path):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((5, 3))
    D = np.abs(X[:, None, :] - X[None, :, :]).mean(axis=2)
    fig = plot_distance_heatmap(D, labels=[f"d{i}" for i in range(5)])
    fig.savefig(tmp_path / "heatmap.png")


def test_plot_confusion_matrix(tmp_path: Path):
    y_true = np.array(["A", "A", "B", "B", "A", "B"])
    y_pred = np.array(["A", "B", "B", "B", "A", "A"])
    fig = plot_confusion_matrix(y_true, y_pred)
    fig.savefig(tmp_path / "confusion.png")


def test_plot_feature_importance(tmp_path: Path):
    names = ["the", "and", "of", "to", "a"]
    importance = np.array([0.5, 0.3, 0.2, 0.15, 0.1])
    fig = plot_feature_importance(names, importance)
    fig.savefig(tmp_path / "importance.png")


def test_plot_zeta(tmp_path: Path):
    df_a = pd.DataFrame(
        {"word": ["alpha", "beta"], "zeta": [0.8, 0.6], "prop_a": [1.0, 0.8], "prop_b": [0.2, 0.2]}
    )
    df_b = pd.DataFrame(
        {
            "word": ["gamma", "delta"],
            "zeta": [-0.7, -0.5],
            "prop_a": [0.1, 0.2],
            "prop_b": [0.8, 0.7],
        }
    )
    fig = plot_zeta(df_a, df_b, label_a="A", label_b="B")
    fig.savefig(tmp_path / "zeta.png")


def test_plot_bootstrap_consensus_tree(tmp_path: Path):
    # Two well-supported clades, a weak one, all 5 leaves accounted for.
    support = {
        "a,b": 0.95,
        "a,b,c": 0.55,
        "d,e": 0.85,
    }
    leaves = ["a", "b", "c", "d", "e"]
    out = tmp_path / "bct.png"
    fig = plot_bootstrap_consensus_tree(support, leaves)
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_rolling_delta_two_targets(tmp_path: Path):
    table = pd.DataFrame(
        [
            {
                "doc_id": "alpha",
                "window_idx": i,
                "window_start_token": i * 100,
                "window_end_token": i * 100 + 400,
                "nearest_author": "A" if i % 2 == 0 else "B",
                "distance_A": 0.5 + 0.05 * i,
                "distance_B": 0.6 - 0.03 * i,
            }
            for i in range(6)
        ]
        + [
            {
                "doc_id": "beta",
                "window_idx": i,
                "window_start_token": i * 100,
                "window_end_token": i * 100 + 400,
                "nearest_author": "B",
                "distance_A": 0.7,
                "distance_B": 0.3,
            }
            for i in range(4)
        ]
    )
    fig = plot_rolling_delta(table)
    out = tmp_path / "rolling.png"
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_imposters_scores_mixed_decisions(tmp_path: Path):
    table = pd.DataFrame(
        [
            {"target_id": "doc_a", "candidate": "Alice", "score": 0.9, "verified": True},
            {"target_id": "doc_b", "candidate": "Alice", "score": 0.55, "verified": True},
            {"target_id": "doc_c", "candidate": "Alice", "score": 0.2, "verified": False},
            {"target_id": "doc_d", "candidate": "Alice", "score": 0.05, "verified": False},
        ]
    )
    fig = plot_imposters_scores(table, threshold=0.5)
    out = tmp_path / "imp.png"
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_pca_biplot_renders(tmp_path: Path):
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
        top_n=5,
    )
    out = tmp_path / "biplot.png"
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_pca_biplot_validates_shapes():
    import pytest

    coords = np.random.default_rng(0).standard_normal((10, 2))
    with pytest.raises(ValueError, match="loadings columns"):
        plot_pca_biplot(coords, np.zeros((2, 5)), ["only", "three"])
    with pytest.raises(ValueError, match="coordinates must be"):
        plot_pca_biplot(np.zeros((10, 1)), np.zeros((2, 3)), ["a", "b", "c"])
    with pytest.raises(ValueError, match="loadings must be"):
        plot_pca_biplot(coords, np.zeros((1, 3)), ["a", "b", "c"])


def test_plot_reliability_diagram_smoke(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.choice(["A", "B"], size=n)
    raw = rng.uniform(size=(n, 2))
    y_proba = raw / raw.sum(axis=1, keepdims=True)
    classes = np.array(["A", "B"])
    fig = plot_reliability_diagram(y_true, y_proba, classes=classes, n_bins=8)
    out = tmp_path / "reliability.png"
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_imposters_scores_rejects_empty():
    import pytest

    with pytest.raises(ValueError, match="at least one target"):
        plot_imposters_scores(pd.DataFrame())


def test_plot_rolling_delta_rejects_empty_table():
    import pytest

    with pytest.raises(ValueError, match="at least one window"):
        plot_rolling_delta(pd.DataFrame())


def test_plot_bootstrap_consensus_tree_rejects_singleton():
    import pytest

    with pytest.raises(ValueError, match="at least 2"):
        plot_bootstrap_consensus_tree({}, ["only"])
