"""Tests for matplotlib renderers. Verifies figures are created and savable."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from tamga.viz.mpl import (
    plot_confusion_matrix,
    plot_dendrogram,
    plot_distance_heatmap,
    plot_feature_importance,
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
