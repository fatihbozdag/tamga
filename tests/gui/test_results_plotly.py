"""Tests for the Results page plotly dispatcher (pure data path, no UI)."""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui")
pytest.importorskip("plotly")

import numpy as np

from bitig.gui.pages.results import _plotly_figures_for
from bitig.result import Result


def test_pca_dispatch_yields_scatter_and_biplot() -> None:
    coords = np.array([[1.0, 0.5], [-0.5, 1.2], [0.3, -0.7]])
    loadings = np.array([[0.6, -0.4, 0.1], [0.2, 0.7, -0.5]])
    result = Result(
        method_name="pca",
        values={
            "coordinates": coords,
            "document_ids": ["a", "b", "c"],
            "loadings": loadings,
            "feature_names": ["the", "of", "and"],
            "explained_variance_ratio": np.array([0.6, 0.3]),
        },
    )
    figs = _plotly_figures_for(result)
    titles = [t for t, _ in figs]
    assert "pca_scatter" in titles
    assert "pca_biplot" in titles


def test_umap_dispatch_yields_scatter_only() -> None:
    coords = np.array([[1.0, 0.5], [-0.5, 1.2]])
    result = Result(
        method_name="umap",
        values={"coordinates": coords, "document_ids": ["a", "b"]},
    )
    figs = _plotly_figures_for(result)
    assert [t for t, _ in figs] == ["umap_scatter"]


def test_hierarchical_dispatch_returns_dendrogram() -> None:
    linkage = np.array([[0.0, 1.0, 0.5, 2.0]])
    result = Result(
        method_name="hierarchical",
        values={"linkage": linkage, "document_ids": ["a", "b"]},
    )
    figs = _plotly_figures_for(result)
    assert [t for t, _ in figs] == ["dendrogram"]


def test_bayesian_dispatch_returns_posterior_heatmap() -> None:
    proba = np.array([[0.7, 0.3], [0.2, 0.8]])
    result = Result(
        method_name="bayesian_authorship",
        values={
            "proba": proba,
            "classes": ["alice", "bob"],
            "document_ids": ["doc1", "doc2"],
        },
    )
    figs = _plotly_figures_for(result)
    assert [t for t, _ in figs] == ["posterior"]


def test_classify_dispatch_returns_confusion_matrix() -> None:
    result = Result(
        method_name="classify_logreg",
        values={
            "y_true": np.array(["alice", "bob", "alice"]),
            "predictions": np.array(["alice", "bob", "bob"]),
        },
    )
    figs = _plotly_figures_for(result)
    assert [t for t, _ in figs] == ["confusion_matrix"]


def test_unknown_method_returns_empty() -> None:
    result = Result(method_name="some_unsupported", values={})
    assert _plotly_figures_for(result) == []


def test_pca_without_loadings_returns_scatter_only() -> None:
    result = Result(
        method_name="pca",
        values={
            "coordinates": np.array([[1.0, 0.5], [-0.5, 1.2]]),
            "document_ids": ["a", "b"],
        },
    )
    figs = _plotly_figures_for(result)
    assert [t for t, _ in figs] == ["pca_scatter"]


def test_classify_without_y_true_returns_empty() -> None:
    result = Result(
        method_name="classify_logreg",
        values={"predictions": np.array(["alice", "bob"])},
    )
    assert _plotly_figures_for(result) == []
