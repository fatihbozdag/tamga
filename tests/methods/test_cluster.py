"""Tests for clustering methods."""

from __future__ import annotations

import numpy as np
import pytest

from tamga.features import FeatureMatrix
from tamga.methods.cluster import HDBSCANCluster, HierarchicalCluster, KMeansCluster


def _fm() -> FeatureMatrix:
    rng = np.random.default_rng(42)
    cluster_a = rng.standard_normal((10, 3)) + np.array([0, 0, 0])
    cluster_b = rng.standard_normal((10, 3)) + np.array([5, 5, 5])
    X = np.vstack([cluster_a, cluster_b])
    return FeatureMatrix(
        X=X,
        document_ids=[f"d{i}" for i in range(20)],
        feature_names=["x", "y", "z"],
        feature_type="test",
    )


def test_hierarchical_ward_returns_linkage() -> None:
    r = HierarchicalCluster(n_clusters=2, linkage="ward").fit_transform(_fm())
    assert "labels" in r.values
    assert "linkage" in r.values
    assert r.values["linkage"].shape == (19, 4)  # noqa: RUF003 — n-1 merges × 4 columns


def test_hierarchical_finds_two_well_separated_clusters() -> None:
    r = HierarchicalCluster(n_clusters=2, linkage="ward").fit_transform(_fm())
    labels = r.values["labels"]
    # The first 10 docs should all share a label; the last 10 should all share the other.
    assert len(set(labels[:10])) == 1
    assert len(set(labels[10:])) == 1
    assert labels[0] != labels[10]


def test_kmeans_finds_two_clusters() -> None:
    r = KMeansCluster(n_clusters=2, random_state=42).fit_transform(_fm())
    labels = r.values["labels"]
    assert set(labels) == {0, 1}


def test_hdbscan_returns_labels_including_noise() -> None:
    r = HDBSCANCluster(min_cluster_size=3).fit_transform(_fm())
    labels = r.values["labels"]
    # HDBSCAN labels noise points as -1; our two blobs should produce at least one cluster.
    assert any(label >= 0 for label in labels)


@pytest.mark.parametrize("linkage", ["ward", "average", "complete", "single"])
def test_hierarchical_all_linkages(linkage: str) -> None:
    r = HierarchicalCluster(n_clusters=2, linkage=linkage).fit_transform(_fm())
    assert r.values["labels"].shape == (20,)
