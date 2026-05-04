"""Tests for dimensionality reducers."""

from __future__ import annotations

import numpy as np
import pytest

from bitig.features import FeatureMatrix
from bitig.methods.reduce import MDSReducer, PCAReducer, TSNEReducer, UMAPReducer


def _fm(n: int = 10, d: int = 5) -> FeatureMatrix:
    rng = np.random.default_rng(42)
    return FeatureMatrix(
        X=rng.standard_normal((n, d)),
        document_ids=[f"d{i}" for i in range(n)],
        feature_names=[f"f{j}" for j in range(d)],
        feature_type="test",
    )


def test_pca_reduces_to_target_components() -> None:
    r = PCAReducer(n_components=2).fit_transform(_fm(10, 5))
    assert r.values["coordinates"].shape == (10, 2)
    assert "explained_variance_ratio" in r.values


def test_mds_reduces_to_target_components() -> None:
    r = MDSReducer(n_components=2, random_state=42).fit_transform(_fm(10, 5))
    assert r.values["coordinates"].shape == (10, 2)


def test_tsne_reduces_to_2d() -> None:
    # t-SNE requires perplexity < n_samples — use small perplexity for tiny fixture.
    r = TSNEReducer(n_components=2, perplexity=3.0, random_state=42).fit_transform(_fm(10, 5))
    assert r.values["coordinates"].shape == (10, 2)


@pytest.mark.slow
def test_umap_reduces_to_target_components() -> None:
    r = UMAPReducer(n_components=2, random_state=42).fit_transform(_fm(20, 5))
    assert r.values["coordinates"].shape == (20, 2)


def test_reduce_result_contains_document_ids() -> None:
    fm = _fm(5, 4)
    r = PCAReducer(n_components=2).fit_transform(fm)
    assert r.values["document_ids"] == fm.document_ids
