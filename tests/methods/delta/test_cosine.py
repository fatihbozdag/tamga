"""Tests for CosineDelta."""

import numpy as np
from sklearn.base import is_classifier

from tamga.features import FeatureMatrix
from tamga.methods.delta.cosine import CosineDelta


def _fm(X: np.ndarray) -> FeatureMatrix:  # noqa: N803
    return FeatureMatrix(
        X=X,
        document_ids=[f"d{i}" for i in range(X.shape[0])],
        feature_names=[f"f{j}" for j in range(X.shape[1])],
        feature_type="zscored-mfw",
    )


def test_cosine_attributes_to_nearest_centroid() -> None:
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array(["A", "B"])
    preds = CosineDelta().fit(_fm(X), y).predict(_fm(np.array([[1.0, 0.1], [0.1, 1.0]])))
    assert list(preds) == ["A", "B"]


def test_cosine_handles_zero_vectors_gracefully() -> None:
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array(["A", "B"])
    clf = CosineDelta().fit(_fm(X), y)
    preds = clf.predict(_fm(np.array([[0.0, 0.0]])))
    # Must not raise; specific output isn't asserted (both centroids equidistant at 1.0).
    assert len(preds) == 1


def test_cosine_is_sklearn_compatible() -> None:
    assert is_classifier(CosineDelta())
