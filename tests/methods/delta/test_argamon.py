"""Tests for ArgamonLinearDelta and QuadraticDelta."""

import numpy as np
from sklearn.base import is_classifier

from tamga.features import FeatureMatrix
from tamga.methods.delta.argamon import ArgamonLinearDelta, QuadraticDelta


def _fm(X: np.ndarray) -> FeatureMatrix:  # noqa: N803
    return FeatureMatrix(
        X=X,
        document_ids=[f"d{i}" for i in range(X.shape[0])],
        feature_names=[f"f{j}" for j in range(X.shape[1])],
        feature_type="zscored-mfw",
    )


def test_argamon_linear_attributes_to_nearest_centroid() -> None:
    X = np.array([[0.0, 0.0], [5.0, 5.0]])
    y = np.array(["A", "B"])
    preds = ArgamonLinearDelta().fit(_fm(X), y).predict(_fm(np.array([[0.0, 0.1], [4.9, 5.0]])))
    assert list(preds) == ["A", "B"]


def test_quadratic_attributes_to_nearest_centroid() -> None:
    X = np.array([[0.0, 0.0], [5.0, 5.0]])
    y = np.array(["A", "B"])
    preds = QuadraticDelta().fit(_fm(X), y).predict(_fm(np.array([[0.0, 0.0], [5.0, 5.0]])))
    assert list(preds) == ["A", "B"]


def test_argamon_linear_is_sklearn_compatible() -> None:
    assert is_classifier(ArgamonLinearDelta())


def test_quadratic_is_sklearn_compatible() -> None:
    assert is_classifier(QuadraticDelta())
