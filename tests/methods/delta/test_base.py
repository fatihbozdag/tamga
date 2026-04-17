"""Tests for the _DeltaBase nearest-author-centroid logic (using a trivial L2 subclass)."""

import numpy as np

from tamga.features import FeatureMatrix
from tamga.methods.delta.base import _DeltaBase


class _L2Delta(_DeltaBase):
    """Minimal concrete subclass for testing the base class machinery."""

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.linalg.norm(X - centroid, axis=1)


def _fm(X: np.ndarray) -> FeatureMatrix:  # noqa: N803
    return FeatureMatrix(
        X=X,
        document_ids=[f"d{i}" for i in range(X.shape[0])],
        feature_names=[f"f{j}" for j in range(X.shape[1])],
        feature_type="test",
    )


def test_fit_stores_centroids_per_author() -> None:
    X = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
    y = np.array(["A", "A", "B", "B"])
    clf = _L2Delta().fit(_fm(X), y)
    assert set(clf.classes_) == {"A", "B"}
    np.testing.assert_allclose(clf.centroids_["A"], [0.05, 0.05])
    np.testing.assert_allclose(clf.centroids_["B"], [10.05, 10.05])


def test_predict_returns_nearest_author() -> None:
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    y = np.array(["A", "B"])
    clf = _L2Delta().fit(_fm(X), y)
    # New points very close to each centroid.
    probe = _fm(np.array([[0.01, 0.0], [9.99, 10.0]]))
    preds = clf.predict(probe)
    assert list(preds) == ["A", "B"]


def test_decision_function_returns_negative_distances() -> None:
    X = np.array([[0.0], [10.0]])
    y = np.array(["A", "B"])
    clf = _L2Delta().fit(_fm(X), y)
    probe = _fm(np.array([[0.0]]))
    scores = clf.decision_function(probe)
    # Closer to A => score for A higher than score for B (scores are negative distances).
    assert scores.shape == (1, 2)
    class_a = list(clf.classes_).index("A")
    class_b = list(clf.classes_).index("B")
    assert scores[0, class_a] > scores[0, class_b]


def test_predict_proba_rows_sum_to_one() -> None:
    X = np.array([[0.0], [10.0]])
    y = np.array(["A", "B"])
    clf = _L2Delta().fit(_fm(X), y)
    probe = _fm(np.array([[0.0], [5.0], [10.0]]))
    probs = clf.predict_proba(probe)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_predict_accepts_numpy_array_for_sklearn_compat() -> None:
    """sklearn's cross_validate passes X as ndarray, not FeatureMatrix."""
    X = np.array([[0.0], [10.0]])
    y = np.array(["A", "B"])
    clf = _L2Delta().fit(X, y)
    preds = clf.predict(np.array([[0.0], [10.0]]))
    assert list(preds) == ["A", "B"]
