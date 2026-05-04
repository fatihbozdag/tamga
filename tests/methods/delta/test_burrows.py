"""Tests for BurrowsDelta."""

import numpy as np

from bitig.features import FeatureMatrix
from bitig.methods.delta.burrows import BurrowsDelta


def _fm(X: np.ndarray) -> FeatureMatrix:  # noqa: N803
    return FeatureMatrix(
        X=X,
        document_ids=[f"d{i}" for i in range(X.shape[0])],
        feature_names=[f"f{j}" for j in range(X.shape[1])],
        feature_type="zscored-mfw",
    )


def test_burrows_attributes_to_nearest_centroid() -> None:
    # Two authors, well-separated.
    X = np.array([[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [5.1, 5.2]])
    y = np.array(["A", "A", "B", "B"])
    probe = _fm(np.array([[0.05, 0.1], [5.05, 5.1]]))
    preds = BurrowsDelta().fit(_fm(X), y).predict(probe)
    assert list(preds) == ["A", "B"]


def test_burrows_distance_is_mean_absolute_difference() -> None:
    X = np.array([[0.0, 0.0]])
    y = np.array(["A"])
    clf = BurrowsDelta().fit(_fm(X), y)
    # Distance from [0, 0] to centroid [0, 0] is 0; from [1, 1] to [0, 0] is 1.0 (mean of |1|, |1|).
    scores = clf.decision_function(_fm(np.array([[0.0, 0.0], [1.0, 1.0]])))
    assert scores[0, 0] == 0.0
    assert scores[1, 0] == -1.0  # score = -distance


def test_burrows_is_sklearn_compatible() -> None:
    from sklearn.base import is_classifier

    assert is_classifier(BurrowsDelta())
