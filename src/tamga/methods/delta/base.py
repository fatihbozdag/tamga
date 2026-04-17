"""_DeltaBase -- shared fit/predict logic for every Delta variant.

Each Delta method is modelled as a nearest-author-centroid classifier. Subclasses differ only in
their `_distance(X, centroid)` implementation -- the distance kernel. Fit stores the mean of each
author's training feature vectors; predict returns the author whose centroid is nearest under
that kernel.

Accepts both `FeatureMatrix` and plain `np.ndarray` inputs so sklearn's `Pipeline`,
`cross_validate`, and `GridSearchCV` work without custom adapters.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tamga.features import FeatureMatrix


def _as_ndarray(X: FeatureMatrix | np.ndarray) -> np.ndarray:  # noqa: N803
    return X.X if isinstance(X, FeatureMatrix) else np.asarray(X)


class _DeltaBase(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self.classes_: np.ndarray = np.empty(0, dtype=object)
        self.centroids_: dict[str, np.ndarray] = {}

    @abstractmethod
    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:  # noqa: N803
        """Return per-row distances from each row of `X` to the given centroid."""

    def fit(self, X: FeatureMatrix | np.ndarray, y: np.ndarray) -> _DeltaBase:  # noqa: N803
        X_arr = _as_ndarray(X)  # noqa: N806
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        self.centroids_ = {label: X_arr[y_arr == label].mean(axis=0) for label in self.classes_}
        return self

    def decision_function(self, X: FeatureMatrix | np.ndarray) -> np.ndarray:  # noqa: N803
        X_arr = _as_ndarray(X)  # noqa: N806
        # Return negative distance so that "higher = more likely" matches sklearn convention.
        return np.column_stack(
            [-self._distance(X_arr, self.centroids_[label]) for label in self.classes_]
        )

    def predict(self, X: FeatureMatrix | np.ndarray) -> np.ndarray:  # noqa: N803
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]  # type: ignore[no-any-return]

    def predict_proba(self, X: FeatureMatrix | np.ndarray) -> np.ndarray:  # noqa: N803
        """Softmax over negative distances -- a monotonic, well-defined probability proxy."""
        scores = self.decision_function(X)
        # Numerically stable softmax.
        scores_shift = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores_shift)
        return exp / exp.sum(axis=1, keepdims=True)  # type: ignore[no-any-return]
