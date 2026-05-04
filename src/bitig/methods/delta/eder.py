"""Eder Delta (Eder 2015) -- rank-weighted Burrows -- and Eder's Simple Delta (Eder 2017)."""

from __future__ import annotations

import numpy as np

from bitig.features import FeatureMatrix
from bitig.methods.delta.base import _DeltaBase


class EderDelta(_DeltaBase):
    """Eder Delta: like Burrows, but each feature's contribution is weighted by `(n - rank) / n`,
    so the most frequent features contribute most. Features are ranked by their training-set mean
    absolute z-score (more discriminating features get higher rank).

    Scalar weighting is computed at `fit` time from the centroids.
    """

    def __init__(self) -> None:
        super().__init__()
        self._weights: np.ndarray | None = None

    def fit(self, X: FeatureMatrix | np.ndarray, y: np.ndarray) -> EderDelta:  # type: ignore[override]  # noqa: N803
        super().fit(X, y)
        # Feature importance proxy: across-centroid variance (discriminating features have high variance).
        stacked = np.vstack(list(self.centroids_.values()))
        importance = stacked.var(axis=0)
        ranks = importance.argsort()[::-1]  # descending by importance
        n = len(importance)
        weights = np.zeros(n)
        for rank_pos, feat_idx in enumerate(ranks):
            weights[feat_idx] = (n - rank_pos) / n
        self._weights = weights
        return self

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:  # noqa: N803
        assert self._weights is not None
        return (self._weights * np.abs(X - centroid)).sum(axis=1) / self._weights.sum()  # type: ignore[no-any-return]


class EderSimpleDelta(_DeltaBase):
    """Eder's Simple Delta (Eder 2017): L1 distance on unweighted z-scored features.

    Differs from Burrows only in that Burrows divides by feature count; Eder Simple does not.
    In practice this only changes a monotone scaling of distances -- rankings are identical.
    """

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.abs(X - centroid).sum(axis=1)  # type: ignore[no-any-return]
