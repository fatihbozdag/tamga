"""Argamon Linear Delta (Argamon 2008) -- L2 distance on z-scored features -- and Quadratic Delta -- squared-L2."""

from __future__ import annotations

import numpy as np

from tamga.methods.delta.base import _DeltaBase


class ArgamonLinearDelta(_DeltaBase):
    """L2 distance on z-scored features: sqrt(sum((x - c)^2))."""

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.linalg.norm(X - centroid, axis=1)  # type: ignore[no-any-return]


class QuadraticDelta(_DeltaBase):
    """Squared-L2 distance: sum((x - c)^2). Preserves ranking vs Argamon, differs in scale."""

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:  # noqa: N803
        diff = X - centroid
        return (diff * diff).sum(axis=1)  # type: ignore[no-any-return]
