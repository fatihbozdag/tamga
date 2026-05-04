"""Burrows Classic Delta (Burrows 2002) -- mean absolute difference of z-scored features."""

from __future__ import annotations

import numpy as np

from bitig.methods.delta.base import _DeltaBase


class BurrowsDelta(_DeltaBase):
    """Burrows Classic Delta.

    Distance = mean(|x_i - c_i|) across features -- the L1 norm divided by the feature count.
    Assumes both `X` and `centroid` are z-scored in the same coordinate system.
    """

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.abs(X - centroid).mean(axis=1)  # type: ignore[no-any-return]
