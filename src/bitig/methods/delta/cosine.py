"""Cosine Delta (Smith & Aldridge 2011; Evert et al. 2017) -- 1 - cosine similarity on z-scored features."""

from __future__ import annotations

import numpy as np

from bitig.methods.delta.base import _DeltaBase


class CosineDelta(_DeltaBase):
    """1 - cosine(x, c). Undefined when either vector is all-zero; add a tiny epsilon to avoid div-by-zero."""

    _EPS = 1e-12

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:  # noqa: N803
        x_norms = np.linalg.norm(X, axis=1)
        c_norm = np.linalg.norm(centroid)
        denom = np.maximum(x_norms * c_norm, self._EPS)
        cosine = (X @ centroid) / denom
        return 1.0 - cosine  # type: ignore[no-any-return]
