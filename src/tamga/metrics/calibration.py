"""Calibration diagnostics for probabilistic classifiers.

For a multi-class problem, calibration is assessed on the *confidence*
channel: each prediction's confidence is the maximum class probability.
A perfectly calibrated classifier achieves observed accuracy equal to
mean confidence within every confidence bin (cf. Guo et al. 2017 "On
Calibration of Modern Neural Networks").

Three quantities live here:

- `calibration_curve`  -- bin centers, mean confidence, and observed
  accuracy per bin (the data behind the reliability diagram).
- `expected_calibration_error` (ECE) -- weighted mean of
  |confidence - accuracy| across bins, in [0, 1]; lower is better.
- `brier_score` -- multi-class Brier (mean squared error between
  one-hot targets and predicted probabilities), in [0, n_classes];
  lower is better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationCurve:
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    mean_confidence: np.ndarray
    accuracy: np.ndarray
    counts: np.ndarray

    def __len__(self) -> int:
        return int(self.counts.size)


def _validate_inputs(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim != 2:
        raise ValueError(f"y_proba must be 2-D (n_samples, n_classes); got shape {y_proba.shape}")
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            f"y_true ({y_true.shape[0]}) and y_proba ({y_proba.shape[0]}) row counts disagree"
        )
    classes = np.unique(y_true) if classes is None else np.asarray(classes)
    if classes.shape[0] != y_proba.shape[1]:
        raise ValueError(
            f"classes ({classes.shape[0]}) does not match y_proba columns ({y_proba.shape[1]})"
        )
    return y_true, y_proba, classes


def calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    classes: np.ndarray | None = None,
    n_bins: int = 10,
) -> CalibrationCurve:
    """Bin predictions by confidence and report mean confidence vs accuracy per bin."""
    y_true, y_proba, classes = _validate_inputs(y_true, y_proba, classes)
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    confidence = y_proba.max(axis=1)
    pred = classes[y_proba.argmax(axis=1)]
    correct = (pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `np.digitize` with right=True puts boundary values into the lower bin,
    # matching the [a, b) convention except for the last bin which is [a, b].
    bin_idx = np.clip(np.digitize(confidence, bin_edges[1:-1], right=False), 0, n_bins - 1)

    counts = np.zeros(n_bins, dtype=int)
    sum_conf = np.zeros(n_bins)
    sum_acc = np.zeros(n_bins)
    for b, c, ok in zip(bin_idx, confidence, correct, strict=True):
        counts[b] += 1
        sum_conf[b] += float(c)
        sum_acc[b] += float(ok)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, sum_conf / np.maximum(counts, 1), np.nan)
        accuracy = np.where(counts > 0, sum_acc / np.maximum(counts, 1), np.nan)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return CalibrationCurve(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        mean_confidence=mean_conf,
        accuracy=accuracy,
        counts=counts,
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    classes: np.ndarray | None = None,
    n_bins: int = 10,
) -> float:
    """ECE = sum over bins of (count_b / N) * |confidence_b - accuracy_b|."""
    curve = calibration_curve(y_true, y_proba, classes=classes, n_bins=n_bins)
    total = int(curve.counts.sum())
    if total == 0:
        return float("nan")
    weights = curve.counts / total
    gap = np.abs(curve.mean_confidence - curve.accuracy)
    # Empty bins contribute zero (their NaN gap times weight 0).
    return float(np.nansum(weights * np.where(curve.counts > 0, gap, 0.0)))


def brier_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    classes: np.ndarray | None = None,
) -> float:
    """Multi-class Brier score: mean squared error between one-hot targets and probabilities."""
    y_true, y_proba, classes = _validate_inputs(y_true, y_proba, classes)
    onehot = np.zeros_like(y_proba)
    label_index = {label: i for i, label in enumerate(classes.tolist())}
    for row, label in enumerate(y_true.tolist()):
        if label not in label_index:
            raise ValueError(f"y_true contains label {label!r} not in classes {classes!r}")
        onehot[row, label_index[label]] = 1.0
    return float(((y_proba - onehot) ** 2).sum(axis=1).mean())
