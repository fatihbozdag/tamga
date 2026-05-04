"""Unit tests for `bitig.metrics.calibration`."""

from __future__ import annotations

import numpy as np
import pytest

from bitig.metrics.calibration import (
    brier_score,
    calibration_curve,
    expected_calibration_error,
)


def test_perfectly_calibrated_classifier_has_zero_ece() -> None:
    """For a known target distribution where confidence == accuracy, ECE -> 0."""
    rng = np.random.default_rng(0)
    n = 4000
    confidence = rng.uniform(0.5, 1.0, size=n)
    # Generate y_true so that accuracy in each bin == confidence.
    y_true = np.array(["A" if rng.uniform() < c else "B" for c in confidence])
    y_pred = np.array(["A"] * n)
    # Build proba in 2-class space [P(A), P(B)] with P(A)=confidence and pred=A.
    y_proba = np.column_stack([confidence, 1 - confidence])
    classes = np.array(["A", "B"])
    ece = expected_calibration_error(y_true, y_proba, classes=classes, n_bins=10)
    assert ece < 0.05, f"ECE={ece} should be tiny for a calibrated classifier"
    # And argmax matches what we constructed.
    assert (y_proba.argmax(axis=1) == 0).all()
    _ = y_pred  # quiet unused-var warning if linter complains


def test_overconfident_classifier_has_high_ece() -> None:
    """A 'always predict A with prob 0.99' classifier on 50/50 data should be far from calibrated."""
    n = 1000
    rng = np.random.default_rng(0)
    y_true = np.where(rng.uniform(size=n) < 0.5, "A", "B")
    y_proba = np.tile([0.99, 0.01], (n, 1))
    classes = np.array(["A", "B"])
    ece = expected_calibration_error(y_true, y_proba, classes=classes, n_bins=10)
    # Confidence ~= 0.99, accuracy ~= 0.5 -> gap ~= 0.49.
    assert ece > 0.4


def test_calibration_curve_shape() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.choice(["A", "B", "C"], size=300)
    y_proba = rng.dirichlet([2, 2, 2], size=300)
    classes = np.array(["A", "B", "C"])
    curve = calibration_curve(y_true, y_proba, classes=classes, n_bins=8)
    assert len(curve.bin_centers) == 8
    assert len(curve.mean_confidence) == 8
    assert len(curve.accuracy) == 8
    assert int(curve.counts.sum()) == 300


def test_brier_score_perfect_classifier_is_zero() -> None:
    classes = np.array(["A", "B"])
    y_true = np.array(["A", "B", "A", "B"])
    y_proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert brier_score(y_true, y_proba, classes=classes) == pytest.approx(0.0)


def test_brier_score_uniform_2_class() -> None:
    classes = np.array(["A", "B"])
    y_true = np.array(["A", "B"])
    y_proba = np.array([[0.5, 0.5], [0.5, 0.5]])
    # Each row contributes (0.5)^2 + (0.5)^2 = 0.5. Mean is 0.5.
    assert brier_score(y_true, y_proba, classes=classes) == pytest.approx(0.5)


def test_calibration_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="y_proba must be 2-D"):
        calibration_curve(np.array([0]), np.array([0.5]), n_bins=4)
    with pytest.raises(ValueError, match="row counts disagree"):
        calibration_curve(
            np.array(["A", "B"]),
            np.array([[0.5, 0.5], [0.4, 0.6], [0.3, 0.7]]),
            classes=np.array(["A", "B"]),
            n_bins=4,
        )
    with pytest.raises(ValueError, match="n_bins must be"):
        calibration_curve(
            np.array(["A"]), np.array([[1.0, 0.0]]), classes=np.array(["A", "B"]), n_bins=1
        )


def test_brier_unknown_label_raises() -> None:
    classes = np.array(["A", "B"])
    y_true = np.array(["C"])
    y_proba = np.array([[0.4, 0.6]])
    with pytest.raises(ValueError, match="not in classes"):
        brier_score(y_true, y_proba, classes=classes)
