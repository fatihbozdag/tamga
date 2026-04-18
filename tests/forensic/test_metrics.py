"""Tests for forensic evaluation metrics (cllr, ece, brier, tippett)."""

from __future__ import annotations

import numpy as np
import pytest

from tamga.forensic.metrics import brier, cllr, ece, tippett


class TestCllr:
    """C_llr (Brümmer & du Preez 2006) — the standard forensic proper scoring rule."""

    def test_prior_only_system_gives_cllr_one(self) -> None:
        """A system that always outputs log-LR=0 (no evidence either way) must score
        C_llr = 1.0 exactly. Verifies the normalisation of the formula."""
        log_lrs = np.zeros(100)
        y = np.array([1] * 50 + [0] * 50)
        assert cllr(log_lrs, y) == pytest.approx(1.0, abs=1e-12)

    def test_perfect_separation_gives_low_cllr(self) -> None:
        """High positive log-LRs for targets, high negative for non-targets → near-zero
        cost."""
        log_lrs = np.concatenate([np.full(50, 6.0), np.full(50, -6.0)])
        y = np.array([1] * 50 + [0] * 50)
        assert cllr(log_lrs, y) < 1e-3

    def test_misleading_system_gives_cllr_greater_than_one(self) -> None:
        """High negative log-LR for targets and high positive for non-targets is worse than
        prior-only and must score C_llr > 1."""
        log_lrs = np.concatenate([np.full(50, -3.0), np.full(50, 3.0)])
        y = np.array([1] * 50 + [0] * 50)
        assert cllr(log_lrs, y) > 3.0

    def test_cllr_requires_both_classes(self) -> None:
        with pytest.raises(ValueError, match="target and one non-target"):
            cllr(np.array([1.0, 2.0]), np.array([1, 1]))

    def test_cllr_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            cllr(np.array([1.0, 2.0, 3.0]), np.array([1, 0]))

    def test_cllr_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            cllr(np.array([]), np.array([]))


class TestEce:
    def test_perfectly_calibrated_gives_zero_ece(self) -> None:
        """A well-calibrated forecast: probs equal to empirical frequencies per bin."""
        probs = np.array([0.1] * 100 + [0.9] * 100)
        # In bin around 0.1 we see 10 positives out of 100 (empirical freq = 0.1, matches).
        # In bin around 0.9 we see 90 positives out of 100 (empirical freq = 0.9, matches).
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.binomial(1, 0.1, size=100), rng.binomial(1, 0.9, size=100)])
        assert ece(probs, y, n_bins=10) < 0.08  # some Monte Carlo slack

    def test_systematically_biased_gives_large_ece(self) -> None:
        """All probs say 0.9 but empirical frequency is 0.1 — large calibration gap."""
        probs = np.full(100, 0.9)
        y = np.zeros(100, dtype=int)
        y[:10] = 1  # empirical freq 0.1
        assert ece(probs, y, n_bins=10) == pytest.approx(0.8, abs=1e-9)

    def test_ece_rejects_out_of_range_probs(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ece(np.array([0.5, 1.2]), np.array([1, 0]))

    def test_ece_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            ece(np.array([0.5, 0.5, 0.5]), np.array([1, 0]))

    def test_ece_rejects_zero_bins(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            ece(np.array([0.5]), np.array([1]), n_bins=0)


class TestBrier:
    def test_perfect_prediction_gives_zero(self) -> None:
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        y = np.array([1, 0, 1, 0])
        assert brier(probs, y) == 0.0

    def test_all_confident_wrong_gives_one(self) -> None:
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        y = np.array([1, 0, 1, 0])
        assert brier(probs, y) == 1.0

    def test_uninformative_gives_quarter(self) -> None:
        probs = np.full(100, 0.5)
        y = np.array([1] * 50 + [0] * 50)
        assert brier(probs, y) == pytest.approx(0.25, abs=1e-12)


class TestTippett:
    def test_tippett_cdfs_are_monotone_decreasing(self) -> None:
        """P(LR ≥ threshold) is a non-increasing function of threshold."""
        rng = np.random.default_rng(0)
        log_lrs = rng.normal(0, 2, size=200)
        y = rng.integers(0, 2, size=200)
        data = tippett(log_lrs, y)
        # Non-increasing as thresholds increase.
        assert np.all(np.diff(data["target_cdf"]) <= 1e-12)
        assert np.all(np.diff(data["nontarget_cdf"]) <= 1e-12)

    def test_tippett_target_cdf_stays_above_nontarget_for_discriminating_system(self) -> None:
        """For a discriminating system, target CDF dominates non-target CDF in the mid-range
        of thresholds (more target trials above any given threshold than non-target)."""
        log_lrs = np.concatenate([np.full(100, 2.0), np.full(100, -2.0)])
        y = np.array([1] * 100 + [0] * 100)
        data = tippett(log_lrs, y)
        # At threshold = 0, all targets are above it, no non-targets are.
        idx_zero = int(np.searchsorted(data["thresholds"], 0.0))
        assert data["target_cdf"][idx_zero] == 1.0
        assert data["nontarget_cdf"][idx_zero] == 0.0

    def test_tippett_requires_both_classes(self) -> None:
        with pytest.raises(ValueError, match="target and one non-target"):
            tippett(np.array([1.0, 2.0]), np.array([1, 1]))
