"""Forensic-evaluation metrics for authorship verification and attribution.

All functions here evaluate the *calibration and discrimination* of a scorer's output against
binary ground truth (1 = target / same-author trial, 0 = non-target / different-author trial).
These are the metrics forensic journals and courtroom gatekeeping (Daubert, *R v T*) expect
above and beyond raw accuracy.

References
----------
Brummer, N., & du Preez, J. (2006). Application-independent evaluation of speaker detection.
    Computer Speech & Language, 20(2-3), 230-275.
Penas, A., & Rodrigo, A. (2011). A simple measure to assess non-response. Proceedings of
    ACL-HLT 2011, 1415-1424.  [c@1 - not in this module; see PAN harness]
Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly
    Weather Review, 78(1), 1-3.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def cllr(log_lrs: np.ndarray, y: np.ndarray) -> float:
    """Log-likelihood-ratio cost (Brümmer & du Preez 2006).

    A proper scoring rule for binary forensic LR output, capturing both calibration and
    discrimination in a single scalar. Interpreted as the average information loss (in bits)
    per trial relative to an optimally-calibrated reference system.

    C_llr = 0.5 * ( mean_{i: y=1} log2(1 + 1/LR_i)
                  + mean_{i: y=0} log2(1 + LR_i) )

    Zero is unattainable in practice; a prior-only (uninformative) system gives C_llr = 1.
    Values below 1 indicate the system is better than prior-only; values above 1 indicate the
    system's LRs are actively misleading.

    Parameters
    ----------
    log_lrs : np.ndarray of shape (n,)
        log10 likelihood ratios for each trial.
    y : np.ndarray of shape (n,)
        Binary labels: 1 = target (same-author trial), 0 = non-target.

    Returns
    -------
    float
        C_llr cost, in bits.
    """
    log_lrs = np.asarray(log_lrs, dtype=float)
    y = np.asarray(y)
    if log_lrs.shape != y.shape:
        raise ValueError("log_lrs and y must have the same shape")
    if log_lrs.size == 0:
        raise ValueError("log_lrs must not be empty")
    target = y == 1
    nontarget = y == 0
    if not target.any() or not nontarget.any():
        raise ValueError("C_llr requires at least one target and one non-target trial")

    # Convert log10-LR → natural-log-LR for log2(1 + exp(.)) computation.
    # log2(1 + 1/LR) = log2(1 + exp(-ln(LR))); log2(1 + LR) = log2(1 + exp(ln(LR))).
    ln_lr = log_lrs * np.log(10.0)
    # Use logaddexp for numerical stability: log2(1 + exp(x)) = log2(e) * np.logaddexp(0, x).
    log2e = 1.0 / np.log(2.0)
    target_cost = log2e * np.logaddexp(0.0, -ln_lr[target]).mean()
    nontarget_cost = log2e * np.logaddexp(0.0, ln_lr[nontarget]).mean()
    return 0.5 * (float(target_cost) + float(nontarget_cost))


def ece(probs: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width binning.

    ECE = sum_{b=1..B} (|B_b|/n) * |accuracy(B_b) - confidence(B_b)|

    Zero indicates perfect calibration (empirical frequency matches predicted probability in
    every bin). Typical forensic thresholds: ECE < 0.05 is considered well-calibrated.

    Parameters
    ----------
    probs : np.ndarray of shape (n,)
        Predicted probability of the positive class for each trial, in [0, 1].
    y : np.ndarray of shape (n,)
        Binary labels (0 or 1).
    n_bins : int
        Number of equal-width probability bins. Default 10.
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    if probs.shape != y.shape:
        raise ValueError("probs and y must have the same shape")
    if not np.all((probs >= 0) & (probs <= 1)):
        raise ValueError("probs must lie in [0, 1]")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(probs.size)
    err = 0.0
    for lo, hi in pairwise(edges):
        # Include right edge on the final bin so p=1.0 is counted.
        in_bin = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        if not in_bin.any():
            continue
        bin_acc = float(y[in_bin].mean())
        bin_conf = float(probs[in_bin].mean())
        err += (in_bin.sum() / total) * abs(bin_acc - bin_conf)
    return err


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Brier score (mean squared error between predicted probability and binary label).

    Zero = perfect probabilistic prediction; 0.25 = uninformed (all probs = 0.5); 1.0 = worst
    possible (confident-wrong on every trial).
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    if probs.shape != y.shape:
        raise ValueError("probs and y must have the same shape")
    return float(np.mean((probs - y) ** 2))


def tippett(log_lrs: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Tippett-plot data: cumulative proportion of target and non-target trials at or above
    each threshold.

    The classical forensic visualisation: plot both CDFs together on a log-LR x-axis.
    - The target CDF should accumulate at high log-LR (right of zero).
    - The non-target CDF should accumulate at low log-LR (left of zero).
    - Where they cross is an empirical equal-error threshold.

    Returns
    -------
    dict
        ``thresholds``: sorted unique log-LR values.
        ``target_cdf``: P(log-LR ≥ t | target) at each threshold.
        ``nontarget_cdf``: P(log-LR ≥ t | non-target) at each threshold.
    """
    log_lrs = np.asarray(log_lrs, dtype=float)
    y = np.asarray(y)
    if log_lrs.shape != y.shape:
        raise ValueError("log_lrs and y must have the same shape")
    target_mask = y == 1
    nontarget_mask = y == 0
    if not target_mask.any() or not nontarget_mask.any():
        raise ValueError("tippett requires at least one target and one non-target trial")

    thresholds = np.unique(log_lrs)
    target_lrs = log_lrs[target_mask]
    nontarget_lrs = log_lrs[nontarget_mask]
    target_cdf = np.array([(target_lrs >= t).mean() for t in thresholds])
    nontarget_cdf = np.array([(nontarget_lrs >= t).mean() for t in thresholds])
    return {
        "thresholds": thresholds,
        "target_cdf": target_cdf,
        "nontarget_cdf": nontarget_cdf,
    }
