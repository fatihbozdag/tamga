# Evaluation (PAN suite)

Forensic publications and courts expect more than raw accuracy. tamga ships the standard
PAN verification-task metric menu behind one call.

## One-call evaluation

```python
from tamga.forensic import compute_pan_report

report = compute_pan_report(
    probs=calibrated_probs,     # from CalibratedScorer
    y=ground_truth_labels,
    log_lrs=log_lr_values,      # optional; enables cllr_bits
)
report.to_dict()
# {
#   "auc": 0.94, "c_at_1": 0.88, "f05u": 0.87,
#   "brier": 0.11, "ece": 0.042, "cllr_bits": 0.31,
#   "n_target": 80, "n_nontarget": 120,
# }
```

## The metrics

| Metric | Measures | Reference |
|---|---|---|
| `auc` | Ranking ability. 1.0 perfect, 0.5 random. | — |
| `c_at_1` | Accuracy with credit for principled abstention | Peñas & Rodrigo 2011 |
| `f05u` | Precision-weighted F-measure with non-answer penalty | Bevendorff et al. PAN 2022 |
| `brier` | Mean-squared posterior error. 0 perfect. | Brier 1950 |
| `ece` | Expected Calibration Error (equal-width bins) | — |
| `cllr` | Log-likelihood-ratio cost — the forensic proper scoring rule | Brümmer & du Preez 2006 |
| `tippett` | Cumulative target / non-target LR distributions | — |

### c@1

$$
\text{c@1} = \frac{1}{n}\!\left( n_\text{correct} + n_\text{unanswered} \cdot \frac{n_\text{correct}}{n} \right)
$$

where *unanswered* trials are those with calibrated probability inside
`[0.5 − margin, 0.5 + margin]`. Margin = 0 (default) reduces c@1 to raw accuracy.

The PAN verification shared task has used c@1 as its primary metric since 2013 because it
rewards systems that know when to abstain — directly aligned with the forensic notion
of "insufficient evidence".

### C_llr

$$
C_\text{llr} = \frac{1}{2}\!\left[
  \frac{1}{|T|}\!\sum_{i \in T} \log_2\!\left(1 + \tfrac{1}{\text{LR}_i}\right)
  +
  \frac{1}{|N|}\!\sum_{i \in N} \log_2\!\left(1 + \text{LR}_i\right)
\right]
$$

where $T$ = target trials, $N$ = non-target. Interpreted as average information loss (in
bits) per trial relative to an optimally-calibrated reference system.

- **Prior-only system** (every log-LR = 0) → C_llr = **1.0** exactly.
- **Perfect, confident system** → C_llr ≈ 0.
- **Misleading system** (wrong sign) → C_llr > 1.

A system with C_llr < 1 beats prior-only. Forensic publications routinely report C_llr
alongside AUC because C_llr captures both discrimination *and* calibration in one scalar.

## Tippett plots

`tippett(log_lrs, y)` returns per-class cumulative distributions you can plot directly:

```python
import matplotlib.pyplot as plt
from tamga.forensic import tippett

data = tippett(log_lrs, y)
plt.step(data["thresholds"], data["target_cdf"], label="same-author")
plt.step(data["thresholds"], data["nontarget_cdf"], label="different-author")
plt.xlabel("log₁₀(LR) threshold")
plt.ylabel("P(log-LR ≥ threshold | class)")
plt.legend()
```

A well-discriminating system shows the target CDF accumulating on the right (high
log-LRs are predominantly target) and the non-target CDF on the left.

## Reference

::: tamga.forensic.metrics.compute_pan_report

::: tamga.forensic.metrics.PANReport

::: tamga.forensic.metrics.auc

::: tamga.forensic.metrics.c_at_1

::: tamga.forensic.metrics.f05u

::: tamga.forensic.metrics.cllr

::: tamga.forensic.metrics.ece

::: tamga.forensic.metrics.brier

::: tamga.forensic.metrics.tippett
