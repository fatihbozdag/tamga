# Reporting

Forensic reports need more than a score — they need the **hypothesis pair** under test,
the **known and questioned** material identified, a **chain-of-custody** trail back to
source files, and the **evidential disclaimer** that the metrics are conditional on the
analysis conditions.

## The forensic report

```python
from tamga.report import build_forensic_report

build_forensic_report(
    "results/case_001",
    output="results/case_001/forensic_report.html",
    title="R v Smith — authorship analysis",
    lr_summaries={
        "general_impostors": {"log_lr": "1.34", "lr": "21.9"},
        "unmasking":          {"log_lr": "1.10", "lr": "12.6"},
    },
)
```

Template sections:

1. **Hypotheses under test** — rendered iff `hypothesis_pair`, `questioned_description`,
   or `known_description` is populated on the `Provenance`.
2. **Chain of custody** — rendered iff `acquisition_notes`, `custody_notes`, or
   `source_hashes` is populated.
3. **Per-method LR block** — rendered iff `lr_summaries` dict is passed. Shows
   log₁₀(LR) + LR + the six-band ENFSI / Nordgaard verbal scale.
4. **Figures + params** per method (from the saved `Result` directory).
5. **Evidentiary disclaimer** — always rendered.
6. **Reproducibility provenance** — always rendered (full JSON of the `Provenance` record).

## Populating chain-of-custody

Pass the forensic fields when building your Provenance:

```python
from tamga.provenance import Provenance

provenance = Provenance.current(
    spacy_model="en_core_web_trf",
    spacy_version=spacy.__version__,
    corpus_hash=corpus.hash(),
    feature_hash=fm.provenance_hash,
    seed=42,
    resolved_config=cfg.model_dump(),
    questioned_description="Email thread seized 2026-03-15 under warrant W-2026-0815",
    known_description="15 personal emails from the suspect's Gmail, 2024-2026",
    hypothesis_pair="H1: written by the suspect; H0: written by someone other than the suspect",
    acquisition_notes="Full drive image; chain of custody intact from seizure to analysis",
    custody_notes="No modifications after acquisition. SHA-256s below match original files.",
    source_hashes={
        "questioned_1": "a1b2c3...",
        "known_1":       "d4e5f6...",
        "known_2":       "...",
    },
)
```

## HTML safety

The forensic report template is rendered with Jinja2 autoescape enabled — every
user-supplied string (custody notes, hypothesis text, source hashes) is HTML-entity
escaped. A `<script>` tag in `custody_notes` is rendered as `&lt;script&gt;`, not
executed when the HTML is opened in a browser.

## Evidentiary disclaimer

The template includes a standard disclaimer adapted from the ENFSI (2015) evaluative
reporting guideline:

> Output is intended to inform, not replace, expert forensic-linguistic judgement.
> Likelihood ratios reported here are conditional on the specific known and questioned
> material, the feature space chosen, and the calibration set used. Extrapolation to
> populations outside the calibration conditions is not warranted.

You may override the disclaimer by providing a custom template; see the bundled
`src/tamga/report/templates/forensic_lr.html.j2` for the reference implementation.

## Reference

::: tamga.report.render.build_forensic_report

::: tamga.provenance.Provenance
    options:
      members:
        - current
        - from_dict
        - to_dict
        - has_forensic_metadata
