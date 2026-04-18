---
hide:
  - navigation
---

# tamga

<p align="center">
  <img src="assets/tamga-banner.svg" alt="tamga — computational stylometry" style="max-width: 100%;">
</p>

**Computational stylometry for authorship attribution, author-group comparison, and forensic-linguistic analysis.** A Python replacement for R's `Stylo`, with a modern NLP pipeline (spaCy, transformer embeddings), a Bayesian layer (PyMC), and a full forensic-evidential toolkit on top.

> Named after the **tamga** — the Turkic clan-mark by which individual and familial
> identity was recognised at a glance — the material-culture counterpart to a stylistic
> fingerprint.

## Architecture

<p align="center">
  <img src="assets/tamga-architecture.svg" alt="corpus → features → methods → forensic → output" style="max-width: 100%;">
</p>

Every layer is `sklearn`-compatible; every `Result` carries full provenance (corpus hash,
feature hash, seed, spaCy version, timestamp, resolved config) so a study written as a
`study.yaml` is reproducible to the exact random draw years later.

## Quick navigation

<div class="grid cards" markdown>

-   :fontawesome-solid-rocket:{ .lg .middle } **Getting started**

    ---

    Install tamga, build your first corpus, and run a Burrows Delta study from the CLI.

    [:octicons-arrow-right-24: Install & quickstart](getting-started.md)

-   :material-book-open-page-variant:{ .lg .middle } **Concepts**

    ---

    Corpus → Features → Methods → Results. The four layers of the pipeline, explained.

    [:octicons-arrow-right-24: Concepts](concepts/index.md)

-   :material-shield-search:{ .lg .middle } **Forensic toolkit**

    ---

    General Impostors verification, Unmasking, LR output + calibration, PAN evaluation.

    [:octicons-arrow-right-24: Forensic toolkit](forensic/index.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Reproduce Mosteller & Wallace on the Federalist Papers; run PAN-style forensic verification end-to-end.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

</div>

## What's in the box

| Layer | Highlights |
|---|---|
| **Corpus** | `.txt` + TSV metadata ingestion, filter / groupby, content-addressed hashing |
| **Features** | MFW, char / word / POS n-grams, dependency bigrams, function words, punctuation, readability, sentence length, lexical diversity, sentence + contextual embeddings |
| **Methods** | Burrows / Eder / Argamon / Cosine / Quadratic Delta; Zeta; PCA / UMAP / t-SNE / MDS; Ward / k-means / HDBSCAN; bootstrap consensus; sklearn classify + CV; Wallace–Mosteller Bayesian |
| **Forensic** | General Impostors, Unmasking, Stamatatos distortion, Sapkota n-gram categories, Platt / isotonic calibration, log-LR + C_llr + AUC + c@1 + F0.5u + ECE + Brier + Tippett, PANReport, chain-of-custody Provenance, LR-framed HTML report |

## Status

**Phase 5 landed** — visualisation, Jinja2 reports, declarative runner (`tamga run`), and a
Rich-based interactive `tamga shell`.

**Forensic phase landed** — six additions (General Impostors, LR + calibration + evaluation
metrics, Sapkota categories + Stamatatos distortion, Unmasking, chain-of-custody + forensic
report template, PAN harness). **123 new tests, 358 total passing.**

**Phase 6 (in progress)** — this MkDocs documentation site; a PAN-CLEF verification
tutorial; PyPI publish.

## License & citation

BSD-3-Clause. See [`LICENSE`](https://github.com/fatihbozdag/tamga/blob/main/LICENSE).

If you use tamga in published work, please cite it via
[`CITATION.cff`](https://github.com/fatihbozdag/tamga/blob/main/CITATION.cff).
