# Changelog

All notable changes to **bitig** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation
- Accuracy audit of the docs against the code: corrected CLI flags and subcommands
  (`cache size`/`list`, the `case` command group, per-command options), the `study.yaml`
  schema reference (`viz`/`report`/`preprocess`/`normalize` blocks; removed the non-existent
  `corpus.strict`), feature/method default values and parameter names, and the forensic
  verbal-scale bands (very strong = log-LR 4–6, extremely strong > 6).
- Added this changelog and a Troubleshooting page (spaCy models, the Turkish
  `torch`/`stanza` version pins, optional-extra import errors).

## [0.3.1] — 2026-06-08

### Fixed
- Made the `bitig[turkish]` extra resolvable (Stanza/spacy-stanza/torch version pinning).

## [0.3.0] — 2026-06-07 — audit remediation

A wide forensic-methodology and robustness pass driven by an external audit.

### Added
- Two-sided ENFSI/Nordgaard verbal scale for likelihood-ratio reporting, centralized in
  `bitig.forensic.verbal_scale` as the single source of truth.

### Fixed
- Chain-of-custody seal closed end-to-end (case signing/verification hardening).
- Corrected four stylometry methods and several forensic methodology details (General
  Impostors fidelity, Delta resubstitution labelling, c@1).
- Path-traversal hardening and atomic writes for the Forensic Lab Case store.
- Packaging hygiene: split out the `cluster` extra (UMAP/HDBSCAN); metadata + typing sweep.
- CI now exercises the court-facing surface (GUI / report / PDF) against its real dependencies.

## [0.2.0] — 2026-05-17 — Forensic Lab Case workflow

### Added
- Forensic Lab Case workflow (`bitig case new/list/open/status/fork/sign/verify`).
- Plotly interactive renderers (rolling-delta, impostors, reliability diagram) and a
  static/interactive toggle on the GUI Results page.
- GitHub Actions Trusted Publishing workflow for PyPI releases.

## [0.1.1] — 2026-05-04 — visual rebrand + PyPI refresh

### Added
- Plotly interactive renderers, PCA biplot, Bayesian posterior heatmap, classifier
  calibration plots (ECE / Brier / reliability diagram).
- General Impostors verification and rolling-delta sliding-window attribution surfaced in
  the GUI; Bootstrap Consensus Tree plot.

### Changed
- Visual rebrand and PyPI project-page refresh.

## [0.1.0] — 2026-04-23 — first public release

The initial release, covering the analytical breadth of R's `Stylo` plus a modern NLP,
Bayesian, and forensic layer. Highlights from the phased build-up:

### Added
- **Corpus & features** — `.txt` + TSV ingestion with content-addressed hashing; MFW, char/
  word/POS n-grams, dependency bigrams, function words, punctuation, readability, sentence
  length, lexical diversity, and (optional) sentence/contextual embeddings.
- **Methods** — Burrows/Eder/Argamon/Cosine/Quadratic Delta; classic + Eder Zeta;
  PCA/MDS/t-SNE/UMAP; Ward/k-means/HDBSCAN; bootstrap consensus trees; sklearn classify with
  stylometry-aware CV; Bayesian Wallace-Mosteller + hierarchical group comparison.
- **Forensic toolkit** — General Impostors, Unmasking, Sapkota topic-invariant char-n-gram
  categories, Stamatatos distortion, calibration (Platt/isotonic), LR + PAN evaluation suite
  (AUC, c@1, F0.5u, Brier, ECE, C_llr, Tippett), chain-of-custody provenance, and an
  LR-framed HTML report template.
- **Multi-language support** — first-class EN/TR/DE/ES/FR behind the `bitig.languages`
  registry, with native readability formulas per language and Turkish via Stanford Stanza
  (BOUN) through `spacy-stanza`.
- **Orchestration & output** — declarative `study.yaml` runner (`bitig run`), Rich-based
  `bitig shell`, NiceGUI desktop GUI, publication-grade matplotlib figures, and uniform
  `Result` records (JSON + Parquet + figures).
- **Docs** — MkDocs Material site with Concepts, Forensic toolkit, and Federalist / PAN-CLEF /
  Turkish tutorials, plus EN + TR localization.

[Unreleased]: https://github.com/fatihbozdag/bitig/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/fatihbozdag/bitig/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/fatihbozdag/bitig/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/fatihbozdag/bitig/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/fatihbozdag/bitig/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/fatihbozdag/bitig/releases/tag/v0.1.0
