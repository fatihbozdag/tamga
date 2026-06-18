# Python API

Auto-generated from the source via mkdocstrings. This page documents the most commonly used
surface; most symbols are re-exported at the `bitig` top level (e.g.
`from bitig import BurrowsDelta`), but a few — noted inline below — must be imported from their
submodule. The full export list lives in
[`src/bitig/__init__.py`](https://github.com/fatihbozdag/bitig/blob/main/src/bitig/__init__.py),
which re-exports every Delta/Zeta variant, every reducer and clusterer, every feature
extractor, the forensic toolkit, and the config/runner helpers.

## Corpus

::: bitig.corpus.Corpus
    options:
      show_root_full_path: false

::: bitig.corpus.Document
    options:
      show_root_full_path: false

## Features

::: bitig.features.base.FeatureMatrix
    options:
      show_root_full_path: false

::: bitig.features.mfw.MFWExtractor
    options:
      show_root_full_path: false

## Methods

### Delta

::: bitig.methods.delta.burrows.BurrowsDelta
    options:
      show_root_full_path: false

### Zeta

::: bitig.methods.zeta.ZetaClassic
    options:
      show_root_full_path: false

### Clustering

::: bitig.methods.cluster.HierarchicalCluster
    options:
      show_root_full_path: false

### Classification

::: bitig.methods.classify.build_classifier

::: bitig.methods.classify.cross_validate_bitig

## Results

::: bitig.result.Result
    options:
      show_root_full_path: false

::: bitig.provenance.Provenance
    options:
      show_root_full_path: false

## Forensic

The verification, distortion, and topic-invariant feature classes are re-exported at the
`bitig` top level; the PAN evaluation helpers (`compute_pan_report`, `PANReport`, and the
individual metrics) live under `bitig.forensic`.

::: bitig.forensic.verify.GeneralImpostors
    options:
      show_root_full_path: false

::: bitig.forensic.unmasking.Unmasking
    options:
      show_root_full_path: false

::: bitig.forensic.lr.CalibratedScorer
    options:
      show_root_full_path: false

::: bitig.forensic.char_ngrams.CategorizedCharNgramExtractor
    options:
      show_root_full_path: false

::: bitig.forensic.distortion.distort_corpus

::: bitig.forensic.metrics.compute_pan_report

::: bitig.forensic.verbal_scale.lr_verbal_statement

## Config

::: bitig.config.StudyConfig
    options:
      show_root_full_path: false

::: bitig.config.load_config

## Runner

::: bitig.runner.run_study

## Reporting

::: bitig.report.render.build_report

`build_forensic_report` is **not** re-exported at the top level — import it from
`bitig.report`.

::: bitig.report.render.build_forensic_report
