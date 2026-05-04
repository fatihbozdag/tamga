# Python API

Kaynaktan mkdocstrings aracılığıyla otomatik oluşturulmuştur. Aşağıda listelenen her sembol,
`bitig` üst düzeyinde yeniden dışa aktarılır (aksi belirtilmedikçe).

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

## Runner

::: bitig.runner.run_study

## Reporting

::: bitig.report.render.build_report

::: bitig.report.render.build_forensic_report
