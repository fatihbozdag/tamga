# Methods

Methods turn a `FeatureMatrix` into a `Result`. Every method is `sklearn`-compatible
where it makes sense (`fit`, `predict`, `fit_transform`).

## Attribution — Delta variants

All Delta variants share `_DeltaBase`, work on z-scored features, and produce a
nearest-centroid classifier.

| Class | Distance | Reference |
|---|---|---|
| `BurrowsDelta` | mean absolute L1 | Burrows 2002 |
| `ArgamonLinearDelta` | L2 (Euclidean) | Argamon 2008 |
| `QuadraticDelta` | squared L2 | — |
| `CosineDelta` | 1 − cosine similarity | Smith & Aldridge 2011 |
| `EderDelta` / `EderSimpleDelta` | weighted Delta variants | Eder 2015 |

```python
from tamga import MFWExtractor, BurrowsDelta
fm = MFWExtractor(n=200, scale="zscore", lowercase=True).fit_transform(corpus)
y = np.array(corpus.metadata_column("author"))
clf = BurrowsDelta().fit(fm, y)
predictions = clf.predict(fm)       # nearest-centroid labels
probs = clf.predict_proba(fm)       # softmax over negative distances
```

## Contrast — Zeta

Craig's Zeta (`ZetaClassic`) and Eder's smoothed variant (`ZetaEder`) extract the
vocabulary most preferred by one author group versus another.

```python
from tamga.methods.zeta import ZetaClassic
result = ZetaClassic(group_by="author", top_k=50, group_a="Hamilton", group_b="Madison").fit_transform(corpus)
df_a, df_b = result.tables   # top-k A-preferred / B-preferred words with proportions
```

## Dimensionality reduction

`PCAReducer`, `MDSReducer`, `TSNEReducer`, `UMAPReducer` — all accept a `FeatureMatrix`
and return a `Result` with 2-D / n-D coordinates plus (for PCA) explained-variance
ratios.

## Clustering

`HierarchicalCluster(linkage=...)` (Ward, average, complete, single), `KMeansCluster`,
`HDBSCANCluster`. Produce a `Result` with cluster labels + (for hierarchical) the linkage
matrix for dendrograms.

## Consensus trees

`BootstrapConsensus(mfw_bands=[100, 200, 300], replicates=20)` — Eder 2017 bootstrap
across MFW bands, producing a Newick consensus tree with clade support values.

## Classification + CV

Any sklearn classifier (Logistic Regression, linear / RBF SVM, Random Forest, HistGBM)
via `build_classifier(name)`, plus `cross_validate_tamga(fm, y, cv_kind=...)` with three
stylometry-aware CV strategies:

- `stratified` — StratifiedKFold, `seed` controls the shuffle
- `loao` — Leave-One-Author-Out (LeaveOneGroupOut with author as group)
- `leave_one_text_out` — LeaveOneOut

## Bayesian

- `BayesianAuthorshipAttributor` — Wallace–Mosteller Dirichlet-smoothed log-rate
  classifier. Uses raw counts (`MFWExtractor(scale="none")`).
- `HierarchicalGroupComparison` — PyMC varying-intercept model for per-author draws from
  a group hyperparameter. Useful for testing whether two author populations differ
  systematically in a stylistic feature. Requires `tamga[bayesian]`.

## Forensic methods

Under `tamga.forensic`:

| Method | Task | Reference |
|---|---|---|
| `GeneralImpostors` | one-class verification | Koppel & Winter 2014 |
| `Unmasking` | long-text verification (accuracy-degradation curve) | Koppel & Schler 2004 |
| `CalibratedScorer` | Platt / isotonic calibration of any scorer | Platt 1999; Niculescu-Mizil & Caruana 2005 |

See [Forensic toolkit](../forensic/index.md).

## Next

- [Results & provenance](results.md) — what every method returns.
