# `bitig` — Python Stylometry Package: Design Specification

**Date:** 2026-04-17
**Status:** Approved (brainstorming), awaiting implementation plan
**Author:** F. Bozdağ (owner) with Claude

---

## 1. Overview

### 1.1 What this package is

`bitig` is a Python package and interactive CLI for computational stylometry.
It is a next-generation replacement for R's `Stylo`, designed from the ground up for:

- **Authorship attribution and verification.**
- **Author-group and style comparison** (e.g., L2 vs. native writers, proficiency bands, genre/register).
- **General-purpose Digital Humanities stylometry** — feature parity with `Stylo` plus modern NLP.

The name derives from the Old Turkic **bitig** — the clan/family mark stamped on livestock, tools, seals,
and rugs to identify ownership and lineage at a glance. Each family's bitig was visually unique and
immediately identifiable; the computational analogue is exactly what stylometric methods recover from prose.

### 1.2 Goals

1. Feature parity with R's `Stylo` for the methods researchers actually use:
   Burrows/Eder/Argamon/cosine/simple Delta, Craig's Zeta, PCA/MDS, hierarchical clustering, dendrograms,
   bootstrap consensus trees.
2. Modern NLP as a native layer:
   spaCy-based linguistic features on day one; sentence-transformer and BERT-style embeddings as an
   optional first-class module; advanced style models (LUAR, Wegmann et al.) as plugins in v0.2.
3. **Progressive disclosure.** One code base exposes:
   a library API, one-shot CLI commands, config-driven studies (`study.yaml`), and full project scaffolds —
   users start at whichever layer fits and graduate upward without rewrites.
4. **Reproducibility-first.** Every result records corpus hash, spaCy model version, seeds, and resolved
   configuration. `lock.yaml` pins the analysis environment.
5. **Publication-quality output.** matplotlib/seaborn static figures at journal DPI (300+) with
   PDF/EPS/TIFF/SVG export; optional Plotly interactive figures and self-contained HTML reports.
6. **sklearn interoperability as an architectural principle.** Every feature extractor is a sklearn
   `TransformerMixin`; every fit-able method is a `ClassifierMixin`/`ClusterMixin`/`TransformerMixin`.
   `Pipeline`, `cross_validate`, `GridSearchCV`, `permutation_importance` all work natively on `bitig` objects.

### 1.3 Non-goals

- Not a general NLP toolkit — parsing/tagging/lemmatization is delegated to spaCy.
- Not a corpus manager or annotation tool — `bitig` consumes corpora, does not curate them.
- Not a GUI app — the "interactive" surface is a terminal shell (Rich + Questionary), not Electron/Qt.
- No `rpy2` wrapper around R's `Stylo` — methods are reimplemented from primary literature.
- No LLM-only "vibes-based" authorship verdicts — if LLM features appear (v0.3+), they are always
  traceable to measurable inputs.

### 1.4 Target users

- Authorship attribution researchers (the classical `Stylo` audience).
- Corpus linguists and SLA researchers comparing learner vs. native writing, proficiency bands, L1
  groups, or genre.
- Digital Humanities scholars running stylistic analyses on literary corpora.
- ML practitioners who want stylometric feature extraction in their own sklearn pipelines.

---

## 2. Architecture

Concentric layers; each outer layer depends only on inner layers. Core layers have no optional
dependencies; outer layers pull in extras as declared in `pyproject.toml`.

```
┌─────────────────────────────────────────────────────────────┐
│  bitig.cli          Typer command tree + interactive shell  │  ← user-facing
│  bitig.report       HTML / Markdown / PDF reports (opt-in)  │
│  bitig.viz          matplotlib / seaborn + plotly backends  │
├─────────────────────────────────────────────────────────────┤
│  bitig.methods      delta, zeta, consensus; v0.2: rolling   │
│  bitig.classify     sklearn wrappers, CV, evaluation        │
│  bitig.reduce       PCA, MDS, t-SNE, UMAP                   │
│  bitig.cluster      hierarchical, k-means, HDBSCAN          │
│  bitig.bayesian     OPTIONAL extra: PyMC models             │
├─────────────────────────────────────────────────────────────┤
│  bitig.features     MFW, n-grams, POS, deps, embeddings     │
│  bitig.preprocess   spaCy pipeline + DocBin cache           │
│  bitig.corpus       Corpus, Document, metadata              │
├─────────────────────────────────────────────────────────────┤
│  bitig.config       study.yaml schema, resolve, hash        │
│  bitig.io           ingest, serialize: parquet / JSON       │
│  bitig.plumbing     seeds, caching, logging, hashing        │
└─────────────────────────────────────────────────────────────┘
```

**Dependency rule:** modules never import "upward." Violations are enforced in CI via `ruff`'s
import-cycle detection and a custom layer-violation linter.

---

## 3. Core Data Model

Five types, all `pydantic` v2 dataclasses (for validation + JSON round-trip) where state matters,
plain dataclasses where they are passive containers.

### 3.1 `Document`

A single text. Immutable.

```python
class Document(BaseModel):
    id: str                           # stable id (filename stem by default)
    text: str
    metadata: dict[str, Any]          # user-defined keys from metadata.tsv
    hash: str                         # sha256 of text, lazy-computed
    _spacy_doc_ref: SpacyDocRef | None  # lazy DocBin reference
```

### 3.2 `Corpus`

Ordered collection of `Document`s sharing a metadata schema.

```python
class Corpus:
    documents: list[Document]
    metadata_schema: dict[str, type]  # inferred from metadata.tsv

    def filter(self, **query) -> Corpus: ...
    def groupby(self, field: str) -> dict[Any, Corpus]: ...
    def split(self, by: str, test_size: float, seed: int) -> tuple[Corpus, Corpus]: ...
    def hash(self) -> str: ...        # stable hash over sorted doc hashes + metadata
```

### 3.3 `FeatureMatrix`

A thin wrapper around `numpy.ndarray` (dense) or `scipy.sparse.csr_matrix` (sparse), with a sibling
`pandas.DataFrame` view for named access.

```python
class FeatureMatrix:
    X: np.ndarray | sparse.csr_matrix
    document_ids: list[str]           # row labels
    feature_names: list[str]          # column labels
    feature_type: str                 # "mfw" | "char_ngram" | "pos_ngram" | "embedding" | ...
    extractor_config: dict[str, Any]  # what produced it
    provenance_hash: str              # hash(extractor_config + corpus_hash + spacy_model + version)

    def as_dataframe(self) -> pd.DataFrame: ...
    def concat(self, other: FeatureMatrix) -> FeatureMatrix: ...  # column concatenation
```

The plain `X` is what sklearn transformers operate on; metadata travels alongside.

### 3.4 `StudyConfig`

Validated contents of `study.yaml`. Round-trips to/from YAML losslessly. See §9 for the full schema.

### 3.5 `Result`

Uniform return type from every method.

```python
class Result(BaseModel):
    method_name: str                  # "burrows_delta", "craig_zeta", ...
    params: dict[str, Any]
    values: dict[str, Any]            # e.g. {"distances": ndarray, "labels": list[str]}
    tables: list[pd.DataFrame]
    figures: list[Figure]             # lazy-rendered
    provenance: Provenance            # see §3.6
```

Serialisable to `parquet` (tables) + `json` (params, values, provenance) + figure files.

### 3.6 `Provenance`

```python
class Provenance(BaseModel):
    bitig_version: str
    python_version: str
    spacy_model: str
    spacy_version: str
    corpus_hash: str
    feature_hash: str | None
    seed: int
    timestamp: datetime
    resolved_config: dict[str, Any]   # the full study.yaml as actually run
```

Every `result.json` contains a `Provenance` block; no run is ever unreproducible.

---

## 4. Preprocessing Pipeline

### 4.1 spaCy backbone

- **Default model:** `en_core_web_trf` (per user preference; transformer-based, highest accuracy).
- **Device:** auto-detect with preference order `mps > cuda > cpu`; overridable via `--device`.
- **Batching:** all parsing goes through `nlp.pipe()` with sensible defaults.
- **Per-corpus override:** users may specify a different spaCy model in `study.yaml` or via
  `--spacy-model`. Non-English corpora use the appropriate spaCy model.
- **Pipeline components:** by default all components; users may `exclude` components via config
  (e.g., `exclude: [ner]`) to speed up parsing when those annotations are not used by any feature.

### 4.2 DocBin caching

- Parsed spaCy docs are serialized as `DocBin` files at `./.bitig/cache/docbin/`.
- **Cache key:** `sha256(document_hash || spacy_model || spacy_version || enabled_components)`.
- Cache is checked before every parse. Cache hit → skip parsing entirely.
- `bitig cache clear | list | size | export | import` for cache management. `export/import` enable
  sharing pre-parsed caches across a team.

### 4.3 Normalization

Per-feature, not global. Each feature extractor accepts:

- `lowercase: bool` (default: `false`)
- `strip_punct: bool` (default: `false` except for word-MFW where it is `true` — mirroring Stylo)
- `collapse_numerals: bool` (default: `false`)
- `expand_contractions: bool` (default: `false`)

Global defaults are configurable in `study.yaml → preprocess.normalize`; per-feature values override.

---

## 5. Feature Extractors (v0.1)

All extractors live in `bitig.features` and implement `sklearn.base.BaseEstimator` +
`TransformerMixin`. Shape contract: `fit_transform(corpus: Corpus) -> FeatureMatrix`.

| Extractor | Params | Output columns | Notes |
|---|---|---|---|
| `MFWExtractor` | `n`, `min_df`, `max_df`, `scale={zscore|l1|l2|none}`, `culling`, normalization flags | top-N words by frequency | `scale=zscore` is Burrows Delta input |
| `CharNgramExtractor` | `n` (range), `include_boundaries` | char n-grams | n ∈ 2..6 standard |
| `WordNgramExtractor` | `n` (range) | word n-grams | n ∈ 1..3 standard |
| `PosNgramExtractor` | `n`, `tagset={coarse|fine}` | POS n-grams | coarse = UPOS, fine = spaCy fine tags |
| `DependencyBigramExtractor` | — | `(head_lemma, dep, child_lemma)` triples | new vs. Stylo |
| `FunctionWordExtractor` | `wordlist` (bundled EN default) | FW frequencies | extensible list |
| `PunctuationExtractor` | — | per-punctuation frequency | |
| `LexicalDiversityExtractor` | `indices=[...]` | TTR, MATTR, MTLD, HD-D, Yule's K, Yule's I, Herdan's C, Simpson's D | selectable |
| `ReadabilityExtractor` | `indices=[...]` | Flesch, Flesch-Kincaid, Gunning Fog, Coleman-Liau, ARI, SMOG | selectable |
| `SentenceLengthExtractor` | — | mean, SD, skew of sentence length in tokens | |
| `SentenceEmbeddingExtractor` *(extra)* | `model`, `pool={mean|cls|max}` | fixed-d embedding vector | requires `bitig[embeddings]` |
| `ContextualEmbeddingExtractor` *(extra)* | `model`, `layer`, `pool` | BERT-layer-k mean/CLS | requires `bitig[embeddings]` |

Extractors are **composable**: `FeatureMatrix.concat(other)` column-stacks matrices with aligned
document IDs, enabling multi-view features in a single classifier.

---

## 6. Methods Catalog

All methods accept `FeatureMatrix` (preferred) or `Corpus` (a sensible default extractor is selected).
Return type is `Result`. Every method records its `feature_hash` in provenance.

### 6.1 Delta family (`bitig.methods.delta`)

Implementations follow primary literature:

| Variant | Reference | Notes |
|---|---|---|
| Burrows Delta | Burrows 2002 | `z-score` then mean absolute difference |
| Eder Delta | Eder 2015 | rank-weighted Burrows |
| Eder's Simple Delta | Eder 2017 | L1, no scaling |
| Argamon Linear Delta | Argamon 2008 | L2 on z-scored features |
| Cosine Delta | Smith & Aldridge 2011; Evert et al. 2017 | cosine on z-scored features |
| Quadratic Delta | Argamon 2008 | squared-L2 |

Each Delta variant is a sklearn `ClassifierMixin` — a nearest-author-centroid classifier under the
chosen metric, exposing `fit`, `predict`, `predict_proba`, `decision_function`.

### 6.2 Craig's Zeta (`bitig.methods.zeta`)

- **Classical Zeta** (Burrows 2007 / Craig & Kinney 2009): `proportion_A(word) - proportion_B(word)` where
  proportion = fraction of *texts* in the group that contain the word (binarised counts).
- **Eder's Zeta variant** (Eder 2017): smoothed proportions with Laplace correction.
- Significance: permutation test (configurable replicates, default 1000) producing p-values per term.
- Returns a `Result` whose `tables` include top-K preferred and top-K dispreferred terms with p-values.

### 6.3 Dimensionality reduction (`bitig.reduce`)

- PCA (sklearn) — default.
- MDS (classical + non-metric; sklearn).
- t-SNE (sklearn).
- UMAP (`umap-learn`, already sklearn-compatible).

### 6.4 Clustering (`bitig.cluster`)

- Hierarchical (Ward / average / complete / single linkage) — `scipy.cluster.hierarchy` +
  thin sklearn wrapper.
- K-means — sklearn.
- HDBSCAN — `hdbscan` package.

All expose sklearn's `ClusterMixin` API; dendrograms are rendered by the viz layer from the linkage
matrix.

### 6.5 Bootstrap consensus trees (`bitig.methods.consensus`)

Implementation after Eder 2017:

- Iterate over MFW bands (e.g., `[100, 200, ..., 1000]`), optionally with replicate
  random-subsample bootstraps per band (default 100).
- For each band × replicate, compute Delta distances + Ward dendrogram.
- Aggregate clade support: each internal clade's support is the fraction of individual dendrograms
  in which it appears.
- Export consensus tree as Newick (compatible with ETE3 / Biopython.Phylo / FigTree).
- Visualise as rectangular or unrooted radial tree with clade support annotations.

### 6.6 Classification (`bitig.classify`)

Thin sklearn wrappers with stylometry-aware CV strategies.

- **Estimators** (aliased in config): `svm_linear`, `svm_rbf`, `logreg`, `rf`, `hgbm`.
- **Cross-validation kinds:**
  - `stratified` — `StratifiedKFold` (standard multi-class CV).
  - `loao` — `LeaveOneGroupOut` with groups from a metadata field; the **right primitive for
    leave-one-author-out**, which matters any time author and target label might be confounded.
  - `group_kfold` — `GroupKFold` for arbitrary grouping.
  - `leave_one_text_out` — `LeaveOneOut` when the unit of interest is the individual text.
- **Metrics:** accuracy, macro-F1, per-author precision/recall/F1 (the `per_author` metric emits the
  full per-class precision/recall/F1 table from sklearn's `classification_report`), confusion matrix.
  All computed via sklearn's `classification_report` + `confusion_matrix`.
- **Permutation importance:** `sklearn.inspection.permutation_importance` on the final feature
  matrix, reported per feature.
- **Out-of-corpus prediction:** `bitig classify predict --model model.pkl <new-corpus>` loads a
  persisted estimator and labels unseen texts. Persistence via `joblib`.

### 6.7 Bayesian extras (`bitig.bayesian`, optional)

Requires `bitig[bayesian]` (pulls `pymc`, `arviz`). Both of the following ship in v0.1:

- **`BayesianAuthorshipAttributor`** — Wallace–Mosteller-style log-posterior over candidate authors
  from per-token rates with Beta priors, modernized with proper regularisation and posterior sampling.
  Implements `ClassifierMixin`; returns per-author posterior probabilities.
- **`HierarchicalGroupComparison`** — PyMC hierarchical model with varying intercepts / varying
  slopes for per-author style components. Designed for L2-vs-native, proficiency-band, and
  L1-group analyses. Reports posterior distributions over group effects with HDI intervals and
  posterior predictive checks.

When the `[bayesian]` extra is not installed, the import gracefully no-ops and CLI commands emit a
"install `bitig[bayesian]` to enable" hint.

---

## 7. sklearn Interoperability

A cross-cutting design principle, not a module.

### 7.1 Protocol adherence

Every class below inherits the stated sklearn mixin and implements its full protocol:

- Feature extractors → `BaseEstimator, TransformerMixin`.
- Delta classifiers → `BaseEstimator, ClassifierMixin`.
- Zeta → `BaseEstimator, TransformerMixin` (produces contrastive vocabulary weights).
- Clustering → `BaseEstimator, ClusterMixin`.
- Bayesian attributor → `BaseEstimator, ClassifierMixin`.

`get_params`, `set_params`, `fit`, `transform` / `predict` all work as expected.

### 7.2 Pipeline compatibility

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, LeaveOneGroupOut

from bitig.features import MFWExtractor
from bitig.methods import BurrowsDelta

pipe = Pipeline([
    ("feat", MFWExtractor(n=1000, min_df=2)),
    ("scale", StandardScaler()),
    ("clf",   BurrowsDelta()),
])

scores = cross_validate(
    pipe, corpus, y=corpus.metadata_column("author"),
    cv=LeaveOneGroupOut(),
    groups=corpus.metadata_column("author"),
    scoring=["accuracy", "f1_macro"],
)
```

### 7.3 Bidirectional config ↔ pipeline

- `bitig.to_sklearn_pipeline(study: StudyConfig) -> Pipeline` builds a pipeline from a config.
- `bitig run study.yaml` executes what is fundamentally a sklearn pipeline under the hood, adding
  corpus loading, caching, provenance, and artifact persistence.
- `bitig.from_sklearn_pipeline(pipe: Pipeline) -> StudyConfig` is a v0.2 convenience.

### 7.4 Utilities we inherit rather than reinvent

`StandardScaler`, `MaxAbsScaler`, `StratifiedKFold`, `LeaveOneGroupOut`, `GroupKFold`,
`LeaveOneOut`, `cross_validate`, `permutation_importance`, `classification_report`,
`confusion_matrix`, `joblib` for persistence.

---

## 8. CLI Surface

### 8.1 Command tree (Typer)

```
bitig init <name>            scaffold a project directory
bitig ingest <path>          parse corpus, cache DocBins, write corpus.parquet
bitig features <corpus>      build MFW / n-gram / embedding matrices
bitig delta <corpus|matrix>  Delta-family methods
bitig zeta <corpus>          Craig's Zeta (two-group comparison)
bitig reduce <matrix>        PCA, MDS, t-SNE, UMAP
bitig cluster <matrix>       hierarchical clustering + dendrogram
bitig consensus <corpus>     bootstrap consensus tree over MFW bands
bitig classify <corpus>      sklearn classifiers + CV + permutation importance
bitig plot <result>          re-render figures from a saved Result
bitig report <result|study>  generate HTML / Markdown report
bitig run <study.yaml>       execute a full config-driven study
bitig shell [<corpus>]       launch the interactive shell
bitig info                   versions, cache stats, resolved config
bitig cache [clear|list|size|export|import]
```

### 8.2 Global flags

Every command honours:

```
--seed <int>              reproducibility (default: 42, overridable in config)
--cache-dir <path>        default: ./.bitig/cache
--spacy-model <name>      default: en_core_web_trf
--device cpu|mps|cuda     default: auto-detect (mps preferred on Apple Silicon)
--output <path>           where to write Result JSON + figures
--format png|pdf|eps|svg  figure output format; repeatable
--dpi <int>               default: 300
--report html|md|none     opt-in report generation
--config <path>           explicit config file
--from <study.yaml>       inherit defaults from a config; CLI flags override
--name <label>            override the default timestamp run-directory name
--quiet / --verbose / --json
```

### 8.3 Worked example

```bash
bitig delta ./my-corpus \
    --method burrows --mfw 1000 --mfw-min 100 \
    --metadata metadata.tsv --group-by author \
    --output results/ --format pdf --dpi 300 --report html
```

### 8.4 Interactive shell

`bitig shell` launches a **wizard-first** guided workflow built on Rich + Questionary.

- Project-aware: if launched inside a scaffolded project, reads `study.yaml` as defaults.
- Each step shows the **equivalent CLI command** it is about to run, so users learn the scriptable
  form by using the shell.
- Every session offers to export the current selections as a `study.yaml`.
- **Escape to IPython:** a menu option drops the user into an embedded IPython REPL with the current
  `corpus`, `study`, and last `Result` in scope for ad-hoc exploration.

Illustrative first screen:

```
╭─ bitig shell ──────────────────────────────────────────╮
│ Corpus: ./efcamdat-sample   (1,240 docs, 4 groups)    │
│ Cache:  ./.bitig/cache      (spaCy trf, 3 days old)   │
╰────────────────────────────────────────────────────────╯

? What would you like to do?
  ❯ Build / inspect features
    Run Delta attribution
    Run Zeta comparison
    Cluster & visualize
    Classify (train / evaluate)
    Bootstrap consensus tree
    Reduce & plot (PCA / UMAP)
    Save current selections as study.yaml
    Drop to Python REPL (IPython)
    Exit
```

### 8.5 Config-driven runs

`bitig run study.yaml` executes the full declarative analysis defined in §9.

---

## 9. Configuration Schema (`study.yaml`)

### 9.1 Full example

```yaml
name: efcamdat-l2-vs-native
seed: 42

corpus:
  path: corpus/
  metadata: corpus/metadata.tsv
  filter:
    group: [native, L2]
    min_tokens: 300

preprocess:
  spacy:
    model: en_core_web_trf
    device: auto              # auto | cpu | mps | cuda
    exclude: []
  normalize:
    lowercase: false
    strip_punct: false

features:                     # list of named extractors; id is referenced by methods
  - {id: mfw1000, type: mfw, n: 1000, min_df: 2, scale: zscore}
  - {id: pos3,    type: pos_ngram, n: 3, tagset: coarse}
  - {id: embed,   type: sentence_embedding,
                  model: sentence-transformers/all-mpnet-base-v2,
                  pool: mean}

methods:
  - id: burrows
    kind: delta
    method: burrows
    features: mfw1000
    group_by: author

  - id: consensus
    kind: consensus
    feature_template: {type: mfw, bands: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}
    group_by: author
    replicates: 100

  - id: classify_svm
    kind: classify
    estimator: svm_linear
    features: [mfw1000, pos3]       # list ⇒ column-concatenate
    cv:
      kind: loao
      groups_from: author
      folds: null                   # null = determined by CV kind
    metrics: [accuracy, macro_f1, confusion, per_author]   # per_author ⇒ per-class precision/recall/F1

  - id: bayes_group                 # requires bitig[bayesian]
    kind: bayesian
    model: hierarchical_group
    features: mfw1000
    group_by: proficiency
    chains: 4
    samples: 2000

viz:
  format: [pdf, png]
  dpi: 300
  style: default
  palette: colorblind

report:
  format: html
  offline: false                    # true → self-contained (bundles plotly.js)
  include: [corpus, config, provenance, results]
  title: "EFCAMDAT: L2 vs. native authorship profile"

cache:
  dir: .bitig/cache
  reuse: true

output:
  dir: results/
  timestamp: true                   # overridden by --name
```

### 9.2 Config resolution

Precedence (first match wins):

1. Explicit CLI flags (`--mfw 1000`).
2. `--config` / `--from` file passed on the command line.
3. Project `study.yaml` (auto-detected when running inside a scaffolded project).
4. Package defaults.

(User-level `~/.config/bitig/defaults.yaml` is deferred to v0.2.)

`bitig config show --resolved` dumps the effective config *as it would run* for any command, so
users can verify layering before executing.

---

## 10. Visualization

### 10.1 Dual backend

Static and interactive share a single plotting API. Backend is chosen contextually, and every
context has a functional fallback if the optional `[viz]` extra (plotly) is not installed.

| Context | Primary backend (`[viz]` installed) | Fallback (no `[viz]`) |
|---|---|---|
| CLI with `--format pdf\|eps\|png\|svg` | matplotlib / seaborn | matplotlib / seaborn (same) |
| Interactive shell | plotly (Rich preview + open-in-browser) | matplotlib static window |
| `--report html` | plotly, embedded (CDN by default; `--offline` for bundled) | matplotlib PNGs embedded inline |
| `--report md` | matplotlib (PNG + EPS attached) | matplotlib (same) |

Static figures never degrade in quality based on extras; interactivity is what `[viz]` unlocks.

### 10.2 Plot catalog (per method)

| Method | Static | Interactive |
|---|---|---|
| `delta` | distance heatmap, author-centroid table, attribution matrix | hover-labeled heatmap |
| `zeta` | Craig's Zeta scatter, preference plot | hover-zoom scatter |
| `reduce` | 2D scatter with group colors, ellipses, labels | zoom + filter by group |
| `cluster` | rectangular & radial dendrograms, cophenetic heatmap | collapsible dendrogram |
| `consensus` | rectangular + unrooted radial tree (via ETE3) | interactive tree (plotly / phylocanvas) |
| `classify` | confusion matrix, permutation-importance bars, per-author F1 | same + hover |
| `bayesian` | posterior distributions (arviz), forest plots, HDI intervals | trace explorer |
| network (v0.2) | networkx + matplotlib; optional graphviz | pyvis / plotly network |

### 10.3 Publication defaults

- Default DPI: 300; `--dpi` overridable.
- Default figure formats: PDF + PNG. EPS, TIFF, SVG available via `--format`.
- Font stack: Source Serif Pro → Linux Libertine → Times New Roman fallback. Overridable via `--style`.
- Palette: colorblind-safe (`seaborn`'s `colorblind`).
- Figure widths preset for single-column (3.5"), 1.5-column (5"), double-column (7") journals.

### 10.4 Optional extras for viz

- `bitig[viz]` installs `ete3` (consensus tree rendering), `plotly`, `kaleido` (plotly static export).
- Without `[viz]`: interactive plots fall back to matplotlib (see §10.1 table); consensus trees are
  exported as Newick strings and rendered via `scipy.cluster.hierarchy.dendrogram` as a rectangular
  layout (no unrooted/radial). All user-facing messages make this degradation explicit and suggest
  `pip install bitig[viz]` to unlock the richer layouts.

---

## 11. Reporting

### 11.1 `bitig report <result|study.yaml> --format html|md [--pdf]`

Produces a single report containing, in order:

1. **Header:** study title, date, `bitig` version, corpus hash, seed.
2. **Corpus summary:** per-group counts, per-document length distribution, metadata fields, sample
   documents.
3. **Resolved config:** the fully-expanded `study.yaml` as actually executed.
4. **Preprocessing provenance:** spaCy model + version, DocBin cache hash, normalization choices.
5. **Results:** one section per method, with tables + figures + a brief narrative summary
   ("Burrows Delta with MFW=1000 correctly attributed 47/50 held-out texts; misattributions cluster
   around Author X…").
6. **Reproducibility appendix:** full command invocations, environment fingerprint (`pip freeze` of
   actually loaded extras), random seed, `lock.yaml` snapshot.

### 11.2 Output formats

- **HTML** (Jinja2 template): Plotly interactive figures, pandas-styled tables, MathJax for
  formulas. **CDN-linked Plotly by default; `--offline` inlines plotly.js** for self-contained reports.
- **Markdown**: same template; static PNG figures; Pandoc-ready for manuscript appendix conversion.
- **PDF**: optional via `weasyprint` (HTML route) or `pandoc` (MD route); **not a hard dependency**.
  Requires `bitig[reports]` extra.

---

## 12. Project Scaffold & Reproducibility

### 12.1 `bitig init <name>` output

```
my-study/
├── study.yaml                    canonical analysis config
├── corpus/
│   └── metadata.tsv              optional: filename → author, group, year, ...
├── .bitig/
│   ├── cache/
│   │   ├── docbin/               spaCy parses, keyed by (doc_hash, model, version)
│   │   └── features/             FeatureMatrix parquet, keyed by extractor config
│   ├── logs/
│   └── lock.yaml                 pinned package/model versions + corpus hash
├── results/
│   └── 2026-04-17T14-22-03/      timestamped per-run (default)
│       ├── delta/
│       │   ├── result.json
│       │   ├── distance.parquet
│       │   ├── dendrogram.pdf
│       │   └── dendrogram.png
│       ├── classify/
│       └── run.log
├── reports/
│   └── 2026-04-17T14-22-03.html
└── README.md                     pre-filled with how-to-reproduce
```

### 12.2 Run-directory naming

- **Default:** timestamped directory `results/YYYY-MM-DDThh-mm-ss/`.
- **Override:** `bitig run study.yaml --name ablation-no-pos` writes to
  `results/ablation-no-pos/`.

### 12.3 Caching

- **Corpus hash:** stable hash over `(sorted(file_hashes), metadata_hash)`.
- **Feature cache key:** `sha256(extractor_config || corpus_hash || spacy_model || spacy_version)`.
- **Cache is content-addressable:** changing any input invalidates the relevant cache entries only.

### 12.4 `lock.yaml`

Pinned on first run, checked on every subsequent run:

- `bitig` version.
- Python version.
- Full `pip freeze` of actually-loaded extras.
- spaCy model + version.
- Corpus hash.

On mismatch, `bitig` **warns and proceeds** by default (friendlier for iteration); `--strict-lock`
hard-blocks without explicit `--force`. The warning lists each drift so users can review before acting.

### 12.5 Seeds

Every stochastic method receives a seeded `numpy.random.Generator` derived from
`hash(study_seed || method_id)` so individual methods are independently reproducible even if other
methods are added/removed from the config.

### 12.6 Git-friendliness

- `.bitig/` is auto-added to `.gitignore` at init.
- `study.yaml`, `metadata.tsv`, `README.md`, `results/*/result.json`, and `reports/*.md` are
  intended to be committed.

---

## 13. Testing & Quality

### 13.1 Testing strategy

- **Framework:** `pytest` + `pytest-cov` + `pytest-xdist` + `hypothesis`.
- **Bundled fixture corpora** (all public domain):
  - **Federalist Papers** (Mosteller & Wallace 1964) — canonical stylometry benchmark.
    Every Delta method must reproduce the established Hamilton-vs-Madison attributions on the
    disputed papers.
  - **Small literary sample** — 10 authors × 5 works from Gutenberg, for clustering and consensus
    sanity.
  - **Synthetic mini-corpus** — 20 texts of fixed vocabulary for deterministic unit tests.
- **Golden-file tests** for every distance metric; any change to numerical output must be justified
  in a commit message and golden files updated in the same commit.
- **Numerical-parity smoke tests against R's `Stylo`:** Burrows Delta on the Federalist Papers must
  match `Stylo`'s output to within `1e-6`. Stylo's outputs are checked into the repo once.
- **CLI integration tests** via `typer.testing.CliRunner`.
- **Doctests** for the library API.
- **Coverage targets:** ≥90% on `bitig.features`, `bitig.methods`, `bitig.reduce`, `bitig.cluster`,
  `bitig.classify`, `bitig.config`; ≥70% on `bitig.viz` and `bitig.cli`.

### 13.2 Quality tooling

- **`ruff`** for linting + import sorting (replaces flake8, isort, black).
- **`mypy --strict`** on the public API surface (everything re-exported from `bitig/__init__.py`);
  relaxed internally.
- **`pre-commit`** hooks: ruff, mypy, trailing whitespace, large-file check.
- **Custom layer-violation linter** to enforce the architecture rule from §2.

### 13.3 Continuous integration

- **GitHub Actions** matrix: Python **3.11, 3.12, 3.13** × **Ubuntu, macOS**.
- **Weekly scheduled run** against spaCy nightly to catch upstream breakage early.
- **Docs build** on every PR; deployment on tag.

---

## 14. Distribution

### 14.1 Build & packaging

- **Build system:** `hatchling`. `pyproject.toml` only (no `setup.py`).
- **Project manager:** `uv` (per user preference; `uv sync`, `uv run pytest`, etc.).
- **Python support:** **3.11+**.

### 14.2 Core dependencies

```
numpy, scipy, pandas, pyarrow, joblib
scikit-learn, umap-learn, hdbscan
spacy >= 3.7
matplotlib, seaborn
typer[all], rich, questionary
pyyaml, pydantic >= 2
jinja2
```

### 14.3 Optional extras

| Extra | Installs | Enables |
|---|---|---|
| `bitig[bayesian]` | `pymc`, `arviz` | Wallace–Mosteller + hierarchical group models |
| `bitig[embeddings]` | `sentence-transformers`, `torch` | Sentence + contextual embeddings |
| `bitig[viz]` | `plotly`, `kaleido`, `ete3` | Plotly backend + consensus tree rendering |
| `bitig[reports]` | `weasyprint` | HTML → PDF report export |
| `bitig[all]` | union of the above | all optional features |

### 14.4 Publication & licensing

- **PyPI name:** `bitig` (availability to be verified pre-release; fallback: `bitig-stylometry`).
- **License:** **BSD-3-Clause** (compatible with scientific Python ecosystem; permissive;
  academic-friendly).
- **Documentation:** MkDocs Material + `mkdocstrings`, hosted on GitHub Pages.
- **Tutorials:**
  1. Federalist Papers replication (the "hello world" of stylometry).
  2. L2-vs-native EFCAMDAT-style comparison.
  3. Embedding-based authorship attribution.
  4. Bayesian hierarchical group comparison.
- **`CITATION.cff`** from day one.

---

## 15. Roadmap

### 15.1 v0.1 — the Stylo replacement + Bayesian author-group comparison

Everything above. The release bar is:

- Full feature parity with `Stylo` on the Delta family, Zeta, PCA/MDS, clustering, dendrograms,
  bootstrap consensus trees.
- sklearn-transformer-compatible feature extractors and classifiers.
- Wizard shell + one-shot CLI + config-driven studies + project scaffolds.
- Wallace–Mosteller + hierarchical group comparison as `[bayesian]` extra.
- Federalist Papers replication passes automated parity tests.

### 15.2 v0.2 — advanced stylometry (targeted 3–6 months post-v0.1)

- Rolling stylometry (`bitig rolling`).
- Author-similarity networks + community detection (Louvain / Leiden).
- LUAR, Wegmann et al. style-embedding plugins.
- Bayesian authorship verification (same-author-yes/no with evidence strength).
- User-level `~/.config/bitig/defaults.yaml`.
- `bitig compare` — side-by-side multi-method report.
- `bitig.from_sklearn_pipeline(pipe) -> StudyConfig`.

### 15.3 v0.3 — research-lab grade

- Plugin API for external extractors and methods (entry-point discovery).
- Active-learning-assisted attribution (query hardest unlabeled texts first).
- Cross-lingual stylometry (multilingual spaCy / XLM-R embeddings).
- TEI/XML and CoNLL-U ingest formats.

---

## 16. Open questions deferred to implementation-planning

- Exact format of the `FeatureMatrix` wrapper: subclass `numpy.ndarray`, wrap it, or dual-store
  `ndarray` + metadata DataFrame. Decision during implementation based on sklearn compatibility
  testing.
- Whether to vendor a small Federalist Papers fixture inside the package (+~1 MB) or fetch it on
  first test run. Leaning toward vendoring for offline CI.
- Details of the `bitig shell` IPython-escape plumbing (`IPython.embed()` vs.
  `IPython.start_ipython()` with scope injection).
- Exact public-API surface re-exported from `bitig/__init__.py` (mypy-strict boundary).
