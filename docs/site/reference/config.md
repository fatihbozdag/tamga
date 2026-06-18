# study.yaml schema

The declarative study config consumed by `bitig run`. A minimal example:

```yaml
name: my-study
seed: 42

corpus:
  path: corpus
  metadata: corpus/metadata.tsv

features:
  - id: mfw200
    type: mfw
    n: 200
    scale: zscore
    lowercase: true

methods:
  - id: burrows
    kind: delta
    method: burrows
    features: mfw200
    group_by: author
```

The config shape is validated by a Pydantic model with `extra="forbid"` — unknown top-level keys
on `corpus`, `preprocess`, `viz`, `report`, `cache`, and `output` are rejected. (Unknown keys on a
**feature** or **method** entry are different: they are collected into that entry's `params` rather
than rejected — see [features](#features) / [methods](#methods).)

## Top-level keys

| Key | Type | Required | Description |
|---|---|---|---|
| `name` | str | no | Study name; shows in reports (default: `"unnamed-study"`) |
| `seed` | int | no | Default seed (42). Threaded to every stochastic method. |
| `corpus` | object | **yes** | Corpus config (below) |
| `features` | list | no | Feature extractors (default: empty — but you need at least one to run a method) |
| `methods` | list | no | Methods to run (default: empty) |
| `preprocess` | object | no | Language + spaCy + normalization settings |
| `viz` | object | no | Figure format / DPI / style |
| `report` | object | no | Report format and contents |
| `output` | object | no | Output directory / timestamping |
| `cache` | object | no | DocBin cache directory + reuse toggle |

## corpus

```yaml
corpus:
  path: corpus                    # directory of .txt files (required)
  metadata: corpus/metadata.tsv   # optional TSV with filename + arbitrary fields
  filter:                         # optional: subset the corpus before running
    role: [train]
```

!!! note
    There is no `corpus.strict` field. Strict-vs-lenient metadata coverage is a flag on the
    `bitig ingest` command (`--strict` / `--no-strict`), not a `study.yaml` setting.

## features

Each feature extractor is a dict with an `id` (referenced by methods), a `type`, and
type-specific params. Any key other than `id`/`type` is folded into the extractor's `params`
and validated against the extractor signature at execution time.

### Supported types

| type | params |
|---|---|
| `mfw` | `n`, `min_df`, `max_df`, `scale` ({none, zscore, l1, l2}), `lowercase` |
| `word_ngram` | `n` (int or [min, max]), `lowercase`, `scale` |
| `char_ngram` | `n` (int or [min, max]), `include_boundaries`, `scale` |
| `pos_ngram` | `n`, `tagset` ({coarse, fine}), `scale` |
| `dependency_bigram` | `scale` |
| `function_word` | `wordlist` (optional list or path), `language`, `scale` |
| `punctuation` | (none) |
| `lexical_diversity` | `indices` (subset of the eight indices) |
| `readability` | `indices` (subset of the per-language indices) |
| `sentence_length` | (none) |
| `sentence_embedding` | `model`, `language`, `device` (extra: `bitig[embeddings]`) |
| `contextual_embedding` | `model`, `language`, `layer`, `pool`, `device` (extra: `bitig[embeddings]`) |

## methods

Each method is a dict with an `id`, a `kind`, an optional `features` (feature id), an optional
`group_by`/`cv`, plus `params` (any other key is folded into `params`).

### Supported kinds

| kind | Description |
|---|---|
| `delta` | Nearest-centroid attribution (`method: burrows` by default; lands in `params`) |
| `rolling_delta` | Rolling-window Delta for collaborative / mixed-authorship texts |
| `verify` | One-class authorship verification (e.g. General Impostors) |
| `zeta` | Craig's Zeta; requires `group_by` and either inferred or specified `params.group_a` / `group_b` |
| `reduce` | Dim-reduction (default PCA); `params.n_components` |
| `cluster` | Hierarchical (default Ward); `params.n_clusters`, `params.linkage` |
| `consensus` | Bootstrap consensus tree; `params.mfw_bands`, `params.replicates` |
| `classify` | sklearn classifier; `params.estimator`, `cv.kind`, `cv.folds`, `cv.groups_from` |
| `bayesian` | Wallace–Mosteller attribution + hierarchical group comparison |

### cv (cross-validation, for `classify`)

```yaml
cv:
  kind: stratified        # stratified | loao | group_kfold | leave_one_text_out
  folds: 5                # for stratified / group_kfold
  groups_from: author     # required for loao / group_kfold
```

## preprocess

```yaml
preprocess:
  language: en              # default; one of the registered languages
  spacy:
    model: null            # default null → resolved per-language (e.g. en_core_web_trf)
    backend: null          # null → resolved from the language (spacy | spacy_stanza)
    device: auto           # auto | cpu | mps | cuda
    exclude: []            # spaCy pipeline components to disable
  normalize:
    lowercase: false
    strip_punct: false
    collapse_numerals: false
    expand_contractions: false
```

## viz

```yaml
viz:
  format: [pdf, png]       # any of pdf, png, svg, eps, tiff
  dpi: 300
  style: default
  palette: colorblind
```

## report

```yaml
report:
  format: none             # none | html | md
  offline: false           # inline assets for a self-contained HTML file
  include: [corpus, config, provenance, results]
  title: null
```

## output

```yaml
output:
  dir: results/            # default
  timestamp: true          # wrap runs in timestamped subdirectories
```

## cache

```yaml
cache:
  dir: .bitig/cache        # spaCy DocBin cache location (default)
  reuse: true              # reuse cached parses when inputs are unchanged
```

## A realistic multi-method example

```yaml
name: federalist
seed: 42
output: { dir: results, timestamp: false }

corpus:
  path: corpus
  metadata: corpus/metadata.tsv
  filter:
    role: [train]

features:
  - id: mfw200
    type: mfw
    n: 200
    scale: zscore
    lowercase: true

methods:
  - id: burrows
    kind: delta
    method: burrows
    features: mfw200
    group_by: author

  - id: pca
    kind: reduce
    features: mfw200
    params: { n_components: 2 }

  - id: ward
    kind: cluster
    features: mfw200
    params: { n_clusters: 3, linkage: ward }

  - id: zeta_h_m
    kind: zeta
    group_by: author
    params:
      top_k: 50
      group_a: Hamilton
      group_b: Madison
```
