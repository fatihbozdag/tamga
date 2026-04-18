# CLI reference

Every tamga CLI command. Installed as `tamga` via the `tamga` entry point.

## Project scaffolding

### `tamga init <name>`

Scaffold a new project directory.

```bash
tamga init my-study
```

Creates:

```
my-study/
├── corpus/             # drop .txt files here
│   └── metadata.tsv    # one row per file
├── study.yaml          # declarative study config
└── README.md           # short pointer
```

## Ingestion

### `tamga ingest <path>`

Parse a corpus directory with optional metadata.

```bash
tamga ingest corpus/ --metadata corpus/metadata.tsv [--strict|--no-strict]
```

- `--strict` (default) — raise if any document lacks a metadata row
- `--no-strict` — allow partial coverage

Output is cached as a spaCy DocBin for subsequent commands.

### `tamga info`

Summarise an ingested corpus: document count, metadata fields + value distributions,
total tokens.

## Features

### `tamga features <path>`

Build a feature matrix and print a summary.

```bash
tamga features corpus/ --metadata corpus/metadata.tsv --type mfw --n 500
```

Types: `mfw`, `word_ngram`, `char_ngram`, `function_word`, `punctuation`,
`lexical_diversity`, `readability`.

## Methods

All method commands accept `--metadata`, `--group-by <field>`, `--seed <int>`.

| Command | Does |
|---|---|
| `tamga delta <path> --method {burrows,argamon,eder,cosine,quadratic}` | Fit Delta, print per-author predictions |
| `tamga zeta <path> --group-a X --group-b Y` | Craig's Zeta contrast between two author groups |
| `tamga reduce <path> --method {pca,mds,tsne,umap} --n-components 2` | Dimensionality reduction → parquet |
| `tamga cluster <path> --method {hierarchical,kmeans,hdbscan} --n-clusters N --seed S` | Clustering with `--seed` for k-means |
| `tamga consensus <path>` | Bootstrap consensus tree across MFW bands |
| `tamga classify <path> --estimator {logreg,svm_linear,svm_rbf,rf,hgbm} --cv-kind {stratified,loao,leave_one_text_out}` | sklearn classifier + stylometry-aware CV |
| `tamga embed <path>` | Sentence or contextual embeddings (extra: `tamga[embeddings]`) |
| `tamga bayesian <path>` | Wallace–Mosteller attribution + hierarchical group comparison (extra: `tamga[bayesian]`) |

## Orchestration

### `tamga run <study.yaml>`

Execute a full declarative study end-to-end.

```bash
tamga run study.yaml --name demo [--output-dir results/]
```

Writes every method's `Result` to its own subdirectory plus a `resolved_config.json`.

### `tamga report <run-dir>`

Render a Jinja2 HTML or Markdown report from a run directory.

```bash
tamga report results/demo --output results/demo/report.html [--format html|md]
```

### `tamga plot <run-dir>`

Render per-method figures (PCA scatter, Ward dendrogram, Zeta preference plot, …) from
saved Results.

### `tamga shell`

Interactive Rich-based wizard that walks you through a study setup.

## Cache

### `tamga cache <cmd>`

Manage the spaCy DocBin cache produced by `tamga ingest`:

- `tamga cache info` — summarise
- `tamga cache clear` — remove

## Getting help

Every command supports `--help`:

```bash
tamga --help
tamga run --help
```
