# CLI reference

Every bitig CLI command. Installed as `bitig` via the `bitig` entry point.

Every command supports `--help`; the flags below are the most common ones, but `bitig <cmd> --help`
is always the authoritative source.

## Project scaffolding

### `bitig init <name>`

Scaffold a new project directory.

```bash
bitig init my-study [--target DIR] [--language en] [--force]
```

- `--target, -t` — parent directory to create the project in (default: current directory)
- `--language, -l` — corpus language for the scaffolded `study.yaml` (default: `en`)
- `--force` — overwrite an existing directory

Creates:

```
my-study/
├── corpus/             # drop .txt files here
│   └── metadata.tsv    # one row per file
├── study.yaml          # declarative study config
└── README.md           # short pointer
```

The scaffolded `study.yaml` declares one MFW feature set (top 1000 words) and a Burrows Delta
method. Add further feature/method blocks to broaden the analysis (see the
[study.yaml schema](config.md)).

## Ingestion

### `bitig ingest <path>`

Parse a corpus directory with optional metadata.

```bash
bitig ingest corpus/ --metadata corpus/metadata.tsv [--strict|--no-strict]
```

- `--metadata, -m` — TSV mapping filename → author, group, year, …
- `--strict` (default) — raise if any document lacks a metadata row
- `--no-strict` — allow partial coverage
- `--language, -l` — corpus language (drives the spaCy backend; default `en`)
- `--spacy-model` — override the spaCy pipeline name
- `--cache-dir` — where to write the cached parses
- `--exclude` — spaCy pipeline components to disable

Output is cached as a spaCy DocBin for subsequent commands.

### `bitig info`

Print runtime information: bitig / Python / platform / spaCy versions, plus the configured
language if a `study.yaml` is present in the working directory. Takes no arguments. (It does
**not** summarise an ingested corpus — `bitig ingest` reports loaded/parsed document counts.)

## Features

### `bitig features <path>`

Build a feature matrix and save it to parquet.

```bash
bitig features corpus/ --metadata corpus/metadata.tsv --type mfw --n 1000 --output features.parquet
```

- `--type` — `mfw` (default), `char_ngram`, `word_ngram`, `function_word`, `punctuation`
- `--n` — number of features for MFW/n-gram types (default: 1000)
- `--min-df`, `--scale`, `--lowercase` — MFW tuning knobs
- `--metadata, -m`, `--output, -o`

!!! note
    `lexical_diversity`, `readability`, POS/dependency n-grams, and embeddings are available
    through the Python API (see [Features](../concepts/features.md)) but are not wired into the
    `bitig features` CLI command.

## Methods

`--metadata` is **required** for `delta`, `zeta`, `classify`, and `bayesian`, and optional for
`reduce`, `cluster`, `consensus`, and `embed`. `--group-by`, `--seed`, and `--mfw` are accepted
by some commands but not all — run `bitig <cmd> --help` for the exact set.

| Command | Does |
|---|---|
| `bitig delta <path> --method {burrows,eder,eder_simple,argamon,cosine,quadratic}` | Fit Delta, print per-author predictions. Also: `--mfw`, `--mfw-min`, `--group-by`, `--test-filter` |
| `bitig zeta <path> [--group-a X --group-b Y]` | Craig's Zeta contrast between two author groups. Also: `--variant {classic,eder}`, `--top-k`, `--group-by`. Groups auto-selected if omitted |
| `bitig reduce <path> --method {pca,mds,tsne,umap} --n-components 2` | Dimensionality reduction → parquet. Also: `--mfw`, `--output/-o`. (UMAP needs `bitig[cluster]`) |
| `bitig cluster <path> --method {hierarchical,kmeans,hdbscan} --n-clusters N` | Clustering. Also: `--linkage`, `--mfw`, `--seed`, `--output/-o`. (HDBSCAN needs `bitig[cluster]`) |
| `bitig consensus <path>` | Bootstrap consensus tree across MFW bands. Also: `--bands`, `--replicates`, `--subsample`, `--support-threshold`, `--seed`, `--output/-o` |
| `bitig classify <path> --estimator {logreg,svm_linear,svm_rbf,rf,hgbm} --cv-kind {stratified,loao,leave_one_text_out}` | sklearn classifier + stylometry-aware CV. Also: `--groups-by`, `--folds`, `--mfw`, `--seed`. (`loao` requires `--groups-by`) |
| `bitig embed <path>` | Sentence or contextual embeddings (extra: `bitig[embeddings]`). Also: `--model`, `--pool`, `--output/-o` |
| `bitig bayesian <path>` | Wallace–Mosteller attribution + hierarchical group comparison. Also: `--group-by`, `--test-filter`, `--mfw`, `--prior-alpha` |

## Orchestration

### `bitig run <study.yaml>`

Execute a full declarative study end-to-end.

```bash
bitig run study.yaml --name demo [--output results/]
```

- `--name` — run name (subdirectory under the output dir)
- `--output, -o` — output directory (default from the study's `output.dir`)

Writes every method's `Result` to its own subdirectory plus a resolved-config record.

### `bitig report <run-dir>`

Render a Jinja2 HTML or Markdown report from a run directory.

```bash
bitig report results/demo --output results/demo/report.html [--format html|md] [--title "My report"]
```

### `bitig plot <run-dir>`

!!! warning "Stub"
    `bitig plot` is currently a stub and does **not** render figures yet (`--format`, `--dpi` are
    accepted but no-ops). To render figures today, call the `bitig.viz.plot_*` functions directly
    from Python, or use the `viz` block in `study.yaml` with `bitig run`.

### `bitig shell [<corpus>]`

Interactive Rich-based wizard that walks you through a study setup.

```bash
bitig shell [corpus/ --metadata corpus/metadata.tsv]
```

## Forensic cases

### `bitig case <cmd>`

Manage Forensic Lab Cases — a chain-of-custody-tracked, signable workspace for a single
examination. Pass `--cases-dir` to point at the case store.

- `bitig case new` — create a new Case directory
- `bitig case list` — list every Case in a Rich table
- `bitig case open` — print the case path + one-line summary
- `bitig case status` — full status: record fields, evidence inventory, custody check
- `bitig case fork` — clone a Case into an unsigned descendant for further iteration
- `bitig case sign` — sign & lock a Case (read-only afterwards)
- `bitig case verify` — verify a signed Case's chain-of-custody seal

## Cache

### `bitig cache <cmd>`

Manage the spaCy DocBin cache produced by `bitig ingest`:

- `bitig cache size` — show total bytes stored
- `bitig cache list` — list cache keys
- `bitig cache clear` — delete every entry

## Getting help

Every command supports `--help`:

```bash
bitig --help
bitig run --help
```
