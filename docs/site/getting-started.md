# Getting started

## Install

bitig requires Python 3.11+.

=== "uv (recommended)"

    ```bash
    uv pip install bitig
    python -m spacy download en_core_web_trf
    ```

=== "pip"

    ```bash
    pip install bitig
    python -m spacy download en_core_web_trf
    ```

### Optional extras

```bash
uv pip install "bitig[bayesian]"    # PyMC + arviz for hierarchical models
uv pip install "bitig[embeddings]"  # sentence-transformers + contextual BERT
uv pip install "bitig[viz]"         # plotly, kaleido, ete3
uv pip install "bitig[reports]"     # weasyprint for PDF report export
uv pip install "bitig[docs]"        # mkdocs + material theme (build this site)
```

## A study in five commands

```bash
bitig init my-study          # (1) scaffold a project directory
cd my-study
# (2) drop .txt files into corpus/
# (3) fill in corpus/metadata.tsv — one row per file with filename → author, group, year, ...
bitig ingest corpus/ --metadata corpus/metadata.tsv  # (4) parse + cache
bitig info                    # (5a) verify the ingest
bitig run study.yaml --name demo   # (5b) run the declared study
bitig report results/demo --output results/demo/report.html
```

The project skeleton from `bitig init` includes a working `study.yaml` with Burrows Delta
+ PCA + Zeta on 200 most-frequent words, so the above sequence runs end-to-end on any
corpus you ingest.

## Your first Python session

```python
from bitig import (
    Corpus, Document,
    MFWExtractor, BurrowsDelta,
    PCAReducer, plot_scatter_2d,
)

corpus = Corpus(documents=[
    Document(id="d1", text=open("doc1.txt").read(), metadata={"author": "Alice"}),
    Document(id="d2", text=open("doc2.txt").read(), metadata={"author": "Alice"}),
    Document(id="d3", text=open("doc3.txt").read(), metadata={"author": "Bob"}),
    Document(id="d4", text=open("doc4.txt").read(), metadata={"author": "Bob"}),
    Document(id="q",  text=open("questioned.txt").read(), metadata={"author": "?"}),
])

# 1. Extract the most-frequent-word feature matrix.
fm = MFWExtractor(n=200, scale="zscore", lowercase=True).fit_transform(corpus)

# 2. Train Burrows Delta on the known docs and predict the questioned one.
import numpy as np
y = np.array(corpus.metadata_column("author"))
train_mask = y != "?"
clf = BurrowsDelta().fit_predict(fm)  # sklearn-compatible
```

## Sample data: the Federalist showcase

Two ready-to-run examples ship with the repo:

- [`examples/quickstart/`](https://github.com/fatihbozdag/bitig/tree/main/examples/quickstart)
  — a beginner-friendly walkthrough using 9 papers including the disputed No. 50.
- [`examples/federalist/`](https://github.com/fatihbozdag/bitig/tree/main/examples/federalist)
  — the full 85-paper analysis reproducing the Mosteller & Wallace (1964) result.

The quickstart produces this PCA plot on first run:

<p align="center">
  <img src="https://raw.githubusercontent.com/fatihbozdag/bitig/main/examples/quickstart/results/demo/pca/pca.png" alt="PCA of Hamilton vs Madison" style="max-width: 82%;">
</p>

## Next

- Learn the shared mental model in [Concepts](concepts/index.md).
- Jump into the [Forensic toolkit](forensic/index.md) for verification, LR output, and
  PAN-style evaluation.
- Reproduce Mosteller & Wallace in the [Federalist tutorial](tutorials/federalist.md).
