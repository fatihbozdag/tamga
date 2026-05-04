# Corpus

A `Corpus` is a list of `Document`s plus (implicitly) their metadata. Both are dataclasses
defined in `bitig.corpus`.

## Building a Corpus

### From the filesystem

```bash
bitig ingest corpus/ --metadata corpus/metadata.tsv
```

`metadata.tsv` is a tab-separated file where the first column is `filename` (matched
against the basenames in `corpus/`) and every other column becomes `Document.metadata`:

```tsv
filename	author	year	genre
fed_01.txt	Hamilton	1787	political essay
fed_10.txt	Madison	1787	political essay
fed_50.txt	Unknown	1788	political essay
```

### In Python

```python
from bitig.io import load_corpus
corpus = load_corpus("corpus/", metadata="corpus/metadata.tsv", strict=True)
```

- `strict=True` (default) raises if any document is missing a metadata row.
- `strict=False` allows partial coverage — useful when a new unlabeled document arrives.

### Programmatically

```python
from bitig.corpus import Corpus, Document

corpus = Corpus(documents=[
    Document(id="q", text=q_text, metadata={"role": "questioned"}),
    Document(id="k1", text=k1_text, metadata={"author": "Alice"}),
    Document(id="k2", text=k2_text, metadata={"author": "Alice"}),
])
```

## Filtering and grouping

`Corpus.filter(**query)` returns a new Corpus with only documents matching every
key-value pair:

```python
hamiltonian = corpus.filter(author="Hamilton")
train_only = corpus.filter(role="train")
```

A value can be a list to match any:

```python
two_authors = corpus.filter(author=["Hamilton", "Madison"])
```

`Corpus.groupby(field)` returns a dict of sub-corpora:

```python
grouped = corpus.groupby("author")
# {"Hamilton": Corpus(...), "Madison": Corpus(...), "Jay": Corpus(...)}
```

## Hashing

`Corpus.hash()` produces a stable SHA-256 derived from the sorted SHA-256 hashes of each
document's text plus sorted metadata entries. This hash ends up on every `Provenance`
record, so two studies with the same corpus share a hash regardless of filesystem path,
run time, or document ordering in the input directory.

!!! note "Order sensitivity"
    `Corpus.hash()` is order-invariant by design (same texts + metadata = same hash). If
    you need an order-sensitive hash (e.g., to detect row-reorderings that affect feature
    matrix layout), hash `[d.id for d in corpus.documents]` separately.

## Next

- [Features](features.md) — turn the corpus into a numeric matrix.
