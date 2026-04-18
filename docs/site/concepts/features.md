# Features

Every feature extractor returns a `FeatureMatrix` — the shared numeric envelope that
methods consume.

## The FeatureMatrix

```python
@dataclass
class FeatureMatrix:
    X: np.ndarray            # (n_docs, n_features)
    document_ids: list[str]
    feature_names: list[str]
    feature_type: str
    extractor_config: dict[str, Any]
    provenance_hash: str
```

Key properties:

- `fm.n_features`, `len(fm)` for `n_docs`
- `fm.as_dataframe()` — pandas `DataFrame` indexed by `document_ids`
- `fm.concat(other)` — column-concatenate two matrices with identical row ids

## Available extractors

Import from `tamga`:

| Extractor | Input | Output |
|---|---|---|
| `MFWExtractor(n=..., scale=..., lowercase=...)` | Corpus | top-n word relative frequencies (z-scored, L1, L2, or raw) |
| `CharNgramExtractor(n=..., include_boundaries=...)` | Corpus | character n-gram counts (delegates to sklearn CountVectorizer) |
| `WordNgramExtractor(n=..., lowercase=...)` | Corpus | word n-gram counts |
| `PosNgramExtractor(n=..., coarse=...)` | Corpus | spaCy POS n-grams |
| `DependencyBigramExtractor()` | Corpus | (head_lemma, dep, child_lemma) triples |
| `FunctionWordExtractor(wordlist=...)` | Corpus | bundled English function-word frequencies |
| `PunctuationExtractor()` | Corpus | ASCII punctuation frequencies |
| `ReadabilityExtractor()` | Corpus | six readability indices (Flesch, FK-grade, Gunning Fog, Coleman-Liau, ARI, SMOG) |
| `SentenceLengthExtractor()` | Corpus | mean, SD, skew of per-sentence tokens |
| `LexicalDiversityExtractor()` | Corpus | TTR, MATTR, MTLD, HD-D, Yule's K/I, Herdan's C, Simpson's D |
| `SentenceEmbeddingExtractor(model=...)` | Corpus | sentence-transformers pooled embedding (extra: `tamga[embeddings]`) |
| `ContextualEmbeddingExtractor(model=..., pooling=...)` | Corpus | HF transformer hidden-state vectors (extra: `tamga[embeddings]`) |

## Composing features

Two ways to build a multi-feature matrix:

### Python

```python
from tamga import MFWExtractor, PunctuationExtractor

mfw = MFWExtractor(n=200, scale="zscore").fit_transform(corpus)
punct = PunctuationExtractor().fit_transform(corpus)
combined = mfw.concat(punct)  # (n_docs, n_mfw + n_punct)
```

### study.yaml

```yaml
features:
  - id: mfw
    type: mfw
    n: 200
    scale: zscore
  - id: punct
    type: punctuation
```

Methods can reference feature ids; the runner builds each matrix once and reuses it.

## Forensic feature extractors

Two topic-invariant extractors live under `tamga.forensic`:

- `CategorizedCharNgramExtractor(n=..., categories=...)` — classifies each n-gram
  occurrence (prefix / suffix / whole_word / mid_word / multi_word / punct / space) per
  Sapkota et al. 2015. `categories=("prefix", "suffix", "punct")` produces the affix-only
  feature set that generalises best across topics.
- `distort_corpus(corpus, mode="dv_ma"|"dv_sa")` — Stamatatos 2013 content masking.
  Returns a new Corpus; feed it into any existing extractor.

See [Topic-invariant features](../forensic/topic-invariance.md).

## Scaling

Most extractors accept `scale ∈ {"none", "zscore", "l1", "l2"}`:

- `none` — raw counts. Use for Bayesian Wallace–Mosteller.
- `l1` — relative frequencies (row sums to 1). Use for Zeta-like contrast methods.
- `l2` — unit-norm rows. Use for cosine-based distances.
- `zscore` — per-column z-score on training means / SDs (Stylo convention). **Required for
  Burrows Delta.**

The z-score mean / SD are learned at `fit` time and applied at `transform` — so scores on
unseen documents use the training distribution.

## Next

- [Methods](methods.md) — take the FeatureMatrix and produce a Result.
