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

Import from `bitig`:

| Extractor | Input | Output |
|---|---|---|
| `MFWExtractor(n=..., scale=..., lowercase=...)` | Corpus | top-n word relative frequencies (z-scored, L1, L2, or raw) |
| `CharNgramExtractor(n=..., include_boundaries=...)` | Corpus | character n-gram counts (delegates to sklearn CountVectorizer) |
| `WordNgramExtractor(n=..., lowercase=...)` | Corpus | word n-gram counts |
| `PosNgramExtractor(n=..., tagset=...)` | Corpus | spaCy POS n-grams |
| `DependencyBigramExtractor()` | Corpus | (head_lemma, dep, child_lemma) triples |
| `FunctionWordExtractor(language=...)` | Corpus | per-language function-word counts |
| `PunctuationExtractor()` | Corpus | ASCII punctuation frequencies |
| `ReadabilityExtractor()` | Corpus | readability indices (English default: Flesch, FK-grade, Gunning Fog) |
| `SentenceLengthExtractor()` | Corpus | mean, SD, skew of per-sentence tokens |
| `LexicalDiversityExtractor()` | Corpus | lexical-diversity indices (default: TTR, Yule's K) |
| `SentenceEmbeddingExtractor(model=...)` | Corpus | sentence-transformers pooled embedding (extra: `bitig[embeddings]`) |
| `ContextualEmbeddingExtractor(model=..., pool=...)` | Corpus | HF transformer hidden-state vectors (extra: `bitig[embeddings]`) |

### Extractor detail

Each extractor above is a callable object; `fit_transform(corpus)` returns a
`FeatureMatrix`.

#### MFWExtractor
`MFWExtractor(n=1000, scale="zscore", lowercase=False)`  *(defaults; e.g. `n=200` is a common override)*

*Use when:* you want the canonical stylometric feature — relative frequencies of
the most-frequent words. Default choice for Delta-family attribution.
*Don't use when:* your corpus is very small (<200 unique tokens), or the question is
topic-invariant (MFW is topic-sensitive; see `CategorizedCharNgramExtractor`).
*Expect:* an `(n_docs, n)` float matrix; rows sum to ~1 under `scale="l1"`,
zero-centred unit-variance under `scale="zscore"`.

#### CharNgramExtractor
`CharNgramExtractor(n=3, include_boundaries=False)`  *(`n` may also be a `(min_n, max_n)` tuple)*

*Use when:* you want features that capture sub-word style (prefixes, suffixes,
punctuation adjacency) and that cope with OOV words or misspellings.
*Don't use when:* your languages mix scripts (n-grams across scripts produce noise),
or you specifically need word-level semantic sensitivity.
*Expect:* sparse count matrix delegated to sklearn's `CountVectorizer`.

#### WordNgramExtractor
`WordNgramExtractor(n=1, lowercase=False)`

*Use when:* unigrams (MFW equivalent) or short bigram phrases are what you need and
you don't want z-scoring. Bigrams useful for detecting fixed expressions.
*Don't use when:* n ≥ 3 in small corpora — sparsity dominates. Use `MFWExtractor`
for unigrams unless you need raw counts.
*Expect:* sparse count matrix; vocabulary grows fast with n.

#### PosNgramExtractor
`PosNgramExtractor(n=2, tagset="coarse")`

*Use when:* you want syntactic-style features (sequences of part-of-speech tags) —
insensitive to content words, sensitive to register and syntactic register.
*Don't use when:* your spaCy pipeline doesn't include a tagger (most `_trf` models
do), or your corpus is very small per-doc.
*Expect:* sparse count matrix over POS n-grams. `tagset="coarse"` (the default) uses UD
coarse tags (fewer dimensions, more robust); `tagset="fine"` uses the fine-grained tags.

#### DependencyBigramExtractor
`DependencyBigramExtractor()`

*Use when:* you want syntax-sensitive style features — specifically, the
(head-lemma, dependency-relation, child-lemma) triples parsed by spaCy.
*Don't use when:* your parser is a bottleneck; dependency parsing is the slowest
step in the spaCy pipeline and you may be able to substitute POS n-grams.
*Expect:* sparse count matrix over dependency triples.

#### FunctionWordExtractor
`FunctionWordExtractor(language=None, wordlist=None, scale="none")`  *(keyword-only)*

*Use when:* you want the short, topic-insensitive function-word list (the classic
anti-topic signal for stylometry) for the document's language.
*Don't use when:* your corpus mixes languages without a per-doc language tag — the
per-language word list won't apply.
*Expect:* `(n_docs, |wordlist|)` matrix of **raw counts** by default (`scale="none"`); pass
`scale="l1"` for relative frequencies. The list is resolved from the corpus language (or the
`language=`/`wordlist=` you pass) — see [Languages](languages.md).

#### PunctuationExtractor
`PunctuationExtractor()`

*Use when:* you want pure-style features that are nearly topic-invariant —
punctuation usage is remarkably author-specific and corpus-robust.
*Don't use when:* your source text has been normalised or stripped of punctuation
(e.g., OCR output without correction).
*Expect:* `(n_docs, 32)` matrix of ASCII punctuation frequencies (one column per
`string.punctuation` symbol).

#### ReadabilityExtractor
`ReadabilityExtractor(indices=None, language=None)`

*Use when:* you want readability-as-style — Flesch, FK-grade, Gunning Fog, etc. —
as a lightweight feature set to combine with MFW.
*Don't use when:* readability itself is the question (for that, read the metric
directly; don't bundle into a Delta). For non-English, the per-language native
formulas are selected automatically from the corpus language — see
[Languages](languages.md).
*Expect:* by default, `(n_docs, 3)` for English — Flesch, Flesch-Kincaid, Gunning Fog.
Three more English indices (Coleman-Liau, ARI, SMOG) are available via
`ReadabilityExtractor(indices=["flesch", "coleman_liau", "ari", "smog", ...])`.

#### SentenceLengthExtractor
`SentenceLengthExtractor()`

*Use when:* you want the sentence-rhythm signature — mean, SD, and skew of
per-sentence token counts. Small but strong stylistic signal.
*Don't use when:* your text has aggressive sentence-boundary errors (e.g., ALL
CAPS legal text breaks most sentencizers).
*Expect:* `(n_docs, 3)` matrix: `[mean, sd, skew]`.

#### LexicalDiversityExtractor
`LexicalDiversityExtractor(indices=("ttr", "yules_k"))`

*Use when:* you want vocabulary-richness features. Eight indices are available — TTR,
MATTR, MTLD, HD-D, Yule's K/I, Herdan's C, Simpson's D — so you can compare
sensitivities.
*Don't use when:* your documents are very short (<200 tokens); most indices become
unstable.
*Expect:* by default, `(n_docs, 2)` — TTR and Yule's K. Select the full set with
`LexicalDiversityExtractor(indices=["ttr", "mattr", "mtld", "hdd", "yules_k", "yules_i", "herdans_c", "simpsons_d"])`.

#### SentenceEmbeddingExtractor
`SentenceEmbeddingExtractor(model=None, language=None)`  *(model `None` → per-language default, e.g. `all-mpnet-base-v2` for English)*

*Use when:* you want a modern neural-embedding feature set — pooled
sentence-transformer output per document. Strong in classification + clustering;
fast enough for moderate corpora.
*Don't use when:* your hardware lacks GPU / MPS and your corpus is large (CPU
inference is slow), or when interpretability matters (these vectors are opaque).
*Expect:* `(n_docs, embedding_dim)` dense matrix. Requires `bitig[embeddings]`.

#### ContextualEmbeddingExtractor
`ContextualEmbeddingExtractor(model=None, pool="mean", layer=-1)`  *(model `None` → per-language default, e.g. `bert-base-uncased` for English, `dbmdz/bert-base-turkish-cased` for Turkish)*

*Use when:* you want HuggingFace-model hidden states aggregated per document —
language-specific embeddings (e.g., `dbmdz/bert-base-turkish-cased` for Turkish)
with configurable pooling.
*Don't use when:* you don't need a specific model's representation — use
`SentenceEmbeddingExtractor` for a lighter, faster default.
*Expect:* `(n_docs, hidden_dim)` dense matrix. Requires `bitig[embeddings]`.

## Composing features

Two ways to build a multi-feature matrix:

### Python

```python
from bitig import MFWExtractor, PunctuationExtractor

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

Two topic-invariant extractors live under `bitig.forensic`:

#### CategorizedCharNgramExtractor
`CategorizedCharNgramExtractor(n=3, categories=None)`  *(`categories=None` → all 7 Sapkota categories; pass a subset like `("prefix","suffix")` to keep only affixes)*

*Use when:* you want topic-invariant character-level features for forensic
verification — n-grams classified by position in the word so you can keep only
the style-carrying categories (affixes, punctuation) and drop the topic-sensitive
whole-word category.
*Don't use when:* topic robustness isn't the goal — a plain `CharNgramExtractor`
is faster and carries more signal per dimension.
*Expect:* sparse count matrix restricted to the chosen n-gram categories.

Sapkota et al. 2015; restricting to the affix categories (e.g. `("prefix","suffix")`) is
the recipe that generalises best across topics.

#### distort_corpus
`distort_corpus(corpus, mode="dv_ma")`

*Use when:* you want Stamatatos (2013) topic masking — replaces content words with
placeholders while keeping function words and punctuation. Pair with any
extractor for a topic-invariant pipeline.
*Don't use when:* your analysis needs content-word signal (e.g., Zeta looking for
distinctive vocabulary).
*Expect:* a new Corpus object you feed to any existing extractor. Modes: `"dv_ma"`
replaces each character of a content word with `*` (length-preserving), `"dv_sa"`
collapses each content word to a single `*` (length not preserved). Content-vs-function
is decided by a function-word lookup, not POS tagging.

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
