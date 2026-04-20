# Topic-invariant features

*Use when:* your questioned and known documents might be on different topics — you
need features that capture style without leaking topic.
*Don't use when:* topic is part of the question (for example, a plagiarism check
where the two documents *should* share content). Use regular features then.
*Expect:* feature extractors that discard most content-word signal while preserving
function-word, morphology, and punctuation patterns.

Two techniques live under `tamga.forensic`: Sapkota char-n-gram *categorisation* and
Stamatatos *distortion*. Both compose with any downstream verifier.

Cross-topic is the most common failure mode of classical stylometry on real forensic
data. A suspect's threat letter and personal email are typically on different topics but
presumably the same author; unfiltered character-n-gram and word-n-gram features
collapse into topic detection in that setting.

tamga ships two complementary tools.

## Sapkota character n-gram categories

*Use when:* you want char-n-gram features for verification but need to strip
topic-sensitive whole-word n-grams — keeping only affixes, punctuation-adjacent, and
space-adjacent categories.
*Don't use when:* your corpus is so small that further filtering collapses the
feature space below ~500 dimensions.
*Expect:* a sparse count matrix with only the chosen categories; default
`("prefix","suffix","punct")` is the affix-only recipe that generalises best across
topics.

`CategorizedCharNgramExtractor` classifies each character n-gram **occurrence** (not just
the string) by its position in the source text. Feature columns are named
`<ngram>|<category>`, so `the|whole_word` and `the|prefix` are separate channels —
explicit and auditable.

Seven categories:

| Category | Description |
|---|---|
| `prefix` | word-start + char-internal (e.g., "the" in "there") |
| `suffix` | char-internal + word-end (e.g., "ing" in "running") |
| `whole_word` | exactly one word, boundaries at both ends |
| `mid_word` | entirely internal to a single word |
| `multi_word` | spans whitespace between two words |
| `punct` | contains any punctuation character |
| `space` | contains whitespace but not enough for multi_word |

Sapkota et al. (2015) showed that selecting only **affix (prefix + suffix) + punct**
dramatically improves cross-topic attribution — the forensic default.

```python
from tamga.forensic import CategorizedCharNgramExtractor

extractor = CategorizedCharNgramExtractor(
    n=3,
    categories=("prefix", "suffix", "punct"),  # topic-invariant subset
    scale="zscore",
    lowercase=True,
)
fm = extractor.fit_transform(corpus)
```

## Stamatatos distortion

*Use when:* you want aggressive topic removal via content-word masking — replaces
content words with placeholders while preserving function words, morphology, and
punctuation.
*Don't use when:* you need any content-word signal downstream (e.g., Zeta on
distinctive vocabulary).
*Expect:* a new `Corpus` object you pass to any existing extractor. Modes: `"dv_ma"`
masks *all* content words, `"dv_sa"` masks selectively by POS.

`distort_corpus` pre-processes documents to mask **content** while preserving **style**:
function words, punctuation, digits, and whitespace remain verbatim; content-word
characters are replaced.

### Two modes

**DV-MA** (*Distortion View — Multiple Asterisks*): each content-word character → `*`.
Length-preserving — morphological habits (typical word lengths) remain visible.

**DV-SA** (*Distortion View — Single Asterisk*): each content word → single `*`.
Aggressive; only function-word and punctuation pattern survives.

```python
from tamga.forensic import distort_corpus
from tamga import MFWExtractor

distorted = distort_corpus(corpus, mode="dv_ma")

# Downstream extractors see the distorted text — topic signal is masked out.
fm = MFWExtractor(n=200, scale="zscore").fit_transform(distorted)
```

### Contractions

Both `_TOKEN_RE` and the bundled function-word list preserve common English contractions
(`don't`, `it's`, `we'll`, `they've`, …) verbatim. `o'clock` and other apostrophised
content words are masked as a single contiguous string (e.g., `*******`) rather than
split into fragments.

### Custom function-word list

```python
distorted = distort_corpus(
    corpus,
    mode="dv_ma",
    function_words={"the", "a", "of", "to", "and"},   # minimal stoplist
)
```

Pass `frozenset()` to treat every word as content (DV-MA will produce an all-`*` text).

## Combining the two

Sapkota categories + Stamatatos distortion compose cleanly:

```python
distorted = distort_corpus(corpus, mode="dv_ma")
extractor = CategorizedCharNgramExtractor(
    n=3, categories=("prefix", "suffix", "punct"), lowercase=True
)
fm = extractor.fit_transform(distorted)
```

This produces a feature set that is **doubly** topic-invariant — affix-and-punctuation
n-grams extracted from content-masked text — and routinely outperforms unfiltered
character n-grams on cross-genre PAN tasks.

## Reference

::: tamga.forensic.char_ngrams.CategorizedCharNgramExtractor
    options:
      show_root_full_path: false

::: tamga.forensic.char_ngrams.classify_ngram

::: tamga.forensic.distortion.distort_corpus

::: tamga.forensic.distortion.distort_text
