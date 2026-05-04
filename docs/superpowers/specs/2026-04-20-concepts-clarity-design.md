# Concepts clarity — design (2026-04-20)

## Problem

Users landing on the bitig docs cannot tell (a) what the overall system is for, (b) which
method to pick for which task, or (c) what each forensic metric means in practice. The
current `concepts/methods.md`, `concepts/features.md`, and `forensic/evaluation.md` pages
are catalogs — they list *what exists* without guiding the reader to *when to use it* or
*what to expect*. The forensic method pages (`verification.md`, `calibration.md`,
`topic-invariance.md`, `reporting.md`) have the same gap at a deeper level.

## Goal

Rewrite the concepts and forensic prose so both first-time researchers and experienced
stylometrists can navigate the package confidently — first-time readers learn which tool
to reach for, experienced readers keep the full technical detail. Ship a new top-level
decision guide, then expand each method and metric entry in place with a plain-language
gloss that sits above the existing technical content.

## Scope

- **New page:** `concepts/choosing.md` — task-oriented decision table, linked from
  `concepts/index.md` as the "start here if you don't know which method to use" entry.
- **Rewritten in place (8 pages):**
  - `concepts/index.md` — add a "What bitig is for" frame up top.
  - `concepts/methods.md` — every method entry gets a gloss block.
  - `concepts/features.md` — every extractor entry gets a gloss block.
  - `forensic/evaluation.md` — every metric gets a "use for" line.
  - `forensic/verification.md` — gloss General Impostors + Unmasking at method level.
  - `forensic/calibration.md` — gloss Platt + isotonic at method level; gloss what
    calibration does at the top of the page.
  - `forensic/topic-invariance.md` — gloss Sapkota categories + Stamatatos distortion.
  - `forensic/reporting.md` — gloss the ENFSI verbal scale at the top.
- **Turkish mirror:** every EN change is mirrored to its `.tr.md` sibling in the same
  commit, using the pinned glossary at `docs/site/_translations/tr-glossary.md`. The
  `fallback_to_default: true` i18n setting means the site never breaks mid-rewrite if one
  side lands before the other, but the discipline is "one commit changes both languages."

## Out of scope

- `concepts/corpus.md`, `concepts/results.md`, `concepts/languages.md` — these are already
  concrete and self-contained; no evidence they confuse readers.
- `tutorials/*.md` — already task-oriented; the new choosing page links to them rather
  than duplicating.
- `reference/*.md` — reference is reference; CLI and schema pages don't teach pedagogy.
- API docstring prose in `src/bitig/`.

## Approach

### The gloss/detail split

Every method and metric entry on the eight rewritten pages gets a two-block structure:

```markdown
### <Method or metric name>

*Use when:* <one sentence on the reader goal this serves>.
*Don't use when:* <one sentence on the obvious misuse>.
*Expect:* <one sentence on the shape of the output>.

<Existing technical prose, formulas, references, code examples — preserved as-is.>
```

The gloss is ~3 italicised lines; the detail block is the existing content. This serves
both audiences: the first-time reader can scan glosses; the experienced reader skips
straight to detail.

Inline method mentions inside prose retain the current tone — no need to repeat the gloss
every time `Burrows Delta` appears. The gloss exists once, at the method's primary entry.

### The decision table (`concepts/choosing.md`)

Structure: one page with a short framing paragraph, the main decision table, and a second
table for forensic-specific tasks. Rows are user goals; columns are method / required data
/ headline metric / tutorial link.

**Main table — authorship attribution & comparison:**

| I want to… | Required data | Method | Headline metric | Tutorial |
|---|---|---|---|---|
| Attribute 1 questioned doc to N candidate authors | N authors × ~2k+ words known each; 1 questioned doc | `BurrowsDelta` (classic); `CosineDelta` (robust default) | nearest-author rank | [Federalist](../tutorials/federalist.md) |
| Cluster an unknown corpus by stylistic similarity | 20+ docs, labels optional | `PCAReducer` + `KMeansClusterer` or `HDBSCANClusterer` | silhouette, visual inspection | *(extend Federalist)* |
| Compare two pre-defined author groups | 10+ docs per group | `Zeta(mode="classic" or "eder")` | distinctiveness score per word | — |
| Classify docs by group with ML | 20+ docs per class | `SklearnClassify` + any feature extractor | CV accuracy / F1 | — |
| Reduce features for visualisation | any FeatureMatrix | `PCAReducer` / `UMAPReducer` / `TSNEReducer` / `MDSReducer` | visual inspection | — |
| Bayesian single-candidate verification | 1 candidate × ≥1k words; 1 questioned doc | `WallaceMosteller` | posterior odds | *(planned)* |

**Forensic table — one-case verification:**

| I want to… | Required data | Method | Headline metric | Tutorial |
|---|---|---|---|---|
| Verify "same author?" between two docs | 2 docs + an impostor pool (~100 docs) | `GeneralImpostors` | calibrated LR + C_llr | [PAN-CLEF](../tutorials/pan-clef.md) |
| Verify "same author?" with topic robustness | same + pooled across multiple topics | `Unmasking` | degradation curve + decision | [PAN-CLEF](../tutorials/pan-clef.md) |
| Minimise topic bias in verification | any corpus | `CategorizedCharNgramExtractor(categories=("prefix","suffix","punct"))` or `distort_corpus(mode="dv_ma")` | same as upstream verifier | — |
| Turn raw scores into evidential LR | any verifier output + labelled dev trials | `CalibratedScorer` + `compute_pan_report` | LR + C_llr + ECE | [PAN-CLEF](../tutorials/pan-clef.md) |
| Generate a court-ready report | Result + chain-of-custody fields | `build_forensic_report` | ENFSI verbal scale | — |

Each cell links to the relevant method's primary entry on its detail page, so clicking
"BurrowsDelta" drops the reader at `concepts/methods.md#burrows-delta` with the full gloss
+ detail.

**Framing paragraphs** above each table answer "what is this table for" in plain Turkish
and English.

### `concepts/index.md` rewrite

Add a new top section before "The pipeline":

```markdown
## What bitig is for

bitig answers three questions about who wrote a text:

- **Attribution** — which of a set of candidate authors most likely wrote this document?
- **Verification** — was this document written by *this specific* person?
- **Group comparison** — how does one author's style differ from another's or from a
  defined group?

It also serves a forensic layer on top: calibrated likelihood ratios, chain-of-custody
metadata, and evaluation metrics tuned for courtroom use.

Not sure which question you're asking? **[Start with the Choosing a method
guide](choosing.md).**
```

The existing "The pipeline", "Provenance, everywhere", and "Read next" sections follow
unchanged.

### `concepts/methods.md` rewrite

Every method entry gets the gloss/detail structure. Example transformation for Burrows
Delta:

**Before:**

```markdown
### Burrows Delta
`BurrowsDelta(metric="manhattan", ...)`

Classic z-score + Manhattan distance on top-n MFW. Published Burrows 2002.
```

**After:**

```markdown
### Burrows Delta
`BurrowsDelta(metric="manhattan", ...)`

*Use when:* you have 2+ candidate authors with ~2000+ words of known writing each and
want to rank which one most likely wrote a questioned doc.
*Don't use when:* you have only one candidate (use `GeneralImpostors` verification
instead), or documents shorter than ~500 words (signal gets noisy).
*Expect:* a distance score per candidate; lowest distance is the predicted author.

Classic z-score + Manhattan distance on top-n MFW. Published Burrows 2002.
```

Apply the same pattern to every entry under: Attribution — Delta variants, Contrast —
Zeta, Dimensionality reduction, Clustering, Consensus trees, Classification + CV,
Bayesian, Forensic methods.

### `concepts/features.md` rewrite

Replace the extractor table's single-sentence descriptions with gloss-enriched list
entries. Example:

**Before (table row):**

| `MFWExtractor(n=..., scale=..., lowercase=...)` | Corpus | top-n word relative frequencies (z-scored, L1, L2, or raw) |

**After (list entry):**

```markdown
#### `MFWExtractor(n, scale, lowercase)`

*Use when:* you want the go-to stylometric feature — relative frequencies of the
most-frequent words. Default choice for Delta-family attribution.
*Don't use when:* your corpus is very small (<200 unique tokens) or the question is
topic-invariant (MFW is topic-sensitive; see `CategorizedCharNgramExtractor`).
*Expect:* an `(n_docs, n)` float matrix; rows sum to ~1 with `scale="l1"`.

Returns top-n word relative frequencies (z-scored, L1, L2, or raw). Lowercase
controls whether tokens are case-folded before counting.
```

Apply to every extractor, including the forensic extractors.

### `forensic/evaluation.md` metrics rewrite

The metrics table gains a "Use for" column. Each metric additionally gets a gloss
paragraph before the existing `::: bitig.forensic.metrics.<fn>` autodoc block.

New table:

| Metric | Measures | Use for | Range | Reference |
|---|---|---|---|---|
| `auc` | Ranking quality | **Choosing between systems.** Higher AUC → the system ranks same-author pairs above different-author pairs more reliably. | 0.5 (random) – 1.0 (perfect) | — |
| `c_at_1` | Accuracy with abstention credit | **Operational decisions** where "don't know" is safer than a wrong answer. | 0 – 1 | Peñas & Rodrigo 2011 |
| `f05u` | Precision-weighted F with non-answer penalty | **PAN-style evaluation.** Penalises over-confident wrong answers. | 0 – 1 | Bevendorff et al. PAN 2022 |
| `brier` | Posterior calibration | **Probabilistic output quality.** Lower = better-calibrated probabilities. | 0 (perfect) – 1 (worst) | Brier 1950 |
| `ece` | Expected calibration error | **Is `predict_proba` honest?** Bins predictions by confidence and compares claimed vs. actual accuracy. | 0 (perfect) – 1 | — |
| `cllr` | Log-likelihood-ratio cost | **Forensic LR quality.** The strict proper scoring rule for evidential output. Lower = better. | 0 (perfect) – ∞ | Brümmer & du Preez 2006 |
| `tippett` | LR distribution plot | **Sanity-check calibration visually.** Cumulative target vs. non-target LR curves should separate. | — | — |

### `forensic/verification.md`, `forensic/calibration.md`, `forensic/topic-invariance.md`, `forensic/reporting.md`

Each page gains:

- A one-paragraph "What this page is for" intro in plain language at the top, under the
  `#` title.
- A gloss block before each method / technique / table within the page.

Existing technical content, code examples, and references remain intact.

### Turkish mirror

Every EN edit has a corresponding TR edit in the same commit. Glossary terms
(`stilometri`, `yazar tespiti`, `yazar doğrulama`, `kalibrasyon`, `olabilirlik oranı`,
`delil zinciri`, etc.) apply. New patterns introduced by the gloss/detail split:

- *Use when:* → *Şu durumda kullanın:*
- *Don't use when:* → *Şu durumda kullanmayın:*
- *Expect:* → *Beklenen sonuç:*

These patterns go into the glossary after this PR lands so future translators stay
consistent.

## PR strategy

1. **Merge PR #20 first** (`feat/multilingual-docs`). It's verified; its scope is clean;
   further changes would bloat review.
2. **New branch `docs/concepts-clarity`** off the updated `main`.
3. **Commits grouped by page.** Each commit updates one EN page and its TR sibling
   together. Roughly 9 commits (1 per page, plus 1 for the new choosing page).
4. **Single PR** for the entire clarity pass — reviewers can evaluate one page at a time
   via the commit-by-commit diff.

## Success criteria

- A researcher new to stylometry can open `concepts/index.md`, follow the link to
  `concepts/choosing.md`, and identify the right method for their task in under a
  minute.
- An experienced reader can scan `concepts/methods.md` and reach the detail block for any
  method in under one page-down.
- Every method and metric on the 8 rewritten pages has a three-line gloss at its primary
  entry.
- `mkdocs build --strict` passes.
- Every EN change has a TR mirror with consistent glossary terminology.
- No existing technical content is removed; all existing references, formulas, and code
  examples are preserved.
