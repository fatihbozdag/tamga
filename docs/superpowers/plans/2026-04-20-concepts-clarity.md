# Concepts Clarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite 8 existing docs pages and add 1 new "Choosing a method" page so readers can tell *what bitig is for* and *which method/metric to use when*, while experienced readers still get full technical detail.

**Architecture:** Each method and metric entry gets a three-line gloss block (*Use when* / *Don't use when* / *Expect*) placed above its existing technical content. A new `concepts/choosing.md` surfaces a task-oriented decision table routing readers to the right method. Turkish mirrors are updated in the same commit per page, using the pinned glossary at `docs/site/_translations/tr-glossary.md`.

**Tech Stack:** MkDocs Material 9.5+ with `mkdocs-static-i18n` 1.3 (already configured), Lunr search (EN + TR). Build: `.venv/bin/mkdocs build --strict`. No code or runtime changes.

**Source spec:** `docs/superpowers/specs/2026-04-20-concepts-clarity-design.md` (commit `cbf6f0d` on branch `docs/concepts-clarity`).

**Branch:** `docs/concepts-clarity` (already checked out, based on `origin/main` which includes the merged multilingual site).

---

## Shared Reference: Gloss Block Pattern

Every method and metric entry on the eight rewritten pages receives this exact structure:

```markdown
### <Method or metric name>
`<constructor or identifier if applicable>`

*Use when:* <one-sentence reader-goal description>.
*Don't use when:* <one-sentence obvious-misuse description>.
*Expect:* <one-sentence output-shape description>.

<Existing technical prose, formulas, code blocks, references — preserved verbatim.>
```

The gloss is three italicised lines separated by plain text. Do not bold. Do not add extra prose between the gloss and the detail block. When a page already has an H3 for a method (e.g., `verification.md`'s `## General Impostors`), the gloss block sits immediately under that heading.

When a page uses a table to list sibling methods (e.g., `methods.md`'s Delta-variants table), the task adds per-method H3 subsections below the table. The table stays as a quick-reference summary.

---

## Turkish Gloss Pattern

These three phrases are the Turkish renderings of the EN gloss markers and must be consistent across all pages:

| English | Turkish |
|---|---|
| *Use when:* | *Şu durumda kullanın:* |
| *Don't use when:* | *Şu durumda kullanmayın:* |
| *Expect:* | *Beklenen sonuç:* |

Turkish gloss lines translate the EN bullet text directly. Existing glossary terms (`stilometri`, `derlem`, `yazar tespiti`, `yazar doğrulama`, `kalibrasyon`, `olabilirlik oranı`, `delil zinciri`, `öznitelik`, `işlev sözcüğü`, `okunabilirlik`, `kümeleme`, `sınıflandırma`, `sınıflandırıcı`, `gömme`) apply.

---

### Task 1: Add Turkish gloss-pattern terms to the glossary

**Files:**
- Modify: `docs/site/_translations/tr-glossary.md`

- [ ] **Step 1: Extend the glossary table**

Open `docs/site/_translations/tr-glossary.md`. Under the existing glossary table (before the "Brand / proper nouns" paragraph), add a new section:

```markdown
## Gloss pattern (for concepts / forensic pages)

| English | Turkish |
|---|---|
| *Use when:* | *Şu durumda kullanın:* |
| *Don't use when:* | *Şu durumda kullanmayın:* |
| *Expect:* | *Beklenen sonuç:* |

Apply these exact renderings in every gloss block across `concepts/` and `forensic/`
pages. Do not paraphrase the marker phrases.
```

- [ ] **Step 2: Verify the build**

Run: `.venv/bin/mkdocs build --strict`

Expected: PASS (glossary page is excluded from nav via `not_in_nav`; it still builds).

- [ ] **Step 3: Commit**

```bash
git add docs/site/_translations/tr-glossary.md
git commit -m "docs(i18n): pin Turkish gloss-pattern phrases for concepts pages"
```

---

### Task 2: Create `concepts/choosing.md` (EN + TR)

**Files:**
- Create: `docs/site/concepts/choosing.md`
- Create: `docs/site/concepts/choosing.tr.md`
- Modify: `mkdocs.yml` (add to nav)

- [ ] **Step 1: Write the English page**

Create `docs/site/concepts/choosing.md` with the following verbatim content:

````markdown
# Choosing a method

Not sure which bitig method fits your question? This page answers *"I want to do X —
what should I reach for?"* for the most common cases. The method names link to their
primary entries in [Methods](methods.md) and [Forensic toolkit](../forensic/index.md)
for the full detail.

## Attribution, comparison, exploration

| I want to… | Required data | Method | Headline metric | Tutorial |
|---|---|---|---|---|
| Attribute 1 questioned doc to N candidate authors | N authors × ~2k+ words known each; 1 questioned doc | [`CosineDelta`](methods.md#cosinedelta) (robust default) or [`BurrowsDelta`](methods.md#burrowsdelta) (classic) | nearest-author rank | [Federalist](../tutorials/federalist.md) |
| Cluster an unknown corpus by stylistic similarity | 20+ docs, labels optional | [`PCAReducer`](methods.md#pcareducer) + [`KMeansCluster`](methods.md#kmeanscluster) or [`HDBSCANCluster`](methods.md#hdbscancluster) | silhouette, visual inspection | — |
| Compare two pre-defined author groups | 10+ docs per group | [`ZetaClassic`](methods.md#zetaclassic) or [`ZetaEder`](methods.md#zetaeder) | per-word distinctiveness score | — |
| Classify docs into groups with ML | 20+ docs per class | [`build_classifier`](methods.md#classification--cv) + [`cross_validate_bitig`](methods.md#classification--cv) | CV accuracy / F1 | — |
| Reduce features for visualisation | any `FeatureMatrix` | [`PCAReducer`](methods.md#pcareducer) / [`UMAPReducer`](methods.md#umapreducer) / [`TSNEReducer`](methods.md#tsnereducer) / [`MDSReducer`](methods.md#mdsreducer) | visual inspection | — |
| Bayesian single-candidate attribution | N candidates × ≥1k words; 1 questioned doc | [`BayesianAuthorshipAttributor`](methods.md#bayesianauthorshipattributor) | posterior probability per candidate | — |
| Bootstrap-consensus tree across MFW bands | 10+ docs, multiple MFW bands | [`BootstrapConsensus`](methods.md#bootstrapconsensus) | Newick tree with clade support | — |

## Forensic — one-case verification

| I want to… | Required data | Method | Headline metric | Tutorial |
|---|---|---|---|---|
| Verify "same author?" between 1 questioned doc and 1 candidate | 1 candidate's known writings + an impostor pool (~100 docs) | [`GeneralImpostors`](../forensic/verification.md#generalimpostors) | calibrated log-LR + `C_llr` | [PAN-CLEF](../tutorials/pan-clef.md) |
| Verify same-author with topic-robustness | Q + K long prose + impostor pool | [`Unmasking`](../forensic/verification.md#unmasking) | accuracy-drop curve | [PAN-CLEF](../tutorials/pan-clef.md) |
| Minimise topic bias in verification features | any corpus | [`CategorizedCharNgramExtractor`](features.md#categorizedcharngramextractor) with `categories=("prefix","suffix","punct")`, or [`distort_corpus(mode="dv_ma")`](features.md#distort_corpus) | same as upstream verifier | [PAN-CLEF](../tutorials/pan-clef.md) |
| Turn raw verifier scores into evidential LR | verifier outputs on labelled dev trials | [`CalibratedScorer`](../forensic/calibration.md#calibratedscorer) + [`compute_pan_report`](../forensic/evaluation.md#compute_pan_report) | log-LR, `C_llr`, `ECE` | [PAN-CLEF](../tutorials/pan-clef.md) |
| Generate a court-ready LR-framed report | `Result` with chain-of-custody fields | [`build_forensic_report`](../forensic/reporting.md) | ENFSI verbal scale | — |

## How to read this page

- **"Required data"** is a minimum — more is always better.
- **"Headline metric"** is the output you should quote in write-ups, not the only output
  the method produces.
- When two methods are listed for the same task, the first one is the recommended default
  and the second is a published alternative worth considering.

## Next

- [Methods](methods.md) — full catalogue with gloss + detail per method.
- [Features](features.md) — extractor catalogue with gloss + detail per extractor.
- [Forensic toolkit](../forensic/index.md) — calibration, evaluation, reporting.
````

- [ ] **Step 2: Write the Turkish page**

Create `docs/site/concepts/choosing.tr.md`. Structure mirrors the English exactly (same tables, same columns, same method-name link targets in English — the i18n plugin resolves the correct language variant). Translate only prose cells and headings. Keep method identifiers, tutorial names, and file links verbatim.

Use this exact content:

````markdown
# Yöntem seçimi

Hangi bitig yönteminin sorunuza uyacağından emin değil misiniz? Bu sayfa en yaygın
durumlar için *"X'i yapmak istiyorum — hangisini kullanmalıyım?"* sorusunu yanıtlar.
Yöntem adları tam ayrıntı için [Yöntemler](methods.md) ve
[Adli araç seti](../forensic/index.md) sayfalarındaki birincil girdilere bağlanır.

## Yazar tespiti, karşılaştırma, keşif

| Amaç | Gerekli veri | Yöntem | Öne çıkan ölçüt | Öğretici |
|---|---|---|---|---|
| 1 sorgulanan belgeyi N aday yazar arasından tespit etmek | Her aday için ~2k+ sözcük bilinen metin; 1 sorgulanan belge | [`CosineDelta`](methods.md#cosinedelta) (sağlam varsayılan) veya [`BurrowsDelta`](methods.md#burrowsdelta) (klasik) | en yakın yazar sıralaması | [Federalist](../tutorials/federalist.md) |
| Bilinmeyen bir derlemi biçemsel benzerlikle kümelemek | 20+ belge, etiketler isteğe bağlı | [`PCAReducer`](methods.md#pcareducer) + [`KMeansCluster`](methods.md#kmeanscluster) veya [`HDBSCANCluster`](methods.md#hdbscancluster) | silhouette, görsel inceleme | — |
| Önceden tanımlı iki yazar grubunu karşılaştırmak | Her grup için 10+ belge | [`ZetaClassic`](methods.md#zetaclassic) veya [`ZetaEder`](methods.md#zetaeder) | sözcük başına ayırt edicilik skoru | — |
| Belgeleri makine öğrenmesiyle sınıflandırmak | Her sınıf için 20+ belge | [`build_classifier`](methods.md#classification--cv) + [`cross_validate_bitig`](methods.md#classification--cv) | CV doğruluğu / F1 | — |
| Görselleştirme için öznitelikleri boyut indirgemek | herhangi bir `FeatureMatrix` | [`PCAReducer`](methods.md#pcareducer) / [`UMAPReducer`](methods.md#umapreducer) / [`TSNEReducer`](methods.md#tsnereducer) / [`MDSReducer`](methods.md#mdsreducer) | görsel inceleme | — |
| Bayes yaklaşımıyla tek-aday yazar tespiti | N aday × ≥1k sözcük; 1 sorgulanan belge | [`BayesianAuthorshipAttributor`](methods.md#bayesianauthorshipattributor) | aday başına sonsal olasılık | — |
| MFW bantları üzerinde bootstrap konsensüs ağacı | 10+ belge, birden fazla MFW bandı | [`BootstrapConsensus`](methods.md#bootstrapconsensus) | klad desteği ile Newick ağacı | — |

## Adli — tek-olgu doğrulama

| Amaç | Gerekli veri | Yöntem | Öne çıkan ölçüt | Öğretici |
|---|---|---|---|---|
| 1 sorgulanan belge ile 1 aday arasında "aynı yazar mı?" sorusunu doğrulamak | 1 adayın bilinen yazıları + bir sahte-aday havuzu (~100 belge) | [`GeneralImpostors`](../forensic/verification.md#generalimpostors) | kalibre edilmiş log-LR + `C_llr` | [PAN-CLEF](../tutorials/pan-clef.md) |
| Konudan bağımsız aynı-yazar doğrulaması | uzun düzyazı Q + K + sahte-aday havuzu | [`Unmasking`](../forensic/verification.md#unmasking) | doğruluk düşüş eğrisi | [PAN-CLEF](../tutorials/pan-clef.md) |
| Doğrulamada konu yanlılığını azaltmak | herhangi bir derlem | [`CategorizedCharNgramExtractor`](features.md#categorizedcharngramextractor), `categories=("prefix","suffix","punct")`; veya [`distort_corpus(mode="dv_ma")`](features.md#distort_corpus) | yukarı akışlı doğrulayıcıyla aynı | [PAN-CLEF](../tutorials/pan-clef.md) |
| Ham doğrulayıcı skorlarını kanıtsal olabilirlik oranına dönüştürmek | etiketli geliştirme denemelerinde doğrulayıcı çıktıları | [`CalibratedScorer`](../forensic/calibration.md#calibratedscorer) + [`compute_pan_report`](../forensic/evaluation.md#compute_pan_report) | log-LR, `C_llr`, `ECE` | [PAN-CLEF](../tutorials/pan-clef.md) |
| Mahkemeye uygun LR çerçeveli rapor üretmek | delil zinciri alanlarıyla birlikte `Result` | [`build_forensic_report`](../forensic/reporting.md) | ENFSI sözel ölçeği | — |

## Bu sayfa nasıl okunur

- **"Gerekli veri"** bir asgaridir — daha fazlası her zaman daha iyidir.
- **"Öne çıkan ölçüt"** yazımda alıntılamanız gereken çıktıdır, yöntemin ürettiği tek
  çıktı değildir.
- Aynı görev için iki yöntem listelendiğinde ilki önerilen varsayılandır; ikincisi
  değerlendirmeye değer yayımlanmış bir alternatiftir.

## Sonraki

- [Yöntemler](methods.md) — yöntem başına gloss + ayrıntı içeren tam katalog.
- [Öznitelikler](features.md) — öznitelik çıkarıcı başına gloss + ayrıntı içeren katalog.
- [Adli araç seti](../forensic/index.md) — kalibrasyon, değerlendirme, raporlama.
````

- [ ] **Step 3: Add the page to `mkdocs.yml` nav**

Edit `mkdocs.yml`. Find the `Concepts:` section under `nav:`. Insert the new page right after `concepts/index.md`:

```yaml
  - Concepts:
      - concepts/index.md
      - Choosing a method: concepts/choosing.md
      - Corpus: concepts/corpus.md
      - Features: concepts/features.md
      - Languages: concepts/languages.md
      - Methods: concepts/methods.md
      - Results & provenance: concepts/results.md
```

Also add the Turkish nav translation under the `tr` locale's `nav_translations:`:

```yaml
            Choosing a method: Yöntem seçimi
```

- [ ] **Step 4: Build strict**

Run: `.venv/bin/mkdocs build --strict`

Expected: PASS. The build log should show "Translated 23 navigation elements to 'tr'" (was 22; +1 for the new page).

- [ ] **Step 5: Commit**

```bash
git add docs/site/concepts/choosing.md docs/site/concepts/choosing.tr.md mkdocs.yml
git commit -m "docs(concepts): add \"Choosing a method\" decision-table page (EN + TR)"
```

---

### Task 3: Rewrite `concepts/index.md` with "What bitig is for" intro (EN + TR)

**Files:**
- Modify: `docs/site/concepts/index.md`
- Modify: `docs/site/concepts/index.tr.md`

- [ ] **Step 1: Insert the "What bitig is for" section in the English page**

Open `docs/site/concepts/index.md`. Immediately after the `# Concepts` heading (and before the existing `## The pipeline` heading), insert this new section verbatim:

```markdown
## What bitig is for

bitig answers three questions about who wrote a text:

- **Attribution** — which of a set of candidate authors most likely wrote this document?
- **Verification** — was this document written by *this specific* person?
- **Group comparison** — how does one author's style differ from another's, or from a
  defined group?

On top of those core questions it ships a forensic layer: calibrated likelihood ratios,
chain-of-custody metadata, and evaluation metrics tuned for courtroom use.

Not sure which question you're asking? **[Start with the Choosing a method
guide](choosing.md).**
```

The existing `## The pipeline`, `## Provenance, everywhere`, and `## Read next` sections remain unchanged below.

- [ ] **Step 2: Insert the equivalent section in the Turkish page**

Open `docs/site/concepts/index.tr.md`. Immediately after the `# Kavramlar` (or equivalent) heading and before the first existing H2, insert:

```markdown
## bitig ne işe yarar

bitig, bir metni kimin yazdığına dair üç soruyu yanıtlar:

- **Yazar tespiti** — bir aday yazar kümesinden hangisinin bu belgeyi yazma olasılığı
  en yüksektir?
- **Yazar doğrulama** — bu belgeyi *bu kişi* mi yazdı?
- **Grup karşılaştırması** — bir yazarın biçemi, bir başkasının biçeminden ya da
  tanımlanmış bir gruptan nasıl ayrılır?

Bu temel soruların üstünde bir adli katman yer alır: kalibre edilmiş olabilirlik oranları,
delil zinciri üstverisi ve mahkeme kullanımına göre ayarlanmış değerlendirme ölçütleri.

Hangi soruyu sorduğunuzdan emin değil misiniz? **[Yöntem seçimi rehberiyle
başlayın](choosing.md).**
```

- [ ] **Step 3: Build strict**

Run: `.venv/bin/mkdocs build --strict`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add docs/site/concepts/index.md docs/site/concepts/index.tr.md
git commit -m "docs(concepts): add \"What bitig is for\" intro to concepts landing (EN + TR)"
```

---

### Task 4: Gloss every method in `concepts/methods.md` (EN + TR)

**Files:**
- Modify: `docs/site/concepts/methods.md`
- Modify: `docs/site/concepts/methods.tr.md`

Current state of `concepts/methods.md`: H2 sections list methods in tables + brief prose. This task adds per-method H3 subsections with gloss blocks **after** each existing H2's summary content. The summary tables / prose stay.

- [ ] **Step 1: Insert per-method H3 subsections under "Attribution — Delta variants"**

Open `concepts/methods.md`. After the existing Delta-variants table and the `BurrowsDelta().fit(...)` code block, before the `## Contrast — Zeta` heading, insert these subsections verbatim:

```markdown
### BurrowsDelta
`BurrowsDelta()`

*Use when:* you have 2+ candidate authors with ~2000+ words of known writing each and
want to rank which one most likely wrote a questioned document.
*Don't use when:* you have only one candidate (use `GeneralImpostors` verification
instead), or when documents are shorter than ~500 words (signal gets noisy).
*Expect:* a distance score per candidate; lowest distance is the predicted author.

The classic Burrows (2002) method: z-score features, mean-absolute-difference (L1)
distance to each candidate's centroid. Good default for literary English.

### CosineDelta
`CosineDelta()`

*Use when:* you want a modern default for Delta-family attribution — cosine is robust
to document-length differences and less sensitive to outlier words than L1.
*Don't use when:* your corpus mixes wildly different genres without care; cosine is
less diagnostic when topic dominates style.
*Expect:* a distance score per candidate in `[0, 2]`; lowest distance wins.

Smith & Aldridge (2011). Standard choice in modern stylometry; often the best
single-method baseline before tuning.

### EderDelta / EderSimpleDelta
`EderDelta()`, `EderSimpleDelta()`

*Use when:* you want to dampen noisy low-frequency words in the MFW tail — Eder's
weighting penalises contributions from less-frequent features.
*Don't use when:* your MFW list is already short (n < 100); there's little tail to down-
weight.
*Expect:* same shape as Burrows Delta; different ranking when tail-MFW contributions
would otherwise dominate.

Eder (2015). Two variants: `EderDelta` with explicit per-feature weights, `EderSimpleDelta`
with a simplified scheme.

### ArgamonLinearDelta
`ArgamonLinearDelta()`

*Use when:* you specifically want Euclidean (L2) distance rather than L1 — appropriate
when features are approximately Gaussian after z-scoring.
*Don't use when:* the MFW distribution is skewed enough to produce outliers; L2
penalises outliers quadratically. Prefer `CosineDelta` or `BurrowsDelta`.
*Expect:* a distance score per candidate; same ranking shape as other Deltas with
different sensitivity to large per-feature deviations.

Argamon (2008). Probabilistic interpretation of Delta under a Gaussian generative
model.

### QuadraticDelta
`QuadraticDelta()`

*Use when:* you want to replicate experiments that use squared-L2 distance — equivalent
to Argamon Delta without the square root.
*Don't use when:* you need a calibrated distance for downstream combining; squared
distances aren't a proper metric.
*Expect:* a distance score per candidate; monotonic with Argamon Linear Delta so
ranking is identical.
```

- [ ] **Step 2: Insert per-method H3 subsections under "Contrast — Zeta"**

After the existing Zeta code block, before `## Dimensionality reduction`, insert:

```markdown
### ZetaClassic
`ZetaClassic(group_by=..., top_k=..., group_a=..., group_b=...)`

*Use when:* you have two pre-defined author groups (or authors) and want to know which
words each group prefers over the other — Craig's classic Zeta.
*Don't use when:* you only want to rank one unknown document against candidates (use a
Delta instead), or when your groups are very small (<10 docs each).
*Expect:* two tables of top-k words; each word has a proportion-in-A and
proportion-in-B; large differences are the distinctive vocabulary.

### ZetaEder
`ZetaEder(group_by=..., top_k=..., group_a=..., group_b=...)`

*Use when:* you want Zeta with the Eder (2017) smoothing — handles very rare or very
common words more gracefully than the classic version.
*Don't use when:* you're reproducing Burrows/Craig-era results for comparison; use
`ZetaClassic` for historical parity.
*Expect:* same output shape as `ZetaClassic`; smoother ranking near the tails.
```

- [ ] **Step 3: Insert per-method H3 subsections under "Dimensionality reduction"**

Replace the current single-paragraph Dimensionality reduction section with:

```markdown
## Dimensionality reduction

All reducers accept a `FeatureMatrix` and return a `Result` with 2-D / n-D coordinates.

### PCAReducer
`PCAReducer(n_components=2)`

*Use when:* you want a fast, interpretable 2-D or 3-D projection where axes are
orthogonal variance directions. Default choice for "plot my corpus" questions.
*Don't use when:* authorship differences are highly non-linear; PCA's linear axes will
miss curved manifolds.
*Expect:* `coords` (n_docs × n_components) + `explained_variance_ratio_` per component.

### UMAPReducer
`UMAPReducer(n_components=2, n_neighbors=15, min_dist=0.1)`

*Use when:* you want a non-linear projection that preserves local *and* global
structure — typically the best-looking 2-D visualisation of stylometric features.
*Don't use when:* you need reproducibility without pinning a seed — UMAP is
stochastic. Always set `random_state`.
*Expect:* `coords` (n_docs × n_components). Requires `bitig[viz]`.

### TSNEReducer
`TSNEReducer(n_components=2, perplexity=30)`

*Use when:* you want a non-linear projection that emphasises local neighbourhood
structure — authors cluster tightly.
*Don't use when:* you need inter-cluster distances to be meaningful (t-SNE warps them),
or when you plan to use the coordinates as features for a downstream method.
*Expect:* `coords` (n_docs × n_components). Non-deterministic without a seed.

### MDSReducer
`MDSReducer(n_components=2, metric=True)`

*Use when:* you want a projection that tries to preserve pairwise Delta distances as
literally as possible — good for interpreting dendrogram + scatter together.
*Don't use when:* you have a large corpus (>500 docs); MDS scales poorly.
*Expect:* `coords` (n_docs × n_components) + `stress` (lower = better fit).
```

- [ ] **Step 4: Insert per-method H3 subsections under "Clustering"**

Replace the current one-paragraph Clustering section with:

```markdown
## Clustering

Clusterers accept a `FeatureMatrix` and produce cluster labels; hierarchical clustering
also returns the linkage matrix for dendrograms.

### HierarchicalCluster
`HierarchicalCluster(linkage="ward")`

*Use when:* you want a dendrogram — the canonical stylometry visualisation — where
leaves are documents and branch heights are distances.
*Don't use when:* your corpus is large enough (>2000 docs) that dendrogram inspection
is no longer practical.
*Expect:* `labels` (n_docs,) + `linkage_matrix` usable with `scipy.cluster.hierarchy.dendrogram`.

Supported linkages: `"ward"` (default, variance-minimising), `"average"`, `"complete"`,
`"single"`.

### KMeansCluster
`KMeansCluster(n_clusters=3, seed=42)`

*Use when:* you have a rough expected cluster count and want spherical clusters of
comparable size — fastest clustering option.
*Don't use when:* cluster sizes are very unequal, cluster shapes are elongated, or you
don't know `n_clusters` ahead of time (use `HDBSCANCluster`).
*Expect:* `labels` (n_docs,) + cluster centroids.

### HDBSCANCluster
`HDBSCANCluster(min_cluster_size=5)`

*Use when:* you don't know the cluster count ahead of time, expect variable cluster
density, or want "noise" points to be labelled as outliers (-1).
*Don't use when:* your corpus is small (<30 docs); HDBSCAN's density estimates get
unstable.
*Expect:* `labels` (n_docs,) with -1 for noise; `probabilities` (cluster-membership
confidence).
```

- [ ] **Step 5: Insert the Consensus-trees subsection**

Replace the existing Consensus trees section with:

```markdown
## Consensus trees

### BootstrapConsensus
`BootstrapConsensus(mfw_bands=[100, 200, 300], replicates=20)`

*Use when:* you want robustness evidence for a dendrogram — repeatedly resample the
MFW feature set and see which clades survive.
*Don't use when:* you need one quick visualisation; bootstrap runs many Delta +
clustering cycles and is slow.
*Expect:* a Newick consensus tree with clade-support values (fraction of replicates
where that clade appears).

Eder (2017). Integrates out the "how many MFW?" knob by sampling across bands.
```

- [ ] **Step 6: Add a gloss block before "Classification + CV"**

Under `## Classification + CV`, immediately after the current paragraph starting "Any sklearn classifier…" and before the bullet list of CV kinds, insert this gloss:

```markdown
*Use when:* you have labelled documents (author or group) and want standard ML
performance numbers — accuracy, F1, confusion matrices — with stylometry-aware CV that
doesn't leak author identity between folds.
*Don't use when:* you have fewer than ~20 documents per class; CV becomes statistically
meaningless. Also don't use for single-case verification (use `GeneralImpostors`).
*Expect:* per-fold predictions, a mean accuracy / macro-F1, and fold-level `Result`
objects for downstream plots.
```

- [ ] **Step 7: Insert per-method H3 subsections under "Bayesian"**

Replace the current bullet list with:

```markdown
## Bayesian

### BayesianAuthorshipAttributor
`BayesianAuthorshipAttributor()`

*Use when:* you want posterior probabilities over N candidate authors with principled
Dirichlet smoothing — the Wallace–Mosteller Federalist approach.
*Don't use when:* your features are z-scored (it expects raw counts; use
`MFWExtractor(scale="none")`).
*Expect:* `predict_proba` returns per-document posterior probability vectors over
candidates. No need for `bitig[bayesian]` — this variant is pure NumPy.

### HierarchicalGroupComparison
`HierarchicalGroupComparison(group_a=..., group_b=..., feature_name=...)`

*Use when:* you want to test whether two author populations differ systematically in a
stylistic feature, with full per-author uncertainty — a PyMC varying-intercept model.
*Don't use when:* you only have one author per group (no pooling signal) or need a
fast screening method (MCMC sampling is slow; use a frequentist Zeta first).
*Expect:* an arviz `InferenceData` with posterior draws for the group-difference
parameter. Requires `bitig[bayesian]`.
```

- [ ] **Step 8: Add a gloss block before the "Forensic methods" table**

Under `## Forensic methods`, right after the `Under bitig.forensic:` line and before the table, insert:

```markdown
*Use when:* you want bitig's one-case verification or calibration layer — see the
dedicated [Forensic toolkit](../forensic/index.md) pages for gloss-per-method detail.
*Don't use when:* you have a closed candidate set and just want attribution — use
Delta variants above.
*Expect:* scorers that return calibrated log-LR + evidence metadata, not classifier
accuracy.
```

- [ ] **Step 9: Mirror all changes in `concepts/methods.tr.md`**

Open `docs/site/concepts/methods.tr.md`. Apply every new section and gloss block from Steps 1–8 in Turkish, using the gloss-pattern phrases (*Şu durumda kullanın:* / *Şu durumda kullanmayın:* / *Beklenen sonuç:*) and the existing glossary terms. Keep all method names (`BurrowsDelta`, `CosineDelta`, etc.), code identifiers, and constructor snippets verbatim.

The TR section headings for the new H3s use transliterated method names directly — e.g., `### BurrowsDelta`, `### CosineDelta`, `### PCAReducer`, `### KMeansCluster`, `### HDBSCANCluster`. Do not translate the class names themselves.

Translate the gloss prose content — do not paraphrase, do not shorten. If a sentence feels awkward in literal Turkish, restructure it for natural SOV syntax but preserve the semantic content.

- [ ] **Step 10: Build strict**

Run: `.venv/bin/mkdocs build --strict`

Expected: PASS.

- [ ] **Step 11: Verify gloss-pattern terms appear consistently**

Run:

```bash
grep -c 'Şu durumda kullanın\|Şu durumda kullanmayın\|Beklenen sonuç' docs/site/concepts/methods.tr.md
```

Expected: at least 45 hits (15 methods × 3 markers). If fewer, a gloss block was skipped in translation.

- [ ] **Step 12: Commit**

```bash
git add docs/site/concepts/methods.md docs/site/concepts/methods.tr.md
git commit -m "docs(concepts): gloss every method in methods.md (EN + TR)"
```

---

### Task 5: Gloss every extractor in `concepts/features.md` (EN + TR)

**Files:**
- Modify: `docs/site/concepts/features.md`
- Modify: `docs/site/concepts/features.tr.md`

Current state: a single "Available extractors" table summarises 12 extractors in one line each. This task keeps the table as a summary and adds per-extractor H4 subsections with gloss blocks below.

- [ ] **Step 1: Add the per-extractor subsections**

Open `docs/site/concepts/features.md`. After the existing "Available extractors" table (and before "Composing features"), insert this block verbatim:

````markdown
### Extractor detail

Each extractor above is a callable object; `fit_transform(corpus)` returns a
`FeatureMatrix`.

#### MFWExtractor
`MFWExtractor(n=200, scale="zscore", lowercase=True)`

*Use when:* you want the canonical stylometric feature — relative frequencies of
the most-frequent words. Default choice for Delta-family attribution.
*Don't use when:* your corpus is very small (<200 unique tokens), or the question is
topic-invariant (MFW is topic-sensitive; see `CategorizedCharNgramExtractor`).
*Expect:* an `(n_docs, n)` float matrix; rows sum to ~1 under `scale="l1"`,
zero-centred unit-variance under `scale="zscore"`.

#### CharNgramExtractor
`CharNgramExtractor(n=3, include_boundaries=True)`

*Use when:* you want features that capture sub-word style (prefixes, suffixes,
punctuation adjacency) and that cope with OOV words or misspellings.
*Don't use when:* your languages mix scripts (n-grams across scripts produce noise),
or you specifically need word-level semantic sensitivity.
*Expect:* sparse count matrix delegated to sklearn's `CountVectorizer`.

#### WordNgramExtractor
`WordNgramExtractor(n=1, lowercase=True)`

*Use when:* unigrams (MFW equivalent) or short bigram phrases are what you need and
you don't want z-scoring. Bigrams useful for detecting fixed expressions.
*Don't use when:* n ≥ 3 in small corpora — sparsity dominates. Use `MFWExtractor`
for unigrams unless you need raw counts.
*Expect:* sparse count matrix; vocabulary grows fast with n.

#### PosNgramExtractor
`PosNgramExtractor(n=2, coarse=False)`

*Use when:* you want syntactic-style features (sequences of part-of-speech tags) —
insensitive to content words, sensitive to register and syntactic register.
*Don't use when:* your spaCy pipeline doesn't include a tagger (most `_trf` models
do), or your corpus is very small per-doc.
*Expect:* sparse count matrix over POS n-grams. `coarse=True` uses UD coarse tags
(fewer dimensions, more robust).

#### DependencyBigramExtractor
`DependencyBigramExtractor()`

*Use when:* you want syntax-sensitive style features — specifically, the
(head-lemma, dependency-relation, child-lemma) triples parsed by spaCy.
*Don't use when:* your parser is a bottleneck; dependency parsing is the slowest
step in the spaCy pipeline and you may be able to substitute POS n-grams.
*Expect:* sparse count matrix over dependency triples.

#### FunctionWordExtractor
`FunctionWordExtractor(wordlist=None)`

*Use when:* you want the short, topic-insensitive function-word list (the classic
anti-topic signal for stylometry) for the document's language.
*Don't use when:* your corpus mixes languages without a per-doc language tag — the
per-language word list won't apply.
*Expect:* `(n_docs, |wordlist|)` relative-frequency matrix. Defaults come from the
bundled per-language list (see [Languages](languages.md)).

#### PunctuationExtractor
`PunctuationExtractor()`

*Use when:* you want pure-style features that are nearly topic-invariant —
punctuation usage is remarkably author-specific and corpus-robust.
*Don't use when:* your source text has been normalised or stripped of punctuation
(e.g., OCR output without correction).
*Expect:* `(n_docs, ~20)` matrix of ASCII punctuation relative frequencies.

#### ReadabilityExtractor
`ReadabilityExtractor()`

*Use when:* you want readability-as-style — Flesch, FK-grade, Gunning Fog, etc. —
as a lightweight feature set to combine with MFW.
*Don't use when:* readability itself is the question (for that, read the metric
directly; don't bundle into a Delta). For non-English, use the per-language
native-formula variant — see `concepts/languages.md`.
*Expect:* `(n_docs, 6)` matrix of readability indices (English defaults: Flesch,
FK-grade, Gunning Fog, Coleman-Liau, ARI, SMOG).

#### SentenceLengthExtractor
`SentenceLengthExtractor()`

*Use when:* you want the sentence-rhythm signature — mean, SD, and skew of
per-sentence token counts. Small but strong stylistic signal.
*Don't use when:* your text has aggressive sentence-boundary errors (e.g., ALL
CAPS legal text breaks most sentencizers).
*Expect:* `(n_docs, 3)` matrix: `[mean, std, skew]`.

#### LexicalDiversityExtractor
`LexicalDiversityExtractor()`

*Use when:* you want vocabulary-richness features — TTR, MATTR, MTLD, HD-D, Yule's
K/I, Herdan's C, Simpson's D. Eight indices let you compare sensitivities.
*Don't use when:* your documents are very short (<200 tokens); most indices become
unstable.
*Expect:* `(n_docs, 8)` matrix; columns are the 8 indices.

#### SentenceEmbeddingExtractor
`SentenceEmbeddingExtractor(model="paraphrase-MiniLM-L6-v2")`

*Use when:* you want a modern neural-embedding feature set — pooled
sentence-transformer output per document. Strong in classification + clustering;
fast enough for moderate corpora.
*Don't use when:* your hardware lacks GPU / MPS and your corpus is large (CPU
inference is slow), or when interpretability matters (these vectors are opaque).
*Expect:* `(n_docs, embedding_dim)` dense matrix. Requires `bitig[embeddings]`.

#### ContextualEmbeddingExtractor
`ContextualEmbeddingExtractor(model="bert-base-multilingual-cased", pooling="mean")`

*Use when:* you want HuggingFace-model hidden states aggregated per document —
language-specific embeddings (e.g., `dbmdz/bert-base-turkish-cased` for Turkish)
with configurable pooling.
*Don't use when:* you don't need a specific model's representation — use
`SentenceEmbeddingExtractor` for a lighter, faster default.
*Expect:* `(n_docs, hidden_dim)` dense matrix. Requires `bitig[embeddings]`.
````

- [ ] **Step 2: Add gloss blocks to the "Forensic feature extractors" section**

Under `## Forensic feature extractors`, after the intro paragraph "Two topic-invariant extractors…" and before the bullet list, expand each bullet into a full H4 entry with a gloss. Replace the existing bullet list:

```markdown
#### CategorizedCharNgramExtractor
`CategorizedCharNgramExtractor(n=4, categories=("prefix","suffix","punct"))`

*Use when:* you want topic-invariant character-level features for forensic
verification — n-grams classified by position in the word so you can keep only
the style-carrying categories (affixes, punctuation) and drop the topic-sensitive
whole-word category.
*Don't use when:* topic robustness isn't the goal — a plain `CharNgramExtractor`
is faster and carries more signal per dimension.
*Expect:* sparse count matrix restricted to the chosen n-gram categories.

Sapkota et al. 2015; `categories=("prefix","suffix","punct")` is the affix-only
recipe that generalises best across topics.

#### distort_corpus
`distort_corpus(corpus, mode="dv_ma")`

*Use when:* you want Stamatatos (2013) topic masking — replaces content words with
placeholders while keeping function words and punctuation. Pair with any
extractor for a topic-invariant pipeline.
*Don't use when:* your analysis needs content-word signal (e.g., Zeta looking for
distinctive vocabulary).
*Expect:* a new Corpus object you feed to any existing extractor. Modes: `"dv_ma"`
masks all content words, `"dv_sa"` masks selectively.
```

- [ ] **Step 3: Mirror the changes in `concepts/features.tr.md`**

Open `docs/site/concepts/features.tr.md`. Apply the equivalent sections in Turkish using the glossary and gloss-pattern phrases. Keep all class names, constructor signatures, and configuration keys verbatim. Translate all prose.

Existing glossary terms to use: *öznitelik* (feature), *öznitelik matrisi* (feature matrix), *en sık sözcükler (MFW)*, *işlev sözcüğü* (function word), *karakter n-gramı*, *sözcük n-gramı*, *sözcük türü (POS)*, *bağımlılık bigramı*, *noktalama* (punctuation), *okunabilirlik* (readability), *cümle uzunluğu*, *sözcüksel çeşitlilik* (lexical diversity), *cümle gömmesi*, *bağlamsal gömme*, *konudan bağımsız* (topic-invariant).

- [ ] **Step 4: Build strict**

Run: `.venv/bin/mkdocs build --strict`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/site/concepts/features.md docs/site/concepts/features.tr.md
git commit -m "docs(concepts): gloss every extractor in features.md (EN + TR)"
```

---

### Task 6: Gloss every metric in `forensic/evaluation.md` (EN + TR)

**Files:**
- Modify: `docs/site/forensic/evaluation.md`
- Modify: `docs/site/forensic/evaluation.tr.md`

- [ ] **Step 1: Replace the metrics table**

In `docs/site/forensic/evaluation.md`, find the current table under `## The metrics`:

```markdown
| Metric | Measures | Reference |
|---|---|---|
| `auc` | Ranking ability. 1.0 perfect, 0.5 random. | — |
...
```

Replace it with a four-column table that adds "Use for":

```markdown
| Metric | Measures | Use for | Range | Reference |
|---|---|---|---|---|
| `auc` | Ranking quality | **Choosing between systems.** Higher AUC → the system ranks same-author pairs above different-author pairs more reliably. | 0.5 (random) – 1.0 (perfect) | — |
| `c_at_1` | Accuracy with abstention credit | **Operational decisions** where "don't know" is safer than a wrong answer. | 0 – 1 | Peñas & Rodrigo 2011 |
| `f05u` | Precision-weighted F with non-answer penalty | **PAN-style evaluation.** Penalises over-confident wrong answers. | 0 – 1 | Bevendorff et al. PAN 2022 |
| `brier` | Posterior calibration | **Probabilistic output quality.** Lower = better-calibrated probabilities. | 0 (perfect) – 1 (worst) | Brier 1950 |
| `ece` | Expected calibration error | **Is `predict_proba` honest?** Bins predictions by confidence; compares claimed vs. actual accuracy. | 0 (perfect) – 1 | — |
| `cllr` | Log-likelihood-ratio cost | **Forensic LR quality.** The strict proper scoring rule for evidential output. | 0 (perfect) – ∞ | Brümmer & du Preez 2006 |
| `tippett` | LR distribution plot | **Sanity-checking calibration visually.** Cumulative target vs. non-target LR curves should separate. | — | — |
```

- [ ] **Step 2: Add a gloss block before each `:::` autodoc directive**

Under `## Reference` (which is a list of `::: bitig.forensic.metrics.<fn>` directives), insert a gloss paragraph immediately before each directive. For example, before `::: bitig.forensic.metrics.auc`:

```markdown
### AUC

*Use when:* comparing two verification systems on the same benchmark — AUC is
threshold-independent.
*Don't use when:* you need an operational decision — AUC says nothing about where to
set the threshold.
*Expect:* a single number in `[0.5, 1]`. Does not depend on predicted probabilities
being calibrated.

::: bitig.forensic.metrics.auc
```

Apply the same pattern before the other six autodoc directives:

```markdown
### c@1

*Use when:* your system can abstain ("don't know") and you want to credit that
honestly — accuracy plus a partial-credit bonus for abstention.
*Don't use when:* your system always outputs a decision; `c@1` reduces to accuracy.
*Expect:* a single number in `[0, 1]`. Dominates accuracy only when abstention rate > 0.

::: bitig.forensic.metrics.c_at_1

### F0.5u

*Use when:* you're scoring a PAN-CLEF verification track — it's the official metric
since PAN 2022, precision-weighted and with a non-answer penalty.
*Don't use when:* you're reporting to a non-PAN audience; it's a specialist metric.
*Expect:* a single number in `[0, 1]`.

::: bitig.forensic.metrics.f05u

### C_llr

*Use when:* you need to quantify **how good your LR output is** in forensic terms —
this is the strict proper scoring rule the speaker-recognition community settled on.
*Don't use when:* your scorer outputs accuracy-style probabilities; `C_llr` expects
log-likelihood ratios.
*Expect:* a single non-negative number; 0 is perfect; 1 is uninformative (matches a
coin flip).

::: bitig.forensic.metrics.cllr

### ECE

*Use when:* you want to audit probabilistic honesty — ECE bins predictions by
claimed confidence and checks whether actual accuracy matches.
*Don't use when:* your dev set is small (<200 trials); ECE's bin estimates become
noisy.
*Expect:* a single number in `[0, 1]`; 0 is perfect calibration.

::: bitig.forensic.metrics.ece

### Brier

*Use when:* you want a proper scoring rule for probabilistic classifiers (not LR
outputs) — classic squared-error between predicted probability and ground truth.
*Don't use when:* you need a forensic LR-specific metric — use `C_llr`.
*Expect:* a single number in `[0, 1]`; 0 is perfect.

::: bitig.forensic.metrics.brier

### Tippett

*Use when:* you want a visual calibration check — plot target-trial vs. non-target
log-LRs as cumulative distributions.
*Don't use when:* you need a single-number summary (use `C_llr`).
*Expect:* two arrays of cumulative LRs (target and non-target) ready for a matplotlib
plot.

::: bitig.forensic.metrics.tippett
```

Also add a gloss block above the `::: bitig.forensic.metrics.compute_pan_report` directive near the top of the `## Reference` section:

```markdown
### compute_pan_report

*Use when:* you have a labelled batch of verification trials and want every standard
metric in one call — AUC, c@1, F0.5u, Brier, ECE, (optionally) C_llr.
*Don't use when:* you only need one metric; each metric function is callable directly.
*Expect:* a `PANReport` dataclass with every field populated.

::: bitig.forensic.metrics.compute_pan_report

::: bitig.forensic.metrics.PANReport
```

- [ ] **Step 3: Mirror the changes in `evaluation.tr.md`**

Translate the new table and every gloss block in `docs/site/forensic/evaluation.tr.md`. Metric abbreviations (`auc`, `c_at_1`, `f05u`, `brier`, `ece`, `cllr`, `tippett`) stay English. Use glossary terms: *kalibrasyon* (calibration), *olabilirlik oranı* (likelihood ratio), *değerlendirme* (evaluation), *doğrulama* (verification).

- [ ] **Step 4: Build strict**

Run: `.venv/bin/mkdocs build --strict`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/site/forensic/evaluation.md docs/site/forensic/evaluation.tr.md
git commit -m "docs(forensic): gloss every metric in evaluation.md (EN + TR)"
```

---

### Task 7: Gloss the verifiers in `forensic/verification.md` (EN + TR)

**Files:**
- Modify: `docs/site/forensic/verification.md`
- Modify: `docs/site/forensic/verification.tr.md`

Current state: `verification.md` has `## General Impostors` and `## Unmasking` as H2 sections, plus a "When to pick which" tail section.

- [ ] **Step 1: Add gloss block under `## General Impostors`**

Immediately after the `## General Impostors (Koppel & Winter 2014)` heading and before the numbered step list ("For a questioned document Q..."), insert:

```markdown
*Use when:* you have one questioned document, one candidate's known documents, and a
pool of ~100+ impostor documents from other authors — the forensically canonical
same-author-or-not question with a closed candidate.
*Don't use when:* you have no impostor pool available, or your candidate's known
writings are less than ~1000 words total (the test becomes sample-size-bound).
*Expect:* a score in `[0, 1]`; calibrate with `CalibratedScorer` before reporting as
an LR.
```

- [ ] **Step 2: Add gloss block under `## Unmasking`**

Immediately after the `## Unmasking (Koppel & Schler 2004)` heading and before "A distribution-free…", insert:

```markdown
*Use when:* you have long same-author prose candidates (novel chapters, long essays,
blog archives) and want a distribution-free verification — the accuracy-drop curve
itself is interpretable evidence.
*Don't use when:* your documents are short (<~1500 words per side) — Unmasking needs
enough chunks to run cross-validation meaningfully.
*Expect:* an accuracy curve across elimination rounds; same-author pairs show a steep
drop, different-author pairs stay near random or above.
```

- [ ] **Step 3: Convert the "When to pick which" bullets to a proper gloss table**

Replace the existing bullet list under `### When to pick which` with a two-column table:

```markdown
### When to pick which

| Situation | Pick |
|---|---|
| Short CMC / threat texts (< ~2000 words total) | `GeneralImpostors`. Unmasking needs more text per side to run CV meaningfully. |
| Long prose (novels, essays, blog archives) | `Unmasking` — the accuracy-drop curve is directly interpretable. Pair with GI as a second opinion. |
| Building an evidential report | Run both, calibrate both with `CalibratedScorer`. Agreement between the two is itself evidential signal (Juola-style multi-method verdict). |
```

- [ ] **Step 4: Mirror the changes in `verification.tr.md`**

Translate each new gloss block and the "When to pick which" table in Turkish. Use glossary terms: *yazar doğrulama*, *delil zinciri*, *kalibrasyon*, *olabilirlik oranı*, *sahte-aday* (impostor).

- [ ] **Step 5: Build strict**

Run: `.venv/bin/mkdocs build --strict`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/site/forensic/verification.md docs/site/forensic/verification.tr.md
git commit -m "docs(forensic): gloss General Impostors + Unmasking (EN + TR)"
```

---

### Task 8: Gloss the calibration methods in `forensic/calibration.md` (EN + TR)

**Files:**
- Modify: `docs/site/forensic/calibration.md`
- Modify: `docs/site/forensic/calibration.tr.md`

- [ ] **Step 1: Add a one-paragraph page intro**

Immediately after the `# Calibration & LR output` (or `# Calibration`) heading, before the first existing paragraph, insert:

```markdown
*Use when:* your verifier outputs raw scores (distances, fractions, probabilities)
that need to be turned into calibrated log-likelihood ratios before an evidential
report.
*Don't use when:* your scorer is already a well-calibrated LR — skip straight to
evaluation.
*Expect:* a scorer wrapper whose `predict_proba` / `log_lr` outputs are
probability-calibrated against a labelled development set.

Raw scores from a verifier are rarely honest probabilities out-of-the-box. This page
covers the two standard post-hoc calibration methods plus the chain-of-custody
metadata that turns a calibrated score into a court-ready LR statement.
```

- [ ] **Step 2: Add gloss blocks for Platt and isotonic calibration**

Find the section(s) describing Platt calibration and isotonic calibration (likely under `## CalibratedScorer` or similar). Add an H3 + gloss for each:

```markdown
### Platt calibration

*Use when:* your scorer's decision boundary is approximately linear in log-odds —
logistic-regression-like shape. Fewer parameters than isotonic; needs fewer labelled
trials.
*Don't use when:* your score-to-probability relationship is non-monotonic or sharply
bent — Platt's sigmoid will underfit.
*Expect:* a scalar-parameter sigmoid fit; `predict_proba` outputs calibrated
probabilities via `1 / (1 + exp(a*score + b))`.

### Isotonic calibration

*Use when:* your scorer's decision boundary is non-linear and you have enough
labelled trials (≥500) to fit a non-parametric curve.
*Don't use when:* your dev set is small — isotonic overfits with few points.
*Expect:* a piecewise-constant calibration function; `predict_proba` outputs the
monotone increasing step function.
```

- [ ] **Step 3: Add gloss for CalibratedScorer**

Under the `## CalibratedScorer` H2 (or wherever the class is documented), right after the heading, insert:

```markdown
*Use when:* you want to wrap any scorer (`GeneralImpostors`, `Unmasking`, a custom
Delta classifier) so it produces calibrated probabilities and log-LRs in one call.
*Don't use when:* your upstream scorer already emits calibrated output.
*Expect:* `score(q, k)` returns raw; `predict_proba(q, k)` returns calibrated
probability; `log_lr(q, k)` returns the evidential quantity.
```

- [ ] **Step 4: Mirror in `calibration.tr.md`**

Translate every gloss block and the page intro. Use glossary terms: *kalibrasyon*, *olabilirlik oranı*, *sınıflandırıcı*, *doğrulayıcı* (verifier).

- [ ] **Step 5: Build strict**

Run: `.venv/bin/mkdocs build --strict`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/site/forensic/calibration.md docs/site/forensic/calibration.tr.md
git commit -m "docs(forensic): gloss Platt + isotonic + CalibratedScorer (EN + TR)"
```

---

### Task 9: Gloss topic-invariance in `forensic/topic-invariance.md` (EN + TR)

**Files:**
- Modify: `docs/site/forensic/topic-invariance.md`
- Modify: `docs/site/forensic/topic-invariance.tr.md`

- [ ] **Step 1: Add page-intro gloss**

Immediately after the `# Topic-invariant features` heading, before the first existing paragraph, insert:

```markdown
*Use when:* your questioned and known documents might be on different topics — you
need features that capture style without leaking topic.
*Don't use when:* topic is part of the question (for example, a plagiarism check
where the two documents *should* share content). Use regular features then.
*Expect:* feature extractors that discard most content-word signal while preserving
function-word, morphology, and punctuation patterns.

Two techniques live under `bitig.forensic`: Sapkota char-n-gram *categorisation* and
Stamatatos *distortion*. Both compose with any downstream verifier.
```

- [ ] **Step 2: Add gloss for Sapkota categories**

Find the `## Sapkota categorised n-grams` (or similar) H2. Immediately after the heading, insert:

```markdown
*Use when:* you want char-n-gram features for verification but need to strip
topic-sensitive whole-word n-grams — keeping only affixes, punctuation-adjacent, and
space-adjacent categories.
*Don't use when:* your corpus is so small that further filtering collapses the
feature space below ~500 dimensions.
*Expect:* a sparse count matrix with only the chosen categories; default
`("prefix","suffix","punct")` is the affix-only recipe that generalises best across
topics.
```

- [ ] **Step 3: Add gloss for Stamatatos distortion**

Find the `## Stamatatos distortion` (or similar) H2. Immediately after the heading, insert:

```markdown
*Use when:* you want aggressive topic removal via content-word masking — replaces
content words with placeholders while preserving function words, morphology, and
punctuation.
*Don't use when:* you need any content-word signal downstream (e.g., Zeta on
distinctive vocabulary).
*Expect:* a new `Corpus` object you pass to any existing extractor. Modes: `"dv_ma"`
masks *all* content words, `"dv_sa"` masks selectively by POS.
```

- [ ] **Step 4: Mirror in `topic-invariance.tr.md`**

Translate the page-intro gloss and both method glosses. Use glossary terms: *konudan bağımsız*, *öznitelik*, *doğrulama*, *işlev sözcüğü*, *noktalama*.

- [ ] **Step 5: Build strict**

Run: `.venv/bin/mkdocs build --strict`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/site/forensic/topic-invariance.md docs/site/forensic/topic-invariance.tr.md
git commit -m "docs(forensic): gloss Sapkota categories + Stamatatos distortion (EN + TR)"
```

---

### Task 10: Gloss the reporting page in `forensic/reporting.md` (EN + TR)

**Files:**
- Modify: `docs/site/forensic/reporting.md`
- Modify: `docs/site/forensic/reporting.tr.md`

- [ ] **Step 1: Add page-intro gloss**

Immediately after the `# Reporting` heading, before the first existing paragraph, insert:

```markdown
*Use when:* you have a calibrated verification `Result` and need a court-ready
report — chain-of-custody metadata, LR statement framed on the ENFSI verbal scale,
and an auditable HTML artefact.
*Don't use when:* you want an exploratory research figure — use the standard
reporting path in `concepts/results.md`.
*Expect:* a rendered HTML report with fixed sections: case metadata, hypothesis
pair, feature pipeline, calibrated LR, verbal-scale statement, and a Tippett plot.
```

- [ ] **Step 2: Add gloss above the ENFSI verbal scale table**

Find the section containing the ENFSI / Nordgaard verbal scale (likely a table of
"weak support / moderate support / strong support / very strong support" entries).
Immediately before that table, insert:

```markdown
### Verbal scale

*Use when:* you need to translate a log-LR into the plain-language descriptor
expected in a forensic report (ENFSI 2015 / Nordgaard et al. 2012).
*Don't use when:* you're reporting to a statistical audience — quote the log-LR with
its `C_llr` directly.
*Expect:* a one-line verbal statement keyed to the log-LR magnitude.
```

- [ ] **Step 3: Add gloss for `build_forensic_report`**

Find the `## build_forensic_report` (or `::: bitig.report.build_forensic_report`) section. Insert before the autodoc directive:

```markdown
*Use when:* you want a single-call path from `Result` to court-ready HTML — hands the
chain-of-custody fields, calibrated scores, verbal scale, and Tippett plot into a
Jinja2 template.
*Don't use when:* you're producing a research paper figure — use the standard
`bitig report` CLI or the plotting helpers in `concepts/methods.md`.
*Expect:* a path to the rendered HTML file; optional PDF export requires
`bitig[reports]`.
```

- [ ] **Step 4: Mirror in `reporting.tr.md`**

Translate every new gloss. Use glossary terms: *delil zinciri*, *olabilirlik oranı*, *sözel ölçek* (verbal scale), *rapor*.

- [ ] **Step 5: Build strict**

Run: `.venv/bin/mkdocs build --strict`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/site/forensic/reporting.md docs/site/forensic/reporting.tr.md
git commit -m "docs(forensic): gloss ENFSI scale + build_forensic_report (EN + TR)"
```

---

### Task 11: End-to-end verification + PR

**Files:** (none — verification + push only)

- [ ] **Step 1: Full clean build**

```bash
rm -rf site/
.venv/bin/mkdocs build --strict
```

Expected: PASS with no warnings. Build log should show "Translated 23 navigation elements to 'tr'" (22 from the multilingual rollout + 1 for the new choosing page).

- [ ] **Step 2: Verify the new choosing page renders in both languages**

```bash
test -f site/concepts/choosing/index.html && test -f site/tr/concepts/choosing/index.html && echo "OK" || echo "MISSING"
```

Expected: `OK`.

- [ ] **Step 3: Verify per-method H3 anchors exist**

Run:

```bash
grep -c 'id="burrowsdelta"\|id="cosinedelta"\|id="pcareducer"\|id="kmeanscluster"\|id="hdbscancluster"\|id="zetaclassic"' site/concepts/methods/index.html
```

Expected: at least 6 hits. If 0, the choosing page's method links will 404 — debug before proceeding.

- [ ] **Step 4: Gloss-marker count sanity check across all edited pages**

Run:

```bash
grep -c '\*Use when:\*\|\*Don.t use when:\*\|\*Expect:\*' docs/site/concepts/methods.md docs/site/concepts/features.md docs/site/forensic/evaluation.md docs/site/forensic/verification.md docs/site/forensic/calibration.md docs/site/forensic/topic-invariance.md docs/site/forensic/reporting.md
```

Expected: each of the 7 files shows at least 6 hits (i.e., 2+ gloss blocks × 3 markers). If any shows 0, that file was missed.

- [ ] **Step 5: Turkish gloss-marker count sanity check**

Run:

```bash
grep -c 'Şu durumda kullanın\|Şu durumda kullanmayın\|Beklenen sonuç' docs/site/concepts/methods.tr.md docs/site/concepts/features.tr.md docs/site/forensic/evaluation.tr.md docs/site/forensic/verification.tr.md docs/site/forensic/calibration.tr.md docs/site/forensic/topic-invariance.tr.md docs/site/forensic/reporting.tr.md
```

Expected: each TR file matches its EN counterpart's hit count (within ±2, allowing for sentence-structure variation).

- [ ] **Step 6: Push the branch**

```bash
git push -u origin docs/concepts-clarity
```

- [ ] **Step 7: Open PR**

```bash
gh pr create --title "docs: concepts clarity — Choosing guide + gloss/detail split" --body "$(cat <<'EOF'
## Summary

- Adds a new decision-table page at `concepts/choosing.md` routing readers to the right
  method per task.
- Adds a three-line gloss (*Use when* / *Don't use when* / *Expect*) above the technical
  detail for every method in `concepts/methods.md`, every extractor in
  `concepts/features.md`, every metric in `forensic/evaluation.md`, and each forensic
  method / calibration / topic-invariance / reporting page.
- Mirrors every change to the Turkish sibling files using the pinned glossary and
  new gloss-pattern phrases (*Şu durumda kullanın* / *Şu durumda kullanmayın* /
  *Beklenen sonuç*).

## Design + plan

- Spec: [`docs/superpowers/specs/2026-04-20-concepts-clarity-design.md`](docs/superpowers/specs/2026-04-20-concepts-clarity-design.md)
- Plan: [`docs/superpowers/plans/2026-04-20-concepts-clarity.md`](docs/superpowers/plans/2026-04-20-concepts-clarity.md)

## Test plan

- [x] `mkdocs build --strict` passes
- [x] Both `/concepts/choosing/` and `/tr/concepts/choosing/` render
- [x] Per-method H3 anchors in `methods/index.html` are present (enables links from the
  choosing table)
- [x] Every rewritten page has gloss markers in EN and TR
- [ ] Native-speaker review of Turkish gloss prose (post-merge, iterative)
- [ ] Live GitHub Pages smoke test (post-merge)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 8: Wait for CI, then merge**

Check CI: `gh pr checks <PR-number>`

When green, squash-merge:

```bash
gh pr merge <PR-number> --squash --delete-branch
```

- [ ] **Step 9: Memory update**

Append a line to `/Users/fatihbozdag/.claude/projects/-Users-fatihbozdag-Downloads-Cursor-Projects-Stylometry/memory/project_scope.md` noting that the concepts-clarity pass landed on 2026-04-20: new `concepts/choosing.md` page + gloss/detail blocks on 7 existing pages × 2 languages.

---

## Self-review notes

**Spec coverage check:**

- ✅ New `concepts/choosing.md` with task-oriented decision table — Task 2.
- ✅ `concepts/index.md` "What bitig is for" intro — Task 3.
- ✅ `concepts/methods.md` per-method gloss blocks — Task 4.
- ✅ `concepts/features.md` per-extractor gloss blocks — Task 5.
- ✅ `forensic/evaluation.md` metric "Use for" column + per-metric glosses — Task 6.
- ✅ `forensic/verification.md` method-level glosses — Task 7.
- ✅ `forensic/calibration.md` intro + method glosses — Task 8.
- ✅ `forensic/topic-invariance.md` intro + method glosses — Task 9.
- ✅ `forensic/reporting.md` intro + ENFSI + report glosses — Task 10.
- ✅ Turkish mirror per-commit — each task ends with `git add both.md both.tr.md`.
- ✅ Glossary extension for gloss pattern — Task 1.
- ✅ Build strict passes after each task — every task has a mkdocs build step.
- ✅ Existing technical content preserved — per-entry instruction "insert above" / "insert before".

**Placeholder scan:** none. Every gloss is pre-written in the plan; every code block shows exact YAML/markdown to paste.

**Type/identifier consistency:** all class names used in Task 2's choosing table (`BurrowsDelta`, `CosineDelta`, `PCAReducer`, `KMeansCluster`, `HDBSCANCluster`, `ZetaClassic`, `ZetaEder`, `BayesianAuthorshipAttributor`, `BootstrapConsensus`, `GeneralImpostors`, `Unmasking`, `CalibratedScorer`, `CategorizedCharNgramExtractor`, `distort_corpus`, `build_forensic_report`, `compute_pan_report`) are introduced as H3 subsections in Tasks 4–10 with matching anchors. The choosing page's fragment links (`methods.md#burrowsdelta`, etc.) resolve to those anchors.
