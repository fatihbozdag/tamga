# Multilingual Docs Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MkDocs Material site multilingual, launching with English (default) and Turkish, on infrastructure that scales to DE/ES/FR without further refactors.

**Architecture:** `mkdocs-static-i18n` plugin with `docs_structure: suffix` — each page lives at `page.md` (EN default) and `page.tr.md` (Turkish translation). English stays at `/`, Turkish at `/tr/`, so every existing permalink continues to resolve. Material theme wires the language switcher via `reconfigure_material: true`. Lunr search gains a Turkish stemmer via `plugins.search.lang: [en, tr]`. A terminology glossary (`docs/site/_translations/tr-glossary.md`) pins recurring technical terms for consistency.

**Tech Stack:** MkDocs 1.6+, Material 9.5+, `mkdocs-static-i18n>=1.2`, mkdocstrings, Python 3.11. CI: `.github/workflows/docs.yml` (unchanged). Deploy: GitHub Pages.

**Source spec:** `docs/superpowers/specs/2026-04-20-multilingual-docs-design.md` (commit `17863b6`).

---

## Translation Task Procedure (shared reference)

Tasks 4–8 all follow this procedure. The implementer reads the English source, drafts the Turkish equivalent into a `.tr.md` sibling file, verifies the build, and commits.

**For each English page `docs/site/<path>/<name>.md`:**

1. Read the English source completely before drafting. Translation is not a line-by-line mapping — Turkish sentence structure differs (SOV, agglutinative morphology), so paraphrasing at the sentence level is expected.
2. Create sibling file `docs/site/<path>/<name>.tr.md` with:
   - **YAML front matter (if any) preserved verbatim** — `hide:`, `status:`, etc. are MkDocs directives, not prose.
   - **Headings translated** (`# Corpus` → `# Derlem`). Match the glossary for recurring nouns.
   - **Prose, admonition bodies, captions, alt-text translated.**
   - **Code blocks copied verbatim** — Python, YAML, shell, JSON. Never translate `bitig ingest`, `cfg.preprocess.language`, CLI flag names, Python identifiers, or YAML keys.
   - **Inline code (backticks) copied verbatim** — `Corpus.language`, `--metadata`, `study.yaml`.
   - **Citations, bibliography, DOIs, URLs copied verbatim.**
   - **Brand/method names copied verbatim** — `bitig`, `Burrows`, `Eder`, `Argamon`, `General Impostors`, `Unmasking`, `PANReport`, `Delta`, `Zeta`.
   - **Mermaid diagram labels:** translate prose labels; keep English for API names.
3. Run `mkdocs build --strict` — must pass with no warnings.
4. Run a targeted grep for glossary consistency — for every glossary term used on the page, confirm the Turkish rendering matches the pinned form.
5. Commit with message `docs(i18n): translate <relative-path>.md to Turkish`.

**Glossary reference** (see `docs/site/_translations/tr-glossary.md` after Task 3 creates it — always re-read before drafting each page):

| English | Turkish |
|---|---|
| stylometry | *stilometri* |
| corpus | *derlem* |
| authorship attribution | *yazar tespiti* |
| authorship verification | *yazar doğrulama* |
| forensic linguistics | *adli dilbilim* |
| feature (stylometric) | *öznitelik* |
| function word | *işlev sözcüğü* |
| readability | *okunabilirlik* |
| classifier / classification | *sınıflandırıcı / sınıflandırma* |
| clustering | *kümeleme* |
| embedding (vector) | *gömme* |
| likelihood ratio | *olabilirlik oranı* |
| calibration | *kalibrasyon* |
| chain of custody | *delil zinciri* |
| provenance | *köken bilgisi* |

---

### Task 1: Add `mkdocs-static-i18n` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add plugin to docs extra**

Edit the `docs` list in `pyproject.toml` to add `mkdocs-static-i18n>=1.2`:

```toml
docs = [
    "mkdocs>=1.6",
    "mkdocs-material>=9.5",
    "mkdocs-static-i18n>=1.2",
    "mkdocstrings[python]>=0.25",
    "pymdown-extensions>=10.7",
]
```

- [ ] **Step 2: Sync the environment**

Run: `uv pip install -e ".[docs]"`

Expected: `mkdocs-static-i18n` installed alongside existing docs deps. No resolution conflicts.

- [ ] **Step 3: Verify baseline build still works**

Run: `mkdocs build --strict`

Expected: PASS. The plugin is installed but not yet configured in `mkdocs.yml`, so the build should behave exactly as before.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build(docs): add mkdocs-static-i18n to docs extra"
```

---

### Task 2: Configure i18n in `mkdocs.yml` + smoke-test with single TR stub

**Files:**
- Modify: `mkdocs.yml`
- Create: `docs/site/index.tr.md` (temporary stub — Task 4 replaces this with a real translation)

- [ ] **Step 1: Configure the plugin**

Edit `mkdocs.yml` — under `plugins:`, replace the existing `- search` entry and add the `- i18n` entry. The final `plugins:` block becomes:

```yaml
plugins:
  - search:
      lang:
        - en
        - tr
  - i18n:
      docs_structure: suffix
      reconfigure_material: true
      fallback_to_default: true
      languages:
        - locale: en
          default: true
          name: English
          build: true
          site_name: bitig
        - locale: tr
          name: Türkçe
          build: true
          site_name: bitig
          nav_translations:
            Home: Ana Sayfa
            Getting started: Başlangıç
            Concepts: Kavramlar
            Corpus: Derlem
            Features: Öznitelikler
            Languages: Diller
            Methods: Yöntemler
            Results & provenance: Sonuçlar ve köken bilgisi
            Forensic toolkit: Adli Araç Seti
            Verification: Doğrulama
            Calibration & LR output: Kalibrasyon ve OO çıktısı
            Topic-invariant features: Konudan Bağımsız Öznitelikler
            Evaluation (PAN suite): Değerlendirme (PAN)
            Reporting: Raporlama
            Tutorials: Öğreticiler
            Federalist Papers: Federalist Yazıları
            PAN-CLEF verification: PAN-CLEF doğrulama
            Turkish stylometry: Türkçe stilometri
            Reference: Referans
            CLI: Komut satırı
            study.yaml schema: study.yaml şeması
            Python API: Python API
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
          options:
            docstring_style: numpy
            show_source: false
            show_root_heading: true
            heading_level: 3
            members_order: source
            separate_signature: true
            show_signature_annotations: true
```

Do not add `extra.alternate` — `reconfigure_material: true` manages it automatically.

- [ ] **Step 2: Create a TR stub for the landing page**

Create `docs/site/index.tr.md` as a minimal smoke-test stub so the build has at least one TR page to render:

```markdown
---
hide:
  - navigation
---

# bitig

**Yazar tespiti, yazar grupları karşılaştırması ve adli dilbilim için hesaplamalı stilometri.**

_(Bu sayfa geçici bir taslaktır; Task 4 gerçek çeviriyle değiştirir.)_
```

- [ ] **Step 3: Build strict and confirm both languages render**

Run: `mkdocs build --strict`

Expected: PASS, no warnings. The `site/` output contains both `site/index.html` (EN) and `site/tr/index.html` (TR stub).

- [ ] **Step 4: Verify existing EN URL paths are preserved**

Run:

```bash
ls site/getting-started/index.html && ls site/concepts/index.html && ls site/forensic/index.html && ls site/tutorials/federalist/index.html && ls site/reference/cli/index.html
```

Expected: all five paths exist. If any are missing, the i18n plugin has not kept EN at root — do not proceed.

- [ ] **Step 5: Verify TR tree exists**

Run: `ls site/tr/index.html`

Expected: exists.

- [ ] **Step 6: Commit**

```bash
git add mkdocs.yml docs/site/index.tr.md
git commit -m "build(docs): enable mkdocs-static-i18n (EN default, TR at /tr/)"
```

---

### Task 3: Seed the Turkish terminology glossary

**Files:**
- Create: `docs/site/_translations/tr-glossary.md`
- Modify: `mkdocs.yml` (exclude glossary from nav via `not_in_nav`)

- [ ] **Step 1: Create the glossary file**

Create `docs/site/_translations/tr-glossary.md`:

```markdown
---
hide:
  - navigation
  - toc
---

# Turkish terminology glossary

This file pins the Turkish rendering of recurring technical terms used across the bitig
documentation. It is not rendered in the public navigation. Translators MUST consult this
table before drafting or updating a Turkish page; consistency is enforced by grep before
each commit.

| English | Turkish | Notes |
|---|---|---|
| stylometry | *stilometri* | Established loan; matches how computational-linguistics literature in Turkish refers to the field. |
| corpus | *derlem* | Standard TDK term. |
| authorship attribution | *yazar tespiti* | Preferred over "yazar atfı" for naturalness. |
| authorship verification | *yazar doğrulama* | Forensic sub-task. |
| forensic linguistics | *adli dilbilim* | Standard. |
| feature (stylometric) | *öznitelik* | Standard ML term. |
| function word | *işlev sözcüğü* | Corpus linguistics standard. |
| readability | *okunabilirlik* | Standard. |
| classifier | *sınıflandırıcı* | Standard ML. |
| classification | *sınıflandırma* | Standard ML. |
| clustering | *kümeleme* | Standard ML. |
| embedding (vector) | *gömme* | Standard ML. |
| likelihood ratio | *olabilirlik oranı* | Stats standard. |
| calibration | *kalibrasyon* | Stats loan. |
| chain of custody | *delil zinciri* | Forensic/legal standard. |
| provenance | *köken bilgisi* | Keeps "provenance" feel; literal "soy" sounds odd for data. |

Brand / proper nouns (never translate): **bitig**, **Burrows**, **Eder**, **Argamon**,
**Cosine**, **Quadratic Delta**, **Zeta**, **General Impostors**, **Unmasking**,
**Stamatatos**, **Sapkota**, **PANReport**, **Mosteller & Wallace**, **Federalist Papers**,
**PAN-CLEF**, **CalibratedScorer**, **study.yaml**.
```

- [ ] **Step 2: Exclude glossary from the public nav**

Edit `mkdocs.yml` — add a top-level `not_in_nav:` entry (or extend it if one exists):

```yaml
not_in_nav: |
  /_translations/*
```

- [ ] **Step 3: Build strict and confirm glossary is excluded**

Run: `mkdocs build --strict`

Expected: PASS, no warnings. `site/_translations/tr-glossary/index.html` may exist (the page still builds), but the nav should not list it. If the build warns about "A reference to 'tr-glossary.md' is included in the 'nav'", open `mkdocs.yml` and move the not_in_nav pattern to the correct key — on older mkdocs versions this is `exclude_docs`.

- [ ] **Step 4: Commit**

```bash
git add docs/site/_translations/tr-glossary.md mkdocs.yml
git commit -m "docs(i18n): seed Turkish terminology glossary"
```

---

### Task 4: Translate landing page + getting-started (highest-traffic pages)

**Files:**
- Modify: `docs/site/index.tr.md` (replace Task 2 stub)
- Create: `docs/site/getting-started.tr.md`

- [ ] **Step 1: Translate `docs/site/index.md` → `docs/site/index.tr.md`**

Follow the **Translation Task Procedure** above.

Pay particular attention to:
- `hide: [navigation]` front matter — preserve verbatim.
- The `<p align="center">` banner blocks with `<img>` tags — keep the HTML; translate only the `alt=` attribute text and any trailing prose.
- The card grid (`<div class="grid cards" markdown>`) — translate the card titles and descriptions, keep the `:octicons-arrow-right-24:` and `:fontawesome-solid-rocket:` emoji shortcodes and `[:octicons-arrow-right-24: ...](path.md)` link syntax.
- The `| Layer | Highlights |` capabilities table — translate the layer names and description prose; keep API names (`Burrows`, `Delta`, `Zeta`, `PANReport`, `General Impostors`) untranslated.
- The `## Status` section — translate all of it; no code or API identifiers inside.

- [ ] **Step 2: Translate `docs/site/getting-started.md` → `docs/site/getting-started.tr.md`**

Follow the **Translation Task Procedure**.

Pay particular attention to:
- All `bash`, `yaml`, `python` code blocks — copy verbatim.
- CLI command flags (`--metadata`, `--language`, `--name`) and `bitig <subcommand>` invocations — copy verbatim.
- Inline code like `study.yaml`, `corpus/`, `results/` — copy verbatim.

- [ ] **Step 3: Build strict**

Run: `mkdocs build --strict`

Expected: PASS.

- [ ] **Step 4: Glossary consistency check**

Run:

```bash
grep -hE '\b(stilometri|derlem|yazar tespiti|yazar doğrulama|adli dilbilim|öznitelik|işlev sözcüğü|okunabilirlik|gömme|olabilirlik oranı|kalibrasyon|delil zinciri|köken bilgisi)\b' docs/site/index.tr.md docs/site/getting-started.tr.md | head -30
```

Expected: visible matches show glossary terms applied consistently. No variant spellings (e.g., "stilometre", "derleme" for corpus-in-the-sense-of-corpus).

- [ ] **Step 5: Commit**

```bash
git add docs/site/index.tr.md docs/site/getting-started.tr.md
git commit -m "docs(i18n): translate landing page + getting-started to Turkish"
```

---

### Task 5: Translate the `concepts/` section (6 pages)

**Files:**
- Create: `docs/site/concepts/index.tr.md`
- Create: `docs/site/concepts/corpus.tr.md`
- Create: `docs/site/concepts/features.tr.md`
- Create: `docs/site/concepts/languages.tr.md`
- Create: `docs/site/concepts/methods.tr.md`
- Create: `docs/site/concepts/results.tr.md`

- [ ] **Step 1: Translate `concepts/index.md`**

Follow the **Translation Task Procedure**. This is the section landing page — expect a short overview with a card grid.

- [ ] **Step 2: Translate `concepts/corpus.md`**

Follow the **Translation Task Procedure**. Heavy on code blocks (`Corpus.from_directory(...)`, `corpus.filter(...)`, `corpus.groupby(...)`). All API calls copied verbatim.

- [ ] **Step 3: Translate `concepts/features.md`**

Follow the **Translation Task Procedure**. Table of feature extractors — translate descriptions, keep class names (`MFWExtractor`, `CharNgramExtractor`, `FunctionWordsExtractor`, etc.) verbatim.

- [ ] **Step 4: Translate `concepts/languages.md`**

Follow the **Translation Task Procedure**. **Priority page** — this is where readers interested in the multilingual story will land. Glossary terms `derlem`, `öznitelik`, `işlev sözcüğü`, `okunabilirlik` appear here repeatedly; consistency matters most on this page.

- [ ] **Step 5: Translate `concepts/methods.md`**

Follow the **Translation Task Procedure**. Dense with method names (`Burrows Delta`, `Eder Delta`, `Argamon Delta`, `Cosine Delta`, `Quadratic Delta`, `Zeta`, `PCA`, `UMAP`, `t-SNE`, `MDS`, `Ward`, `k-means`, `HDBSCAN`) — all copied verbatim.

- [ ] **Step 6: Translate `concepts/results.md`**

Follow the **Translation Task Procedure**. The `Result` class name, JSON field names, and Parquet column names stay English.

- [ ] **Step 7: Build strict after each page**

Run `mkdocs build --strict` after each page translation (not just at the end). If a page introduces a syntax error, fixing immediately is cheaper than debugging six pages later.

Expected: PASS after every page.

- [ ] **Step 8: Glossary consistency check across all six pages**

Run:

```bash
grep -hE '\b(stilometri|derlem|yazar tespiti|yazar doğrulama|adli dilbilim|öznitelik|işlev sözcüğü|okunabilirlik|gömme|olabilirlik oranı|kalibrasyon|delil zinciri|köken bilgisi)\b' docs/site/concepts/*.tr.md | sort | uniq -c | sort -rn | head -30
```

Expected: glossary terms dominate; no variant spellings.

- [ ] **Step 9: Commit**

```bash
git add docs/site/concepts/*.tr.md
git commit -m "docs(i18n): translate concepts section to Turkish"
```

---

### Task 6: Translate the `forensic/` section (6 pages)

**Files:**
- Create: `docs/site/forensic/index.tr.md`
- Create: `docs/site/forensic/verification.tr.md`
- Create: `docs/site/forensic/calibration.tr.md`
- Create: `docs/site/forensic/topic-invariance.tr.md`
- Create: `docs/site/forensic/evaluation.tr.md`
- Create: `docs/site/forensic/reporting.tr.md`

- [ ] **Step 1: Translate `forensic/index.md`**

Follow the **Translation Task Procedure**. Section landing page.

- [ ] **Step 2: Translate `forensic/verification.md`**

Follow the **Translation Task Procedure**. Covers General Impostors and Unmasking — keep both names verbatim. Glossary terms `yazar doğrulama`, `delil zinciri` are load-bearing here.

- [ ] **Step 3: Translate `forensic/calibration.md`**

Follow the **Translation Task Procedure**. Covers Platt / isotonic calibration, log-LR, C_llr. Keep statistical abbreviations (C_llr, LR, AUC, ECE, Brier) verbatim. Glossary terms `kalibrasyon`, `olabilirlik oranı`.

- [ ] **Step 4: Translate `forensic/topic-invariance.md`**

Follow the **Translation Task Procedure**. Covers Sapkota character n-gram categories and Stamatatos distortion — keep author names verbatim. Technical term "topic-invariant" is "konudan bağımsız" (matches the nav translation).

- [ ] **Step 5: Translate `forensic/evaluation.md`**

Follow the **Translation Task Procedure**. PAN evaluation suite. Metric abbreviations (AUC, c@1, F0.5u, Brier, ECE, C_llr) stay English.

- [ ] **Step 6: Translate `forensic/reporting.md`**

Follow the **Translation Task Procedure**. ENFSI / Nordgaard verbal scale — keep these proper nouns verbatim. The verbal scale entries ("weak support", "moderate support", etc.) SHOULD be translated — they are the user-facing scale labels; include both EN and TR side-by-side in a table so the forensic reader retains the EN anchor.

- [ ] **Step 7: Build strict after each page**

Run `mkdocs build --strict` after each page.

Expected: PASS.

- [ ] **Step 8: Glossary consistency check**

Run:

```bash
grep -hE '\b(stilometri|derlem|yazar tespiti|yazar doğrulama|adli dilbilim|öznitelik|işlev sözcüğü|okunabilirlik|gömme|olabilirlik oranı|kalibrasyon|delil zinciri|köken bilgisi)\b' docs/site/forensic/*.tr.md | sort | uniq -c | sort -rn | head -30
```

Expected: glossary terms dominate; `yazar doğrulama`, `olabilirlik oranı`, `delil zinciri`, `kalibrasyon` should be the top hits.

- [ ] **Step 9: Commit**

```bash
git add docs/site/forensic/*.tr.md
git commit -m "docs(i18n): translate forensic toolkit section to Turkish"
```

---

### Task 7: Translate the `tutorials/` section (4 pages)

**Files:**
- Create: `docs/site/tutorials/index.tr.md`
- Create: `docs/site/tutorials/federalist.tr.md`
- Create: `docs/site/tutorials/pan-clef.tr.md`
- Create: `docs/site/tutorials/turkish.tr.md`

- [ ] **Step 1: Translate `tutorials/index.md`**

Follow the **Translation Task Procedure**. Short index page.

- [ ] **Step 2: Translate `tutorials/federalist.md`**

Follow the **Translation Task Procedure**. Heavy on CLI walkthroughs, `study.yaml` snippets, and matplotlib figures. All code blocks and YAML keys verbatim. Figure captions translated.

- [ ] **Step 3: Translate `tutorials/pan-clef.md`**

Follow the **Translation Task Procedure**. PAN-CLEF is a proper noun. Dataset file names, evaluation script invocations stay verbatim.

- [ ] **Step 4: Translate `tutorials/turkish.md`**

Follow the **Translation Task Procedure**. **Meta-moment:** the English original explains Turkish stylometry to English readers. The Turkish version explains Turkish stylometry to Turkish readers — this lets the meta-commentary ("here's how to use Turkish with bitig") become less meta and more direct in Turkish. Read the whole page first and adjust framing paragraphs accordingly; do not mechanically translate the "as a Turkish reader you already know …" asides.

- [ ] **Step 5: Build strict after each page**

Run `mkdocs build --strict` after each page.

Expected: PASS.

- [ ] **Step 6: Glossary consistency check**

Run:

```bash
grep -hE '\b(stilometri|derlem|yazar tespiti|yazar doğrulama|adli dilbilim|öznitelik|işlev sözcüğü|okunabilirlik|gömme|olabilirlik oranı|kalibrasyon|delil zinciri|köken bilgisi)\b' docs/site/tutorials/*.tr.md | sort | uniq -c | sort -rn | head -30
```

Expected: consistent glossary usage.

- [ ] **Step 7: Commit**

```bash
git add docs/site/tutorials/*.tr.md
git commit -m "docs(i18n): translate tutorials section to Turkish"
```

---

### Task 8: Translate the `reference/` section (4 pages)

**Files:**
- Create: `docs/site/reference/index.tr.md`
- Create: `docs/site/reference/cli.tr.md`
- Create: `docs/site/reference/config.tr.md`
- Create: `docs/site/reference/api.tr.md`

- [ ] **Step 1: Translate `reference/index.md`**

Follow the **Translation Task Procedure**. Section index.

- [ ] **Step 2: Translate `reference/cli.md`**

Follow the **Translation Task Procedure**. Every `bitig <command>` invocation and flag (`--metadata`, `--language`, `--name`, `--output`) stays verbatim. Only the descriptive prose between commands is translated.

- [ ] **Step 3: Translate `reference/config.md`**

Follow the **Translation Task Procedure**. `study.yaml` schema — every key name (`corpus`, `preprocess`, `spacy`, `language`, `model`, `backend`, `features`, `methods`) stays verbatim. Only descriptions of each key are translated.

- [ ] **Step 4: Translate `reference/api.md`**

Follow the **Translation Task Procedure**. This page renders mkdocstrings-generated Python API documentation. The top-level prose introducing the API is translatable; the embedded docstring content comes from `src/bitig/` Python source (English) and is NOT in scope per the spec. That means this page is mostly short — a paragraph of intro above the mkdocstrings autodoc block.

- [ ] **Step 5: Build strict**

Run: `mkdocs build --strict`

Expected: PASS. mkdocstrings still renders the (English) autodoc block — that is intentional.

- [ ] **Step 6: Glossary consistency check**

Run:

```bash
grep -hE '\b(stilometri|derlem|yazar tespiti|yazar doğrulama|adli dilbilim|öznitelik|işlev sözcüğü|okunabilirlik|gömme|olabilirlik oranı|kalibrasyon|delil zinciri|köken bilgisi)\b' docs/site/reference/*.tr.md | sort | uniq -c | sort -rn | head -30
```

Expected: consistent glossary usage.

- [ ] **Step 7: Commit**

```bash
git add docs/site/reference/*.tr.md
git commit -m "docs(i18n): translate reference section to Turkish"
```

---

### Task 9: End-to-end verification + deploy

**Files:** (none — verification only)

- [ ] **Step 1: Full strict build from scratch**

```bash
rm -rf site/
mkdocs build --strict
```

Expected: PASS, no warnings.

- [ ] **Step 2: Verify every EN permalink still exists**

Run:

```bash
for p in \
  index \
  getting-started \
  concepts/index \
  concepts/corpus \
  concepts/features \
  concepts/languages \
  concepts/methods \
  concepts/results \
  forensic/index \
  forensic/verification \
  forensic/calibration \
  forensic/topic-invariance \
  forensic/evaluation \
  forensic/reporting \
  tutorials/index \
  tutorials/federalist \
  tutorials/pan-clef \
  tutorials/turkish \
  reference/index \
  reference/cli \
  reference/config \
  reference/api \
; do
  test -f "site/${p}/index.html" || test -f "site/${p}.html" || echo "MISSING: ${p}"
done
```

Expected: no `MISSING:` lines printed.

- [ ] **Step 3: Verify every TR page exists**

Run the same loop under the `site/tr/` prefix:

```bash
for p in \
  index \
  getting-started \
  concepts/index \
  concepts/corpus \
  concepts/features \
  concepts/languages \
  concepts/methods \
  concepts/results \
  forensic/index \
  forensic/verification \
  forensic/calibration \
  forensic/topic-invariance \
  forensic/evaluation \
  forensic/reporting \
  tutorials/index \
  tutorials/federalist \
  tutorials/pan-clef \
  tutorials/turkish \
  reference/index \
  reference/cli \
  reference/config \
  reference/api \
; do
  test -f "site/tr/${p}/index.html" || test -f "site/tr/${p}.html" || echo "MISSING TR: ${p}"
done
```

Expected: no `MISSING TR:` lines printed.

- [ ] **Step 4: Serve locally and smoke-test language switcher**

Run: `mkdocs serve`

In a browser at `http://127.0.0.1:8000/`:
- Verify English renders and nav reads "Home / Getting started / Concepts / Forensic toolkit / Tutorials / Reference".
- Click the language switcher (Material header, top-right) → select Türkçe.
- Verify URL becomes `http://127.0.0.1:8000/tr/` and nav reads "Ana Sayfa / Başlangıç / Kavramlar / Adli Araç Seti / Öğreticiler / Referans".
- Navigate to `/tr/getting-started/` — page renders in Turkish.
- Navigate to `/tr/concepts/languages/` — page renders in Turkish.
- Use the search box: query `stilometri` → should surface the TR landing page; query `Burrows` → should surface matching pages.

If any of these fail: do not proceed. Fix and re-verify.

- [ ] **Step 5: Push to main and confirm Pages deploy**

```bash
git push origin main
```

Watch the `Docs` workflow on GitHub Actions — it must pass (build + deploy). Once deployed, smoke-test the live URLs:

- `https://fatihbozdag.github.io/bitig/` — EN renders, permalinks work.
- `https://fatihbozdag.github.io/bitig/tr/` — TR renders.
- `https://fatihbozdag.github.io/bitig/getting-started/` — EN permalink unchanged.
- `https://fatihbozdag.github.io/bitig/tr/concepts/languages/` — deep TR link works.
- Language switcher works in both directions.

- [ ] **Step 6: Update the README + docs landing page**

Edit `README.md` and `docs/site/index.md` to reflect multilingual docs in the **Status** section. Add a single line under the existing "Docs site landed" paragraph:

> **Docs site is multilingual** — English (default) and Turkish (`/tr/`) launched; DE/ES/FR infrastructure ready (translation content deferred).

Commit:

```bash
git add README.md docs/site/index.md
git commit -m "docs: note multilingual docs site (EN + TR launched)"
git push origin main
```

- [ ] **Step 7: Final memory update**

Record what shipped to persistent memory. Update `/Users/fatihbozdag/.claude/projects/-Users-fatihbozdag-Downloads-Cursor-Projects-Stylometry/memory/project_scope.md` — note that multilingual docs (EN + TR) landed on 2026-04-20 and the glossary + `mkdocs-static-i18n` infrastructure is in place for DE/ES/FR to be added later as translation-content-only work.

---

## Self-review notes

**Spec coverage check:**

- ✅ URL structure (EN at `/`, TR at `/tr/`) — Task 2 config + Task 9 Step 2/3 verification.
- ✅ `mkdocs-static-i18n` with suffix structure — Task 1 + Task 2.
- ✅ `fallback_to_default: true` — Task 2 Step 1.
- ✅ `reconfigure_material: true` — Task 2 Step 1.
- ✅ `nav_translations` for all 21 labels — Task 2 Step 1.
- ✅ Language-specific search (`search.lang: [en, tr]`) — Task 2 Step 1.
- ✅ Terminology glossary — Task 3; re-referenced in every translation task.
- ✅ All 22 pages translated — Tasks 4–8 (2 + 6 + 6 + 4 + 4 = 22).
- ✅ Code blocks / CLI / API identifiers untranslated — spelled out in Translation Task Procedure + reinforced in per-task notes.
- ✅ Citations / DOIs / brand names untranslated — same.
- ✅ CI unchanged — no workflow edits required; plugin picked up via `pyproject.toml`.
- ✅ Permalink preservation — Task 9 Step 2 hard-gate verification.
- ✅ Native-speaker review — the plan is structured to support commit-per-section PR review (each of Tasks 4–8 ends with its own commit, so the user can review section-by-section).
- ✅ Incremental rollout support — `fallback_to_default: true` handles any intermediate state where a TR page is missing.

**Placeholder scan:** none — every step contains actual YAML, commands, grep patterns, and file paths.

**Type/identifier consistency:** plugin name `mkdocs-static-i18n` and plugin key `i18n` match the project's convention (plugin-key is the short name, package-name is the pip name). Nav labels in `nav_translations` exactly match the existing `nav:` labels in `mkdocs.yml`. Glossary terms on the shared reference and in every grep check match character-for-character.
