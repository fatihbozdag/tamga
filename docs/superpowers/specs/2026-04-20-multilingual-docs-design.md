# Multilingual documentation site — design (2026-04-20)

## Goal

Turn the bitig MkDocs Material site into a multilingual site, launching with English
(default) and Turkish. Lay down infrastructure so German, Spanish, and French can be added
later without further refactors. The analyzer advertises EN/TR/DE/ES/FR as first-class
languages — the docs site is the one user-facing surface where a half-finished multilingual
story would directly undermine that claim, so the Turkish rollout must read as native.

## Scope

- **Languages shipped now:** English (default) and Turkish.
- **Languages deferred:** German, Spanish, French — infrastructure is identical; adding them
  is a translation-content task, not an engineering task.
- **Pages to translate now:** all 22 markdown pages under `docs/site/`.
- **Not in scope:** README translation, CHANGELOG translation, docstring translation in
  `src/bitig/`, CLI help-text localization. These stay English; the package API and tool
  output remain English (see *Translation policy* below).

## Constraints

- Must preserve every existing public URL (`https://fatihbozdag.github.io/bitig/...`) —
  breaking permalinks costs reputational capital and inbound links on a freshly published
  site. English stays at the root; Turkish goes to `/tr/`.
- Must keep `mkdocs build --strict` passing in CI.
- Must not add a second build or deploy workflow — one `Docs` workflow, one artifact, one
  Pages deploy.
- Turkish rendering must read as native Turkish, not machine-translated Turkish. The user
  (native speaker, linguistics researcher) reviews every page.
- Brand and API names stay untranslated: `bitig`, `Corpus`, `Delta`, `Zeta`, `Burrows`,
  `PANReport`, CLI flags, YAML keys, Python identifiers.

## Approach

**Plugin: `mkdocs-static-i18n` (suffix structure).** This is the de-facto MkDocs i18n
plugin and integrates with Material theme's built-in language switcher via `extra.alternate`.
Suffix structure (`page.md` for default, `page.tr.md` for Turkish) keeps each page
co-located with its translation — easier to spot drift than a separate directory tree.

**Translation workflow:** Claude drafts Turkish from the English source; the user reviews
each TR page as the native speaker and commits corrections. For future English updates, the
same loop runs on the diff. A terminology glossary (`docs/site/_translations/tr-glossary.md`)
pins the Turkish rendering of recurring technical terms so translations stay consistent
across pages and across updates.

## Architecture

### URL structure

| Language | URL prefix | Example |
|---|---|---|
| English (default) | `/` | `https://fatihbozdag.github.io/bitig/getting-started/` |
| Turkish | `/tr/` | `https://fatihbozdag.github.io/bitig/tr/getting-started/` |

Default-at-root means every existing permalink keeps working. The language switcher in the
Material header offers a toggle between the two.

### Source layout

```
docs/site/
├── index.md              ← EN (default, existing)
├── index.tr.md           ← TR
├── getting-started.md
├── getting-started.tr.md
├── concepts/
│   ├── index.md
│   ├── index.tr.md
│   ├── corpus.md
│   ├── corpus.tr.md
│   └── ...
├── forensic/             ← same pattern
├── tutorials/            ← same pattern
├── reference/            ← same pattern
└── _translations/
    └── tr-glossary.md    ← terminology pin (not rendered; excluded from nav)
```

22 EN pages + 22 TR pages + 1 glossary = 45 files under `docs/site/` after rollout.

### mkdocs.yml changes

1. Add `mkdocs-static-i18n` to `plugins:` before `mkdocstrings`, configure:
   - `docs_structure: suffix`
   - `fallback_to_default: true` (untranslated pages fall back to EN — safety net during
     incremental rollout)
   - `languages:` list with `en` (default, `build: true`) and `tr` (`build: true`)
   - `nav_translations:` per language — ~30 labels (top-level tabs + section indexes).
   - `reconfigure_material: true` — lets the plugin wire Material's `alternate` list
     automatically.
2. Add `extra.alternate:` — the language switcher data. The i18n plugin generates this
   automatically when `reconfigure_material: true`; we leave `extra.alternate` unset in
   `mkdocs.yml`.
3. Language-specific search: `plugins.search.lang: [en, tr]` — Material's Lunr search
   already supports Turkish stemming via the `tr` language pack.

### pyproject.toml changes

Add `mkdocs-static-i18n>=1.2` to the `docs` extra.

### Translation policy (what translates vs. what stays English)

| Element | Policy |
|---|---|
| Prose paragraphs, headings, captions, admonition bodies | Translate |
| Navigation labels, tab titles | Translate (`nav_translations`) |
| Code blocks (Python, YAML, shell) | Do **not** translate |
| CLI command examples (`bitig ingest ...`) | Do **not** translate — tool output is English |
| Python identifiers, CLI flags, YAML keys | Do **not** translate |
| Citations, bibliography, URLs, DOIs | Do **not** translate |
| The word "bitig" and method names (Burrows, Delta, Zeta, General Impostors, Unmasking) | Do **not** translate — proper nouns |
| Mermaid diagram node labels | Translate if the label is prose; keep English if the label is an API name |
| Image alt text and figure captions | Translate |

### Terminology glossary (tr-glossary.md)

A pinned translation table for recurring technical terms. Initial entries:

| English | Turkish | Notes |
|---|---|---|
| stylometry | *stilometri* | Established loan; simpler than "üslup bilim"; matches how computational-linguistics literature in Turkish refers to the field |
| corpus | *derlem* | Standard TDK term |
| authorship attribution | *yazar tespiti* | Preferred over "yazar atfı" for naturalness |
| authorship verification | *yazar doğrulama* | Forensic sub-task |
| forensic linguistics | *adli dilbilim* | Standard |
| feature (stylometric) | *öznitelik* | Standard ML term |
| function word | *işlev sözcüğü* | Corpus linguistics standard |
| readability | *okunabilirlik* | Standard |
| classifier / classification | *sınıflandırıcı / sınıflandırma* | Standard ML |
| clustering | *kümeleme* | Standard ML |
| embedding (vector) | *gömme* | Standard ML |
| likelihood ratio | *olabilirlik oranı* | Stats standard |
| calibration | *kalibrasyon* | Stats loan |
| chain of custody | *delil zinciri* | Forensic/legal standard |
| provenance | *köken bilgisi* | Keeps "provenance" feel; literal "soy" sounds odd for data |

The user will expand this list during review; consistency is enforced by grep before each
commit.

### CI

`.github/workflows/docs.yml` already builds with `mkdocs build --strict`. The i18n plugin is
strict-safe by default. No workflow-level changes needed; only `pyproject.toml`'s `docs`
extra needs to include the plugin so the `uv pip install` step picks it up.

## Build sequence

1. **Infrastructure** — add plugin dep, configure `mkdocs.yml`, verify `mkdocs build
   --strict` passes with a single `index.tr.md` stub present and a second stub page in each
   top-level section. Confirm the language switcher renders, both `/` and `/tr/` resolve,
   and EN permalinks are unchanged.
2. **Terminology glossary** — seed `docs/site/_translations/tr-glossary.md` from the table
   above. Add a `not_in_nav` exclusion so the glossary never appears in the public nav.
3. **Landing + getting-started** — translate `index.tr.md` and `getting-started.tr.md`
   first; these are the highest-traffic pages and the worst place for rough Turkish. User
   review gate before moving on.
4. **Concepts** — 6 pages. Includes `concepts/languages.md`, which is the page most readers
   interested in the multilingual story will land on.
5. **Forensic toolkit** — 6 pages. Densest technical terminology; glossary does heavy
   lifting here.
6. **Tutorials** — 4 pages. Includes `tutorials/turkish.md`, which needs the most careful
   Turkish because the English original discusses Turkish as the subject matter.
7. **Reference** — 4 pages. CLI reference and API reference are mostly code; prose volume
   is lowest.
8. **Deploy** — merge to `main`; confirm GitHub Pages redeploy picks up both language
   trees; smoke-test `/tr/` end-to-end in a browser.

Each batch lands as a separate PR so the user's review surface stays bounded.

## Risks and mitigations

- **Machine-translation feel.** Mitigated by native-speaker review gate and the terminology
  glossary.
- **Translation drift as English evolves.** Mitigated by `fallback_to_default: true` (a
  stale TR page is better than a 404) and by running the same draft-and-review loop on every
  English-side diff.
- **Permalink regression.** Mitigated by keeping EN at `/`, verified via a smoke test on the
  live Pages site after first deploy.
- **Search ranking.** Material's Lunr `tr` stemmer is not as good as the EN stemmer; users
  searching TR may get rougher results. Acceptable trade-off — the alternative (ship no
  search) is worse.

## Success criteria

- `mkdocs build --strict` passes both locally and in CI.
- Live site shows a working language switcher in the Material header.
- Every existing EN URL continues to resolve (sample: `/getting-started/`, `/concepts/`,
  `/forensic/`, `/tutorials/federalist/`, `/reference/cli/`).
- `/tr/` tree renders all 22 pages with translated prose and translated nav labels.
- Turkish search box finds `stilometri`, `derlem`, `yazar tespiti`.
- Glossary terms are applied consistently — `grep -r "authorship attribution" docs/site
  --include='*.tr.md'` returns nothing; `grep -r "yazar tespiti" docs/site
  --include='*.tr.md'` returns the expected hits.
