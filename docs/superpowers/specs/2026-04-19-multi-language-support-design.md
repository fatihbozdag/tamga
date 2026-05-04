# Multi-language support — design

**Status:** design approved, awaiting implementation plan
**Author:** bitig maintainers
**Date:** 2026-04-19
**Scope:** five first-class languages — English, Turkish, German, Spanish, French — behind a single language-aware interface. Expansion pattern documented for additional languages.

## Problem

bitig today is English-only. Five sites hard-code that assumption:

1. `SpacyConfig.model` defaults to `en_core_web_trf`.
2. `FunctionWordExtractor` loads a single bundled `function_words_en.txt`.
3. `ReadabilityExtractor` wraps `textstat` formulas calibrated to English syllable counts (Flesch, Flesch-Kincaid, Gunning Fog, Coleman-Liau, ARI, SMOG).
4. Contextual and sentence embedding extractors have English-oriented default models.
5. `NormalizeConfig.expand_contractions` is an English-specific option.

Other extractors (MFW, char/word n-grams, punctuation, lexical diversity, sentence length) are already language-agnostic. POS n-grams and dependency bigrams work cross-lingually via Universal Dependencies tags.

The goal is multi-language support that treats Turkish as a first-class citizen, not a second-class bolt-on, and does so without forking the codebase into per-language branches.

## Non-goals

- **Code-mixed corpora.** Within a single study, every document is in the same language. Mixed-language corpora would require per-document language tracking plus cross-lingual Delta — a separate design.
- **Automatic language detection** as a default. An optional `bitig.corpus.detect_language()` convenience helper may ship, but ingestion requires the language to be declared explicitly.
- **Claiming support for every spaCy language.** Only the five named above are first-class (bundled resources, tested, tutorialised). Additional languages can be added by following the documented pattern.
- **Alternative NLP backends beyond spaCy-compatible ones.** Stanza is consumed through `spacy-stanza`, which returns native spaCy `Doc` objects — no separate `Doc` abstraction is introduced.

## Selected models and readability formulas

| Language | spaCy pipeline / Stanza lang | Backend | Contextual embedding | Sentence embedding | Readability formulas |
|---|---|---|---|---|---|
| English (en) | `en_core_web_trf` | native spaCy | `bert-base-uncased` | `sentence-transformers/all-mpnet-base-v2` | Flesch, Flesch-Kincaid, Gunning Fog, Coleman-Liau, ARI, SMOG |
| Turkish (tr) | Stanza `tr` (BOUN) | `spacy-stanza` | `dbmdz/bert-base-turkish-cased` | `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` | Ateşman (1997), Bezirci–Yılmaz (2010) |
| German (de) | `de_dep_news_trf` | native spaCy | `deepset/gbert-base` | `deepset/gbert-base-sts` | Flesch-Amstad (1978), Wiener Sachtextformel (Bamberger & Vanecek 1984) |
| Spanish (es) | `es_dep_news_trf` | native spaCy | `dccuchile/bert-base-spanish-wwm-cased` (BETO) | `hiiamsid/sentence_similarity_spanish_es` | Fernández-Huerta (1959), Szigriszt-Pazos / INFLESZ (1992) |
| French (fr) | `fr_dep_news_trf` | native spaCy | `almanach/camembert-base` | `dangvantuan/sentence-camembert-base` | Kandel–Moles (1958), LIX (Björnsson 1968) |

**Rationale:**

- **Turkish via Stanza, not turkish-nlp-suite.** The community `tr_core_news_trf` pipeline is pinned to `spacy>=3.4.2,<3.5.0` and incompatible with bitig's `spacy>=3.7`. Stanford Stanza's Turkish BOUN pipeline is actively maintained, canonical in Turkish NLP research, and is exposed through a native spaCy interface by the `spacy-stanza` package.
- **Contextual embeddings are all ~110 M BERT-family models.** Stylometric signal lives in token-level representations; scaling to 1.4 B-parameter models (BERT5urk, CamemBERT-large, XLM-large) gives diminishing returns on feature quality while multiplying VRAM cost. Heavyweight alternatives are documented as overrides, not defaults.
- **Function-word lists are generated, not curated.** Each list is derived programmatically from the language's UD treebank(s), counting closed-class tokens (UPOS ∈ {DET, PRON, ADP, CCONJ, SCONJ, AUX, PART}) by frequency. This avoids the "whose canon do we use" problem, stays license-clean (UD is CC-BY-SA), and gives methodologically uniform lists across all five languages. A regeneration script ships alongside.
- **Non-English readability formulas are implemented natively.** `textstat`'s per-language support is partial (Turkish is absent; DE/FR syllable counters are approximate). Implementing the source-paper formulas directly in `bitig.languages.readability_*` gives unit-testable closed-form calculations with known reference values from the originating papers.

## Architecture

A new module `bitig.languages` owns a registry of `LanguageSpec` dataclasses. Every language-dependent site reads from this registry. `Corpus` gains a `language` attribute stamped at ingestion. `StudyConfig.preprocess` gains a `language:` field that cascades as the resolution root for every downstream default.

```
src/bitig/languages/
├── __init__.py            # re-exports get(), REGISTRY, LanguageSpec
├── registry.py            # dataclass, REGISTRY dict, get()
├── readability_tr.py      # Ateşman, Bezirci–Yılmaz (native)
├── readability_de.py      # Wiener Sachtextformel, Flesch-Amstad (native)
├── readability_es.py      # Fernández-Huerta, Szigriszt-Pazos (native)
└── readability_fr.py      # Kandel–Moles, LIX (native)

src/bitig/resources/languages/
├── en/function_words.txt  # moved from src/bitig/resources/function_words_en.txt
├── tr/function_words.txt  # generated from UD BOUN
├── de/function_words.txt  # generated from UD GSD + HDT
├── es/function_words.txt  # generated from UD AnCora + GSD
└── fr/function_words.txt  # generated from UD GSD + Sequoia

scripts/
└── regenerate_function_words.py  # reproducible generator

src/bitig/preprocess/pipeline.py   # gains backend="spacy" | "spacy_stanza"
```

Two backends behind a single interface — `spacy.load()` for native pipelines, `spacy_stanza.load_pipeline()` for Stanza-wrapped ones. Both return a `spacy.Language` object and produce native `Doc` objects, so every downstream feature extractor remains untouched. The DocBin cache key incorporates a backend identifier; English caches built on a prior bitig version remain valid because the native-spaCy branch of the key preserves the existing format.

## Components

### `bitig.languages.LanguageSpec` + `REGISTRY`

```python
@dataclass(frozen=True)
class LanguageSpec:
    code: str                                 # "tr"
    name: str                                 # "Turkish"
    default_model: str                        # spaCy pipeline name or Stanza lang code
    backend: Literal["spacy", "spacy_stanza"]
    readability_indices: tuple[str, ...]      # canonical default set
    contextual_embedding_default: str         # HF id
    sentence_embedding_default: str           # HF / sentence-transformers id

REGISTRY: dict[str, LanguageSpec] = {
    "en": LanguageSpec(...),
    "tr": LanguageSpec(..., backend="spacy_stanza", default_model="tr"),
    "de": LanguageSpec(...),
    "es": LanguageSpec(...),
    "fr": LanguageSpec(...),
}

def get(code: str) -> LanguageSpec: ...
```

### Config schema

```python
class PreprocessConfig(BaseModel):
    language: str = "en"                                  # NEW
    spacy: SpacyConfig = Field(default_factory=SpacyConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)

class SpacyConfig(BaseModel):
    model: str | None = None                              # was defaulted; now resolved from language
    backend: Literal["spacy", "spacy_stanza"] | None = None  # NEW; resolved from language
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    exclude: list[str] = Field(default_factory=list)
```

Unknown language codes raise at pydantic validation. `NormalizeConfig.expand_contractions` is retained but documented as English-only (no-op on other languages).

### `Corpus` gains `language: str = "en"`

Stamped at ingestion time. `Corpus.from_directory(..., language="tr")` and `bitig ingest corpus/ --language tr`. Default `"en"` keeps existing call sites working without changes. An optional helper `bitig.corpus.detect_language(corpus) -> str` (using `langdetect` or `fasttext`) is available for users who want automatic detection — never run implicitly.

### Feature extractor changes — all additive and back-compatible

- **`FunctionWordExtractor`** gains `language: str | None = None`. When unset, resolves from `corpus.language`. Loads `resources/languages/<lang>/function_words.txt`. Missing list → clear `FileNotFoundError` with supported-languages list and `wordlist=[...]` override hint.
- **`ReadabilityExtractor`** gains `language: str | None = None`. Indices are registered per-language (`_INDEX_REGISTRY: dict[str, dict[str, Callable]]`). `indices=None` → use the language's canonical default set from the registry. Requesting an unsupported index for a language (e.g. `flesch` for Turkish) raises `ValueError` listing available indices for that language.
- **`ContextualEmbeddingExtractor`** and **`SentenceEmbeddingExtractor`** gain `language: str | None = None`. `model_name=None` → look up per-language default from the registry.
- **`MFWExtractor`, `CharNgramExtractor`, `WordNgramExtractor`, `PunctuationExtractor`, `LexicalDiversityExtractor`, `SentenceLengthExtractor`, `PosNgramExtractor`, `DependencyBigramExtractor`** — no API changes. They work on whatever `Doc` the pipeline produces.

### `SpacyPipeline`

```python
class SpacyPipeline:
    def __init__(
        self,
        *,
        language: str = "en",
        model: str | None = None,
        backend: Literal["spacy", "spacy_stanza"] | None = None,
        cache_dir: Path | str = ".bitig/cache/docbin",
        exclude: list[str] | None = None,
    ) -> None:
        spec = bitig.languages.get(language)
        self.language = language
        self.model = model or spec.default_model
        self.backend = backend or spec.backend
        ...

    @property
    def nlp(self) -> Language:
        if self._nlp is None:
            if self.backend == "spacy_stanza":
                if self.exclude:
                    _log.warning("exclude=%s ignored on spacy_stanza backend", self.exclude)
                import spacy_stanza
                self._nlp = spacy_stanza.load_pipeline(lang=self.model)
            else:
                self._nlp = spacy.load(self.model, exclude=self.exclude)
        return self._nlp

    @property
    def backend_version(self) -> str:
        if self.backend == "spacy_stanza":
            import spacy_stanza, stanza
            return f"spacy_stanza={spacy_stanza.__version__};stanza={stanza.__version__}"
        return f"spacy={spacy.__version__}"  # preserves existing cache-key format
```

`DocBinCache.cache_key()` replaces its `spacy_version` argument with `backend_version`. The native-spaCy branch emits exactly the previous string format, preserving existing English caches across the upgrade.

### Function-word list generation

`scripts/regenerate_function_words.py`:

1. Loads a language's UD treebank(s) (documented list per language).
2. Iterates all tokens, keeps those with UPOS ∈ {DET, PRON, ADP, CCONJ, SCONJ, AUX, PART}.
3. Counts token frequencies (case-normalised), sorts descending.
4. Takes the top 200 (matching the existing English list size).
5. Writes `resources/languages/<lang>/function_words.txt` with a commit-block header: source treebank identifiers, UD release version, generation date, script commit SHA.

Reproducible, license-clean, methodologically uniform across all languages.

### Dependency changes

- **Core** — no change to existing deps; add `pyphen>=0.14` (pure Python, ~1 MB) for DE/FR syllable counting.
- **New optional extra `turkish`** — `spacy-stanza>=1.0.4`, `stanza>=1.6`.
- **Umbrella extra `multilang`** — includes `turkish` plus any future per-language extras.

### CLI surface

- `bitig init <dir> --language tr` — scaffolds `study.yaml` with `preprocess.language: tr`.
- `bitig ingest corpus/ --language tr --metadata …` — stamps `Corpus.language`.
- `bitig info` — prints configured language alongside existing corpus stats.

### Public Python API

`bitig/__init__.py` re-exports:

```python
from bitig.languages import REGISTRY as LANGUAGES, LanguageSpec, get as get_language
```

## Data flow

```
study.yaml: preprocess.language: tr
        │
        ▼
Corpus(language="tr")          ← stamped at ingest time
        │
        ▼
SpacyPipeline(language="tr")
  → registry.get("tr") → backend="spacy_stanza", model="tr"
  → spacy_stanza.load_pipeline(lang="tr")
  → ParsedCorpus (native spaCy Doc objects)
        │
        ▼
FunctionWordExtractor()    loads resources/languages/tr/function_words.txt
ReadabilityExtractor()     picks (atesman, bezirci_yilmaz) as defaults
ContextualEmbeddingExtractor()  uses dbmdz/bert-base-turkish-cased
        ↑ resolved from corpus.language unless explicitly overridden
```

Language is declared once at ingestion and cascades through pipeline and extractors via `corpus.language` unless an extractor receives an explicit `language=` override.

**Config-resolution precedence** (highest → lowest):
extractor's explicit `language=` → `SpacyConfig.backend`/`model` → `PreprocessConfig.language` → `Corpus.language` → `"en"` default.

## Error handling

| Condition | Behavior |
|---|---|
| Unknown language code | `ValueError` at registry lookup / pydantic validation, with supported-languages list |
| `language="tr"` but `spacy-stanza` not installed | `ImportError("bitig requires spacy-stanza for Turkish. Install with: uv pip install 'bitig[turkish]'")` |
| Stanza model not downloaded | `RuntimeError("Stanza model for 'tr' not found. Run: python -c \"import stanza; stanza.download('tr')\"")` |
| No bundled function-word list for a language | `FileNotFoundError` listing supported languages, with `wordlist=[...]` override hint |
| Extractor's `language` ≠ `corpus.language` | Warning log; does not block (intentional override use-case) |
| `preprocess.language` ≠ corpus's stamped language | Hard `ValueError` at study runtime |
| Readability index not supported for language (e.g. `flesch` on Turkish) | `ValueError` listing that language's available indices |
| `SpacyConfig.exclude=[...]` on spacy_stanza backend | Warning log; exclude is a no-op on this backend |

## Cache-invalidation strategy

DocBin cache keys gain a backend segment. The native-spaCy branch emits exactly the previous string format (`"spacy=3.7.2"`, matching prior cache-key bytes), so existing English caches remain valid across the upgrade. Only new non-English studies build fresh caches. Cross-backend collisions are impossible because the keys differ structurally (`"spacy_stanza=…;stanza=…"`).

## Testing

### Unit tests — fast, runs in every CI pass

- Registry: all five codes resolve; unknown code raises; `LanguageSpec` is frozen.
- Function-word list loads for each of five languages; file nonempty; deterministic contents.
- Each non-English readability formula against a **published reference value** from its source paper:
  - Ateşman (1997) worked example
  - Bezirci–Yılmaz (2010) reference text
  - Flesch-Amstad (1978) sample
  - Wiener Sachtextformel (Bamberger & Vanecek 1984)
  - Fernández-Huerta (1959)
  - Szigriszt-Pazos (1992)
  - Kandel-Moles (1958)
  - LIX (Björnsson 1968)

  Tolerance ±1.0 on the index scale.

- `SpacyPipeline` backend resolution: `language="tr"` → `backend="spacy_stanza"`; `language="en"` → `backend="spacy"`; explicit `backend=` overrides.
- Cache key: `backend="spacy"` emits byte-exact pre-upgrade format (regression test); `backend="spacy_stanza"` emits new format; no cross-backend collision.
- `FunctionWordExtractor`, `ReadabilityExtractor`, `ContextualEmbeddingExtractor`: `language=None` resolves from `corpus.language`; explicit `language=` overrides.
- Config validation: `preprocess.language: xx` (unknown) → pydantic error; `language: tr` with explicit `spacy.model: custom` respected.
- `spacy_stanza` backend path: `exclude=[...]` emits a warning log.

### Integration tests — gated, runs on tags + weekly cron

- Turkish end-to-end: small synthetic Turkish corpus (3 docs) → spacy-stanza parse → MFW + function words + Ateşman readability → Burrows Delta → result artifacts. Requires `stanza.download('tr')` in CI setup.
- Same structure for DE, ES, FR using official spaCy `_trf` pipelines (downloaded via `python -m spacy download`).
- Cross-language smoke: EN corpus + TR corpus loaded in the same process; verify cache keys do not collide and extractors do not leak state.

Marked `@pytest.mark.slow`; skipped by default in local runs and PR CI. A new workflow `tests-multilang.yml` runs on release tags and weekly schedule.

### Tutorial validation

- `docs/site/tutorials/turkish.md` — Turkish stylometry walkthrough using a public-domain corpus (Ömer Seyfettin short stories from Turkish Wikisource). Runnable end-to-end; verified as part of the multi-lang CI workflow.
- `docs/site/concepts/languages.md` — overview of multi-language architecture, registry, per-language resource layout, extension pattern.
- Updates to `concepts/corpus.md`, `concepts/features.md`, `reference/config.md`.

## Backward compatibility

| Existing usage | New behavior |
|---|---|
| `bitig ingest corpus/` (no `--language`) | `Corpus.language = "en"`; unchanged from before |
| Existing `study.yaml` (no `preprocess.language`) | Defaults to `"en"`; unchanged |
| `MFWExtractor(n=200)` no language param | Works unchanged (language-agnostic) |
| `FunctionWordExtractor()` (no language) | Uses `corpus.language` → `"en"` → loads English list; same result as before |
| Existing DocBin caches on English corpora | Preserved — native-spaCy cache-key format unchanged |
| `SpacyConfig.model: en_core_web_trf` in existing study.yaml | Still accepted; explicit override wins |

Only user-visible change for existing English workflows: `FunctionWordExtractor` now reads from `resources/languages/en/function_words.txt` (moved, same contents).

## Phase / build sequence

The design divides into six implementation steps, producible in order without breaking existing tests at any step:

1. **Language registry + resource layout.** Create `bitig.languages` with the registry. Move `function_words_en.txt` to `resources/languages/en/function_words.txt` and update `FunctionWordExtractor._load_bundled_list()` to read from the new path (keeping its current English-only behavior). Unit-test the registry. At this point nothing else changes in feature behavior; the existing English test suite stays green.
2. **Corpus.language + config schema.** Add `Corpus.language: str = "en"` stamped at ingestion (`bitig ingest --language`, `Corpus.from_directory(..., language=...)`). Add `PreprocessConfig.language: str = "en"` with pydantic validation against the registry. No extractor behavior changes yet.
3. **spacy-stanza backend in SpacyPipeline.** New optional extra `turkish`. `SpacyPipeline` gains `language`, `backend`, and `backend_version`; dispatch between `spacy.load()` and `spacy_stanza.load_pipeline()`. Backend-aware cache-key format with an English byte-exact regression test to confirm prior caches remain valid.
4. **Extractors pick up language.** `FunctionWordExtractor`, `ReadabilityExtractor`, `ContextualEmbeddingExtractor`, `SentenceEmbeddingExtractor` gain `language: str | None = None`, resolve from `corpus.language` when unset, accept explicit overrides.
5. **Non-English resources.** Native readability formulas (TR, DE, ES, FR) with paper-referenced unit tests; generated function-word lists committed with provenance headers; `scripts/regenerate_function_words.py`. At this point all five languages are fully usable end-to-end.
6. **Tutorial, docs, CLI flags polish, integration workflow.** `bitig init/ingest --language` wiring (if not already in step 2), `bitig info` displays configured language, Turkish tutorial page, `concepts/languages.md`, `tests-multilang.yml` CI workflow for gated model-download integration tests.

Each step is independently mergeable and produces a releasable state.

## References

- Ateşman, E. (1997). Türkçede okunabilirliğin ölçülmesi. *Dil Dergisi*, 58, 71–74.
- Bamberger, R., & Vanecek, E. (1984). *Lesen — Verstehen — Lernen — Schreiben: Die Schwierigkeitsstufen von Texten in deutscher Sprache*. Wien: Jugend und Volk.
- Bezirci, B., & Yılmaz, A. E. (2010). Metinlerin okunabilirliğinin ölçülmesi üzerine bir yazılım kütüphanesi ve Türkçe için yeni bir okunabilirlik ölçütü. *Dokuz Eylül Üniversitesi Mühendislik Fakültesi Fen ve Mühendislik Dergisi*, 12(3), 49–62.
- Björnsson, C. H. (1968). *Läsbarhet*. Stockholm: Liber.
- Fernández-Huerta, J. (1959). Medidas sencillas de lecturabilidad. *Consigna*, 214, 29–32.
- Flesch, R. (1948). A new readability yardstick. *Journal of Applied Psychology*, 32(3), 221–233.
- Kandel, L., & Moles, A. (1958). Application de l'indice de Flesch à la langue française. *Cahiers d'études de radio-télévision*, 19, 253–274.
- Szigriszt-Pazos, F. (1992). *Sistemas predictivos de legibilidad del mensaje escrito: fórmula de perspicuidad*. Doctoral dissertation, Universidad Complutense de Madrid.
- Turkish NLP Suite. (2023). Turkish spaCy models. [turkish-nlp-suite/turkish-spacy-models](https://github.com/turkish-nlp-suite/turkish-spacy-models).
- Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. *ACL 2020 System Demonstrations*.
- Universal Dependencies (v2.13). <https://universaldependencies.org/>.
