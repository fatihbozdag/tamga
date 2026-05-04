# Turkish stylometry walkthrough

A complete runnable example: attribute a small Turkish short-story corpus using MFW +
Ateşman readability + Burrows Delta. The stories are public-domain Ömer Seyfettin texts from
Turkish Wikisource.

## Setup

```bash
uv pip install 'bitig[turkish]'
python -c "import stanza; stanza.download('tr')"
bitig init seyfettin --language tr
cd seyfettin
```

This scaffolds a project directory with `study.yaml` pre-configured for Turkish. Confirm
with:

```bash
bitig info
```

The `language` row shows `tr`.

## Corpus

Place 3-5 Turkish short stories in `corpus/` as UTF-8 `.txt` files. A good public-domain
source is [Ömer Seyfettin on Turkish
Wikisource](https://tr.wikisource.org/wiki/Yazar:%C3%96mer_Seyfettin) — dozens of early-20th-
century short stories are already transcribed there.

Add a `corpus/metadata.tsv`:

```tsv
filename	author	year
bomba.txt	Omer_Seyfettin	1910
kesik_biyik.txt	Omer_Seyfettin	1911
forsa.txt	Omer_Seyfettin	1913
pembe_incili_kaftan.txt	Omer_Seyfettin	1917
```

A real study would include several authors. For a single-author demo, pair Seyfettin with
a few stories by Refik Halit Karay (also public domain) to give Delta something to
discriminate.

## Running the study

```bash
bitig ingest corpus/ --language tr --metadata corpus/metadata.tsv
bitig run study.yaml --name first-run
```

`bitig ingest` runs Stanza through `spacy-stanza`. The first run parses every document and
caches the DocBins; subsequent runs hit the cache and finish in seconds.

## What you get

A default Turkish study computes:

- **MFW** (top 1000 tokens, z-scored relative frequencies)
- **Turkish function words** — loaded from `resources/languages/tr/function_words.txt`,
  derived from UD Turkish BOUN closed-class tokens
- **Ateşman and Bezirci-Yılmaz** readability indices
- **Burrows Delta** + PCA/MDS reduction plots

The output folder `results/first-run/` contains:

- `result.json` with Delta scores and provenance
- `table_*.parquet` feature matrices
- PNG / PDF figures (distance heatmap, PCA scatter)
- a `provenance.json` that records the corpus hash, seed, and full resolved config

## Customising

Edit `study.yaml` to swap features or methods. For example, to use contextual embeddings
instead of MFW:

```yaml
features:
  - id: bert_tr
    type: contextual_embedding
    # model auto-resolves to `dbmdz/bert-base-turkish-cased` via the language registry
    pool: mean
```

For a heavier Turkish encoder, point `model:` at any HuggingFace checkpoint:

```yaml
features:
  - id: bert5urk
    type: contextual_embedding
    model: stefan-it/bert5urk
    pool: mean
```

## Notes on Turkish specifics

- **Morphology.** Turkish is agglutinative; a token like `evlerinizden` packs
  `ev+ler+iniz+den` into one form. Stanza's Turkish BOUN model lemmatises and tags these
  correctly, which matters for POS-ngram and dependency-based features.
- **Syllable counting.** Both Ateşman and Bezirci-Yılmaz count syllables using a
  vowel-counter specialised for Turkish orthography (including `ı`, `ğ`, `ş`, `ç`, `ü`,
  `ö`).
- **Function words.** The bundled list leans on Turkish's closed-class postpositions,
  conjunctions, and discourse particles (e.g. `ile`, `ancak`, `fakat`, `çünkü`, `ki`,
  `ise`).

## Troubleshooting

- **`ModuleNotFoundError: No module named 'spacy_stanza'`** — run
  `uv pip install 'bitig[turkish]'`.
- **`FileNotFoundError: ... stanza_resources/tr/default.zip`** — run
  `python -c "import stanza; stanza.download('tr')"`. The model is about 600 MB.
- **Very slow first ingest on MPS.** Stanza's Turkish model does not yet support Apple
  Silicon MPS. Expect CPU parse rates on first run; subsequent runs are cache hits.
