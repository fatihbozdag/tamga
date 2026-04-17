# tamga

**Next-generation computational stylometry — a Python replacement for R's Stylo.**

`tamga` ("mark, brand, clan-sign" — from Old Turkic) is a Python package and interactive CLI for
authorship attribution, author-group style comparison, and Digital Humanities stylometric
analysis. It reimplements the analytical breadth of R's `Stylo` and adds modern NLP and ML on top.

> Named after the **tamga**, the Turkic clan-mark by which individual and familial identity was
> recognised at a glance — the material-culture counterpart to a stylistic fingerprint.

## Status

**Phase 2 — Features & Delta.** Ships feature extractors (MFW, char/word/POS n-grams,
dependency bigrams, function words, punctuation, lexical diversity, readability, sentence length)
and the full Delta family (Burrows, Eder, Eder-Simple, Argamon Linear, Cosine, Quadratic).
`tamga features` and `tamga delta` CLI commands work end-to-end; the Federalist Papers parity
test attributes the disputed paper 49 to Madison using Burrows/Eder/Cosine Delta on 500 MFW.

Phases 3 (Zeta/reducers/clustering/consensus/classify), 4 (embeddings + Bayesian),
5 (viz + reports + wizard shell), and 6 (docs + PyPI) remain.

See `docs/superpowers/specs/2026-04-17-tamga-stylometry-package-design.md` for the full design.

## Install

```bash
uv pip install tamga
python -m spacy download en_core_web_trf
```

## Quickstart

```bash
tamga init my-study
cd my-study
# drop .txt files into corpus/
# optionally add a metadata.tsv with filename → author/group/year/...
tamga ingest corpus/ --metadata corpus/metadata.tsv
tamga info
```

## License

BSD-3-Clause. See `LICENSE`.

## Citation

If you use tamga in published work, please cite it — see `CITATION.cff`.
