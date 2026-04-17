# tamga

**Next-generation computational stylometry — a Python replacement for R's Stylo.**

`tamga` ("mark, brand, clan-sign" — from Old Turkic) is a Python package and interactive CLI for
authorship attribution, author-group style comparison, and Digital Humanities stylometric
analysis. It reimplements the analytical breadth of R's `Stylo` and adds modern NLP and ML on top.

> Named after the **tamga**, the Turkic clan-mark by which individual and familial identity was
> recognised at a glance — the material-culture counterpart to a stylistic fingerprint.

## Status

**Phase 1 — Foundation.** Currently: corpus ingestion + spaCy parsing + DocBin caching +
project scaffolding + skeleton CLI. Delta family, Zeta, consensus trees, classifiers,
visualisations, reports, and the interactive wizard shell land in Phases 2–5.

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
