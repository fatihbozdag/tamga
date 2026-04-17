# tamga

**Next-generation computational stylometry — a Python replacement for R's Stylo.**

`tamga` ("mark, brand, clan-sign" — from Old Turkic) is a Python package and interactive CLI for
authorship attribution, author-group style comparison, and Digital Humanities stylometric
analysis. It reimplements the analytical breadth of R's `Stylo` and adds modern NLP and ML on top.

> Named after the **tamga**, the Turkic clan-mark by which individual and familial identity was
> recognised at a glance — the material-culture counterpart to a stylistic fingerprint.

## Status

**Phase 4 — Optional extras.** `tamga[embeddings]` adds sentence-transformer + contextual-BERT
embeddings (pool mean/cls/max). `tamga[bayesian]` adds Wallace-Mosteller Bayesian authorship
attribution (sklearn ClassifierMixin, plugs into Pipeline/cross_validate) and a PyMC
hierarchical group-comparison model. CLI: `tamga embed`, `tamga bayesian`. Phases 1-3 remain
installable without either extra.

Phase 5 (viz + reports + wizard shell), Phase 6 (docs + PyPI) remain.

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
