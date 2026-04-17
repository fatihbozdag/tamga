# tamga

**Next-generation computational stylometry — a Python replacement for R's Stylo.**

`tamga` ("mark, brand, clan-sign" — from Old Turkic) is a Python package and interactive CLI for
authorship attribution, author-group style comparison, and Digital Humanities stylometric
analysis. It reimplements the analytical breadth of R's `Stylo` and adds modern NLP and ML on top.

> Named after the **tamga**, the Turkic clan-mark by which individual and familial identity was
> recognised at a glance — the material-culture counterpart to a stylistic fingerprint.

## Status

**Phase 5 — Viz, reports, runner, shell.** Ships publication-grade matplotlib renderers
(dendrogram, scatter, distance heatmap, confusion matrix, feature importance, Zeta preference
plot) with 300-DPI defaults + colorblind palette; HTML/Markdown reports rendered from saved
Result directories via Jinja2; `tamga run study.yaml` orchestrator that executes a full
declarative study (features + methods) end-to-end; Rich-based `tamga shell` wizard.
CLI: `tamga run`, `report`, `plot`, `shell`. Phase 4 extras: `tamga[embeddings]` (sentence +
contextual-BERT embeddings), `tamga[bayesian]` (Wallace-Mosteller + PyMC hierarchical model).

Phase 6 (MkDocs site + Federalist/EFCAMDAT tutorials + PyPI publish) remains.

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
