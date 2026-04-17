# tamga

**Next-generation computational stylometry — a Python replacement for R's Stylo.**

`tamga` ("mark, brand, clan-sign" — from Old Turkic) is a Python package and interactive CLI for
authorship attribution, author-group style comparison, and Digital Humanities stylometric
analysis. It reimplements the analytical breadth of R's `Stylo` and adds modern NLP and ML on top.

> Named after the **tamga**, the Turkic clan-mark by which individual and familial identity was
> recognised at a glance — the material-culture counterpart to a stylistic fingerprint.

## Status

**Phase 3 — Analytical breadth.** Ships Craig's Zeta (classic + Eder variants),
dimensionality reducers (PCA/MDS/t-SNE/UMAP), clustering (hierarchical Ward/avg/complete/single,
k-means, HDBSCAN), bootstrap consensus trees (Newick output), and sklearn-wrapped classifiers
(logreg/SVM/RF/HGBM) with stylometry-aware CV (LOAO, leave-one-text-out, stratified).
CLI: `tamga zeta`, `reduce`, `cluster`, `consensus`, `classify` — all live. Phase 2 ships 10
feature extractors and the full Delta family (Burrows/Eder/Argamon/Cosine/Quadratic);
the Federalist Papers parity test attributes the disputed paper 49 to Madison at MFW=500.

Phases 4 (embeddings + Bayesian), 5 (viz + reports + wizard shell), 6 (docs + PyPI) remain.

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
