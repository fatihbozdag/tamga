# Federalist Papers — full stylometric analysis

A complete, reproducible demonstration of `bitig` against the classical
stylometry benchmark: the 85 Federalist Papers (1787–1788, public domain),
with attribution of the 11 historically disputed essays (49–57, 62, 63).

## Corpus

85 texts — Hamilton (51), Madison (15), Jay (5), joint Hamilton+Madison (3),
disputed (11) — copied verbatim from Project Gutenberg ebook 1404 and
labelled in `metadata.tsv` following the canonical consensus (Mosteller &
Wallace 1964; subsequent scholarship).

| Author                | Count |
|-----------------------|-------|
| Hamilton              | 51    |
| Madison               | 15    |
| Jay                   | 5     |
| Hamilton + Madison    | 3     |
| Disputed              | 11    |

## Reproduce

```bash
# From the repository root
uv pip install -e ".[dev,embeddings,bayesian]"
python -m spacy download en_core_web_sm

# Run the full study (Delta / Zeta / PCA / Ward / Consensus)
bitig run examples/federalist/study.yaml --name demo

# Render the publication-quality figures
python examples/federalist/render_figures.py

# Generate the HTML report (figures included)
bitig report examples/federalist/results/demo \
    --output examples/federalist/results/demo/report.html \
    --title "Federalist Papers — full analysis"

# Disputed-paper attribution (train on 71 undisputed single-author papers, test on 11)
bitig delta examples/federalist/corpus --method burrows --mfw 500 \
    --metadata examples/federalist/metadata.tsv --group-by author \
    --test-filter role=test

bitig bayesian examples/federalist/corpus --mfw 500 \
    --metadata examples/federalist/metadata.tsv --group-by author \
    --test-filter role=test
```

## Result

Both Burrows Delta and the Wallace–Mosteller-style Bayesian attribution
assign **every one of the 11 disputed papers to Madison**. The Bayesian
posterior probability is 1.000 across all disputed papers — the model is
effectively certain. This reproduces the classical Mosteller & Wallace
(1964) result.

## What the study.yaml does

`examples/federalist/study.yaml` runs five methods on the 71 single-author
papers (Joint and Disputed excluded via `role: [train]`):

1. **Burrows Delta** — nearest-author-centroid on 500 most-frequent words (z-scored).
2. **PCA** — 2-D projection of the same feature matrix. Hamilton / Madison /
   Jay form three clear clusters; see `results/demo/pca/pca.png`.
3. **Hierarchical clustering (Ward)** — 3 clusters. Ideally segregates
   Hamilton vs. Madison vs. Jay. Dendrogram in `results/demo/ward/ward.png`.
4. **Craig's Zeta** — contrastive vocabulary between Hamilton and Madison.
   Shows the "upon" vs. "whilst" etc. signature Mosteller & Wallace relied on.
   Preference plot in `results/demo/zeta_hamilton_madison/zeta.png`.
5. **Bootstrap consensus tree** — 5 MFW bands × 20 replicates = 100 Ward
   dendrograms, with majority-support clade extraction. Newick string in
   `results/demo/consensus/result.json`.

The disputed-paper attribution is a separate step (via `bitig delta` +
`bitig bayesian` with the `--test-filter role=test` flag) because it
requires a train/test split that the declarative `bitig run` workflow
does not yet expose.

## License

Federalist Papers texts are in the public domain. `metadata.tsv` and
`study.yaml` are BSD-3-Clause along with the rest of `bitig`.
