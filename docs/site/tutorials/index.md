# Tutorials

Three runnable tutorials ship with bitig.

## [Federalist Papers](federalist.md)

Reproduce the classical Mosteller & Wallace (1964) authorship attribution on the 85
Federalist Papers, projecting the disputed papers (Nos. 49–57, 62, 63) onto the
Hamilton / Madison feature space.

Illustrates: corpus ingestion, MFW feature extraction, Burrows Delta training,
cross-validation, PCA visualisation, Ward clustering, bootstrap consensus dendrograms,
Zeta contrast tables.

## [PAN-CLEF verification](pan-clef.md)

An end-to-end forensic-verification pipeline on a PAN-CLEF-style setup: pair a
questioned document with a candidate's known samples plus an impostor pool, score with
General Impostors, calibrate via Platt scaling, and report the full PAN metric suite —
AUC, c@1, F0.5u, Brier, ECE, C_llr — alongside a forensic HTML report with LR framing
and chain-of-custody.

Illustrates: the full `bitig.forensic` workflow end-to-end.

## [Turkish stylometry](turkish.md)

An end-to-end Turkish authorship walkthrough — ingest a Turkish corpus through the
Stanford Stanza (BOUN treebank) backend via `spacy-stanza`, then run the standard bitig
feature extractors and methods unchanged.

Illustrates: the multi-language registry, the `bitig[turkish]` extra, native Turkish
readability formulas, and Turkish contextual embeddings.
