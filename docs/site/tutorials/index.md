# Tutorials

Two runnable tutorials ship with tamga.

## [Federalist Papers](federalist.md)

Reproduce the classical Mosteller & Wallace (1964) authorship attribution on the 85
Federalist Papers, projecting the disputed papers (Nos. 49–58, 62, 63) onto the
Hamilton / Madison feature space.

Illustrates: corpus ingestion, MFW feature extraction, Burrows Delta training,
cross-validation, PCA visualisation, Ward consensus dendrogram, Zeta contrast tables.

## [PAN-CLEF verification](pan-clef.md)

An end-to-end forensic-verification pipeline on a PAN-CLEF-style setup: pair a
questioned document with a candidate's known samples plus an impostor pool, score with
General Impostors, calibrate via Platt scaling, and report the full PAN metric suite —
AUC, c@1, F0.5u, Brier, ECE, C_llr — alongside a forensic HTML report with LR framing
and chain-of-custody.

Illustrates: the full `tamga.forensic` workflow end-to-end.
