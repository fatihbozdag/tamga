---
hide:
  - navigation
  - toc
---

# Turkish terminology glossary

This file pins the Turkish rendering of recurring technical terms used across the tamga
documentation. It is not rendered in the public navigation. Translators MUST consult this
table before drafting or updating a Turkish page; consistency is enforced by grep before
each commit.

| English | Turkish | Notes |
|---|---|---|
| stylometry | *stilometri* | Established loan; matches how computational-linguistics literature in Turkish refers to the field. |
| corpus | *derlem* | Standard TDK term. |
| authorship attribution | *yazar tespiti* | Preferred over "yazar atfı" for naturalness. |
| authorship verification | *yazar doğrulama* | Forensic sub-task. |
| forensic linguistics | *adli dilbilim* | Standard. |
| feature (stylometric) | *öznitelik* | Standard ML term. |
| function word | *işlev sözcüğü* | Corpus linguistics standard. |
| readability | *okunabilirlik* | Standard. |
| classifier | *sınıflandırıcı* | Standard ML. |
| classification | *sınıflandırma* | Standard ML. |
| clustering | *kümeleme* | Standard ML. |
| embedding (vector) | *gömme* | Standard ML. |
| likelihood ratio | *olabilirlik oranı* | Stats standard. |
| calibration | *kalibrasyon* | Stats loan. |
| chain of custody | *delil zinciri* | Forensic/legal standard. |
| provenance | *köken bilgisi* | Keeps "provenance" feel; literal "soy" sounds odd for data. |

## Gloss pattern (for concepts / forensic pages)

| English | Turkish |
|---|---|
| *Use when:* | *Şu durumda kullanın:* |
| *Don't use when:* | *Şu durumda kullanmayın:* |
| *Expect:* | *Beklenen sonuç:* |

Apply these exact renderings in every gloss block across `concepts/` and `forensic/`
pages. Do not paraphrase the marker phrases.

Brand / proper nouns (never translate): **tamga**, **Burrows**, **Eder**, **Argamon**,
**Cosine**, **Quadratic Delta**, **Zeta**, **General Impostors**, **Unmasking**,
**Stamatatos**, **Sapkota**, **PANReport**, **Mosteller & Wallace**, **Federalist Papers**,
**PAN-CLEF**, **CalibratedScorer**, **study.yaml**.
