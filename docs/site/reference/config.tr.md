# study.yaml şeması

`bitig run` tarafından tüketilen bildirimsel çalışma yapılandırması. Minimal bir örnek:

```yaml
name: my-study
seed: 42

corpus:
  path: corpus
  metadata: corpus/metadata.tsv

features:
  - id: mfw200
    type: mfw
    n: 200
    scale: zscore
    lowercase: true

methods:
  - id: burrows
    kind: delta
    method: burrows
    features: mfw200
    group_by: author
```

Yapılandırma biçimi, `extra="forbid"` olan bir Pydantic modeli ile doğrulanır — `corpus`,
`preprocess`, `viz`, `report`, `cache` ve `output` üzerindeki bilinmeyen üst düzey anahtarlar
reddedilir. (Bir **öznitelik** veya **yöntem** girişindeki bilinmeyen anahtarlar farklıdır:
reddedilmek yerine o girişin `params` alanında toplanır — bkz. [öznitelikler](#features) /
[yöntemler](#methods).)

## Üst düzey anahtarlar

| Anahtar | Tür | Zorunlu | Açıklama |
|---|---|---|---|
| `name` | str | hayır | Çalışma adı; raporlarda gösterilir (varsayılan: `"unnamed-study"`) |
| `seed` | int | hayır | Varsayılan seed değeri (42). Her stokastik yönteme iletilir. |
| `corpus` | object | **evet** | Derlem yapılandırması (aşağıya bakınız) |
| `features` | list | hayır | Öznitelik çıkarıcılar (varsayılan: boş — ancak bir yöntem çalıştırmak için en az birine ihtiyacınız var) |
| `methods` | list | hayır | Çalıştırılacak yöntemler (varsayılan: boş) |
| `preprocess` | object | hayır | Dil + spaCy + normalleştirme ayarları |
| `viz` | object | hayır | Şekil formatı / DPI / stil |
| `report` | object | hayır | Rapor formatı ve içeriği |
| `output` | object | hayır | Çıktı dizini / zaman damgalama |
| `cache` | object | hayır | DocBin önbellek dizini + yeniden kullanım anahtarı |

## corpus

```yaml
corpus:
  path: corpus                    # .txt dosyalarının bulunduğu dizin (zorunlu)
  metadata: corpus/metadata.tsv   # isteğe bağlı: dosya adı + rastgele alanları içeren TSV
  filter:                         # isteğe bağlı: çalıştırmadan önce derlemi filtrele
    role: [train]
```

!!! note
    `corpus.strict` diye bir alan yoktur. Katı/esnek üst veri kapsamı, bir `study.yaml`
    ayarı değil, `bitig ingest` komutu üzerindeki bir bayraktır (`--strict` / `--no-strict`).

## features

Her öznitelik çıkarıcı, bir `id` (yöntemler tarafından başvurulan), bir `type` ve
türe özgü parametreler içeren bir sözlüktür. `id`/`type` dışındaki herhangi bir anahtar,
çıkarıcının `params` alanına katlanır ve yürütme sırasında çıkarıcı imzasına karşı
doğrulanır.

### Desteklenen türler

| type | parametreler |
|---|---|
| `mfw` | `n`, `min_df`, `max_df`, `scale` ({none, zscore, l1, l2}), `lowercase` |
| `word_ngram` | `n` (int veya [min, max]), `lowercase`, `scale` |
| `char_ngram` | `n` (int veya [min, max]), `include_boundaries`, `scale` |
| `pos_ngram` | `n`, `tagset` ({coarse, fine}), `scale` |
| `dependency_bigram` | `scale` |
| `function_word` | `wordlist` (isteğe bağlı liste veya yol), `language`, `scale` |
| `punctuation` | (yok) |
| `lexical_diversity` | `indices` (sekiz indeksin bir alt kümesi) |
| `readability` | `indices` (dile özgü indekslerin bir alt kümesi) |
| `sentence_length` | (yok) |
| `sentence_embedding` | `model`, `language`, `device` (ek: `bitig[embeddings]`) |
| `contextual_embedding` | `model`, `language`, `layer`, `pool`, `device` (ek: `bitig[embeddings]`) |

## methods

Her yöntem, bir `id`, bir `kind`, isteğe bağlı bir `features` (öznitelik id'si), isteğe
bağlı bir `group_by`/`cv` ve `params` içeren bir sözlüktür (başka herhangi bir anahtar
`params` alanına katlanır).

### Desteklenen türler

| kind | Açıklama |
|---|---|
| `delta` | En yakın-centroid yazar tespiti (varsayılan olarak `method: burrows`; `params` içine düşer) |
| `rolling_delta` | İşbirlikli / karma yazarlı metinler için kayan-pencere Delta |
| `verify` | Tek sınıflı yazar doğrulama (ör. General Impostors) |
| `zeta` | Craig's Zeta; `group_by` ve çıkarılan ya da belirtilen `params.group_a` / `group_b` gerektirir |
| `reduce` | Boyut indirgeme (varsayılan PCA); `params.n_components` |
| `cluster` | Hiyerarşik kümeleme (varsayılan Ward); `params.n_clusters`, `params.linkage` |
| `consensus` | Önyükleme fikir birliği ağacı; `params.mfw_bands`, `params.replicates` |
| `classify` | sklearn sınıflandırıcısı; `params.estimator`, `cv.kind`, `cv.folds`, `cv.groups_from` |
| `bayesian` | Wallace–Mosteller yazar tespiti + hiyerarşik grup karşılaştırması |

### cv (çapraz doğrulama, `classify` için)

```yaml
cv:
  kind: stratified        # stratified | loao | group_kfold | leave_one_text_out
  folds: 5                # stratified / group_kfold için
  groups_from: author     # loao / group_kfold için zorunlu
```

## preprocess

```yaml
preprocess:
  language: en              # varsayılan; kayıtlı dillerden biri
  spacy:
    model: null            # varsayılan null → dile göre çözümlenir (ör. en_core_web_trf)
    backend: null          # null → dilden çözümlenir (spacy | spacy_stanza)
    device: auto           # auto | cpu | mps | cuda
    exclude: []            # devre dışı bırakılacak spaCy ardışık düzen bileşenleri
  normalize:
    lowercase: false
    strip_punct: false
    collapse_numerals: false
    expand_contractions: false
```

## viz

```yaml
viz:
  format: [pdf, png]       # pdf, png, svg, eps, tiff seçeneklerinden herhangi biri
  dpi: 300
  style: default
  palette: colorblind
```

## report

```yaml
report:
  format: none             # none | html | md
  offline: false           # kendi kendine yeten bir HTML dosyası için satır içi varlıklar
  include: [corpus, config, provenance, results]
  title: null
```

## output

```yaml
output:
  dir: results/            # varsayılan
  timestamp: true          # çalıştırmaları zaman damgalı alt dizinlere sarar
```

## cache

```yaml
cache:
  dir: .bitig/cache        # spaCy DocBin önbellek konumu (varsayılan)
  reuse: true              # girdiler değişmediğinde önbelleğe alınmış ayrıştırmaları yeniden kullan
```

## Gerçekçi çok yöntemli bir örnek

```yaml
name: federalist
seed: 42
output: { dir: results, timestamp: false }

corpus:
  path: corpus
  metadata: corpus/metadata.tsv
  filter:
    role: [train]

features:
  - id: mfw200
    type: mfw
    n: 200
    scale: zscore
    lowercase: true

methods:
  - id: burrows
    kind: delta
    method: burrows
    features: mfw200
    group_by: author

  - id: pca
    kind: reduce
    features: mfw200
    params: { n_components: 2 }

  - id: ward
    kind: cluster
    features: mfw200
    params: { n_clusters: 3, linkage: ward }

  - id: zeta_h_m
    kind: zeta
    group_by: author
    params:
      top_k: 50
      group_a: Hamilton
      group_b: Madison
```
