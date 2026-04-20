# Öznitelikler

Her öznitelik çıkarıcı, yöntemlerin tükettiği ortak sayısal zarf olan bir `FeatureMatrix` döndürür.

## FeatureMatrix

```python
@dataclass
class FeatureMatrix:
    X: np.ndarray            # (n_docs, n_features)
    document_ids: list[str]
    feature_names: list[str]
    feature_type: str
    extractor_config: dict[str, Any]
    provenance_hash: str
```

Temel özellikler:

- `fm.n_features`, `n_docs` için `len(fm)`
- `fm.as_dataframe()` — `document_ids` ile indekslenmiş pandas `DataFrame`
- `fm.concat(other)` — aynı satır kimliklerine sahip iki matrisi sütun bazında birleştirir

## Mevcut çıkarıcılar

`tamga`'dan içe aktarın:

| Çıkarıcı | Girdi | Çıktı |
|---|---|---|
| `MFWExtractor(n=..., scale=..., lowercase=...)` | Corpus | en sık n sözcüğün göreli frekansları (z-puanlı, L1, L2 veya ham) |
| `CharNgramExtractor(n=..., include_boundaries=...)` | Corpus | karakter n-gramı sayımları (sklearn CountVectorizer'a devredilir) |
| `WordNgramExtractor(n=..., lowercase=...)` | Corpus | sözcük n-gramı sayımları |
| `PosNgramExtractor(n=..., coarse=...)` | Corpus | spaCy sözcük türü (POS) n-gramları |
| `DependencyBigramExtractor()` | Corpus | (head_lemma, dep, child_lemma) üçlüleri |
| `FunctionWordExtractor(wordlist=...)` | Corpus | paketlenmiş İngilizce işlev sözcüğü frekansları |
| `PunctuationExtractor()` | Corpus | ASCII noktalama frekansları |
| `ReadabilityExtractor()` | Corpus | altı okunabilirlik indeksi (Flesch, FK-grade, Gunning Fog, Coleman-Liau, ARI, SMOG) |
| `SentenceLengthExtractor()` | Corpus | cümle başına belirteç sayısının ortalama, standart sapma ve çarpıklığı |
| `LexicalDiversityExtractor()` | Corpus | TTR, MATTR, MTLD, HD-D, Yule's K/I, Herdan's C, Simpson's D |
| `SentenceEmbeddingExtractor(model=...)` | Corpus | sentence-transformers havuzlanmış gömmesi (ek: `tamga[embeddings]`) |
| `ContextualEmbeddingExtractor(model=..., pooling=...)` | Corpus | HF dönüştürücü gizli durum vektörleri (ek: `tamga[embeddings]`) |

## Öznitelikleri birleştirme

Çok öznitelikli matris oluşturmanın iki yolu:

### Python

```python
from tamga import MFWExtractor, PunctuationExtractor

mfw = MFWExtractor(n=200, scale="zscore").fit_transform(corpus)
punct = PunctuationExtractor().fit_transform(corpus)
combined = mfw.concat(punct)  # (n_docs, n_mfw + n_punct)
```

### study.yaml

```yaml
features:
  - id: mfw
    type: mfw
    n: 200
    scale: zscore
  - id: punct
    type: punctuation
```

Yöntemler öznitelik kimliklerine başvurabilir; çalıştırıcı her matrisi bir kez oluşturur ve yeniden kullanır.

## Adli dilbilim öznitelik çıkarıcıları

Konu-değişmez iki çıkarıcı `tamga.forensic` altında yer alır:

- `CategorizedCharNgramExtractor(n=..., categories=...)` — Sapkota ve ark. 2015'e göre her n-gram örneğini sınıflandırır (prefix / suffix / whole_word / mid_word / multi_word / punct / space). `categories=("prefix", "suffix", "punct")` parametresi, konular arası en iyi genellemeyi sağlayan ek odaklı öznitelik kümesini üretir.
- `distort_corpus(corpus, mode="dv_ma"|"dv_sa")` — Stamatatos 2013 içerik maskeleme. Yeni bir Corpus döndürür; mevcut herhangi bir çıkarıcıya beslenebilir.

Bkz. [Konu-değişmez öznitelikler](../forensic/topic-invariance.md).

## Ölçekleme

Çoğu çıkarıcı `scale ∈ {"none", "zscore", "l1", "l2"}` parametresini kabul eder:

- `none` — ham sayımlar. Bayesian Wallace–Mosteller için kullanın.
- `l1` — göreli frekanslar (satır toplamı 1'e eşit). Zeta-benzeri karşıtlık yöntemleri için kullanın.
- `l2` — birim normlu satırlar. Kosinüs tabanlı uzaklıklar için kullanın.
- `zscore` — eğitim ortalamaları / standart sapmalarına göre sütun bazında z-puanı (Stylo kuralı). **Burrows Delta için zorunludur.**

Z-puanı ortalaması / standart sapması `fit` aşamasında öğrenilir ve `transform` sırasında uygulanır; dolayısıyla görülmemiş belgeler üzerindeki puanlar eğitim dağılımını kullanır.

## Sonraki adım

- [Methods](methods.md) — FeatureMatrix'i alıp Result üretmek.
