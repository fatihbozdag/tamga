# Konudan bağımsız öznitelikler

Konu geçişleri, klasik stilometrinin gerçek adli veriler üzerindeki en yaygın başarısızlık nedenidir. Bir şüphelinin tehdit mektubu ile kişisel e-postası tipik olarak farklı konulardadır; ancak muhtemelen aynı yazara aittir. Bu durumda filtrelenmemiş karakter n-gram ve sözcük n-gram öznitelikleri konu tespitine dönüşür.

tamga iki tamamlayıcı araç sunar.

## Sapkota karakter n-gram kategorileri

`CategorizedCharNgramExtractor`, her karakter n-gram **oluşumunu** (yalnızca dizgiyi değil) kaynak metindeki konumuna göre sınıflandırır. Öznitelik sütunları `<ngram>|<category>` biçiminde adlandırılır; bu sayede `the|whole_word` ve `the|prefix` ayrı kanallar olur — açık ve denetlenebilir.

Yedi kategori:

| Kategori | Açıklama |
|---|---|
| `prefix` | sözcük başı + karakter içi (ör., "there" içindeki "the") |
| `suffix` | karakter içi + sözcük sonu (ör., "running" içindeki "ing") |
| `whole_word` | tam olarak bir sözcük, her iki uçta sınır |
| `mid_word` | tek bir sözcüğün tamamen içinde |
| `multi_word` | iki sözcük arasındaki boşluğu kapsıyor |
| `punct` | herhangi bir noktalama karakteri içeriyor |
| `space` | boşluk içeriyor ancak multi_word için yeterli değil |

Sapkota et al. (2015), yalnızca **ek (prefix + suffix) + punct** seçiminin konu geçişli atıflamayı önemli ölçüde iyileştirdiğini göstermiştir — adli dilbilim varsayılanı budur.

```python
from tamga.forensic import CategorizedCharNgramExtractor

extractor = CategorizedCharNgramExtractor(
    n=3,
    categories=("prefix", "suffix", "punct"),  # konudan bağımsız alt küme
    scale="zscore",
    lowercase=True,
)
fm = extractor.fit_transform(corpus)
```

## Stamatatos bozunumu

`distort_corpus`, **içeriği** maskelerken **stili** korumak için belgeler üzerinde ön işlem uygular: işlev sözcükleri (function words), noktalama, rakamlar ve boşluk karakterleri aynen bırakılır; içerik sözcüklerinin karakterleri değiştirilir.

### İki mod

**DV-MA** (*Bozunum Görünümü — Çoklu Yıldız*): her içerik-sözcük karakteri → `*`. Uzunluk korunur — morfolojik alışkanlıklar (tipik sözcük uzunlukları) görünür kalır.

**DV-SA** (*Bozunum Görünümü — Tek Yıldız*): her içerik sözcüğü → tek `*`. Agresif; yalnızca işlev-sözcük ve noktalama deseni hayatta kalır.

```python
from tamga.forensic import distort_corpus
from tamga import MFWExtractor

distorted = distort_corpus(corpus, mode="dv_ma")

# Aşağı akış çıkarıcıları bozulmuş metni görür — konu sinyali maskelenir.
fm = MFWExtractor(n=200, scale="zscore").fit_transform(distorted)
```

### Kısaltmalar

`_TOKEN_RE` ve yerleşik işlev-sözcük listesi, yaygın İngilizce kısaltmaları (`don't`, `it's`, `we'll`, `they've`, …) olduğu gibi korur. `o'clock` ve kesme işareti içeren diğer içerik sözcükleri, parçalara ayrılmak yerine tek bir bitişik dize olarak maskelenir (ör., `*******`).

### Özel işlev-sözcük listesi

```python
distorted = distort_corpus(
    corpus,
    mode="dv_ma",
    function_words={"the", "a", "of", "to", "and"},   # minimal durdurma listesi
)
```

Her sözcüğü içerik sözcüğü olarak değerlendirmek için `frozenset()` geçirin (DV-MA tümüyle `*` içeren bir metin üretir).

## İkisini birleştirme

Sapkota kategorileri + Stamatatos bozunumu temiz biçimde bir araya gelir:

```python
distorted = distort_corpus(corpus, mode="dv_ma")
extractor = CategorizedCharNgramExtractor(
    n=3, categories=("prefix", "suffix", "punct"), lowercase=True
)
fm = extractor.fit_transform(distorted)
```

Bu, **çift** konudan bağımsız bir öznitelik kümesi üretir — içerik maskelenmiş metinden çıkarılan ek ve noktalama n-gramları — ve çapraz tür PAN görevlerinde filtrelenmemiş karakter n-gramlarını düzenli olarak geride bırakır.

## Referans

::: tamga.forensic.char_ngrams.CategorizedCharNgramExtractor
    options:
      show_root_full_path: false

::: tamga.forensic.char_ngrams.classify_ngram

::: tamga.forensic.distortion.distort_corpus

::: tamga.forensic.distortion.distort_text
