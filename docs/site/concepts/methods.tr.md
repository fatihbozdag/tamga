# Yöntemler

Yöntemler, bir `FeatureMatrix`'i `Result`'a dönüştürür. Her yöntem, anlamlı olduğu yerde `sklearn` uyumludur (`fit`, `predict`, `fit_transform`).

## Yazar tespiti — Delta varyantları

Tüm Delta varyantları `_DeltaBase`'i paylaşır, z-puanlı öznitelikler üzerinde çalışır ve en yakın-merkez sınıflandırıcı üretir.

| Sınıf | Uzaklık | Referans |
|---|---|---|
| `BurrowsDelta` | ortalama mutlak L1 | Burrows 2002 |
| `ArgamonLinearDelta` | L2 (Öklid) | Argamon 2008 |
| `QuadraticDelta` | karesel L2 | — |
| `CosineDelta` | 1 − kosinüs benzerliği | Smith & Aldridge 2011 |
| `EderDelta` / `EderSimpleDelta` | ağırlıklı Delta varyantları | Eder 2015 |

```python
from tamga import MFWExtractor, BurrowsDelta
fm = MFWExtractor(n=200, scale="zscore", lowercase=True).fit_transform(corpus)
y = np.array(corpus.metadata_column("author"))
clf = BurrowsDelta().fit(fm, y)
predictions = clf.predict(fm)       # en yakın-merkez etiketleri
probs = clf.predict_proba(fm)       # negatif uzaklıklar üzerinde softmax
```

## Karşıtlık — Zeta

Craig'in Zeta'sı (`ZetaClassic`) ve Eder'in yumuşatılmış varyantı (`ZetaEder`), bir yazar grubunun diğerine kıyasla en çok tercih ettiği sözcük dağarcığını çıkarır.

```python
from tamga.methods.zeta import ZetaClassic
result = ZetaClassic(group_by="author", top_k=50, group_a="Hamilton", group_b="Madison").fit_transform(corpus)
df_a, df_b = result.tables   # oranlarla birlikte en yüksek k A-tercihli / B-tercihli sözcükler
```

## Boyut indirgeme

`PCAReducer`, `MDSReducer`, `TSNEReducer`, `UMAPReducer` — hepsi bir `FeatureMatrix` kabul eder ve 2 boyutlu / n boyutlu koordinatların yanı sıra (PCA için) açıklanan varyans oranlarını içeren bir `Result` döndürür.

## Kümeleme

`HierarchicalCluster(linkage=...)` (Ward, average, complete, single), `KMeansCluster`, `HDBSCANCluster`. Küme etiketleri + (hiyerarşik için) dendrogramlara yönelik bağlantı matrisi içeren bir `Result` üretir.

## Konsensüs ağaçları

`BootstrapConsensus(mfw_bands=[100, 200, 300], replicates=20)` — MFW bantları arasında Eder 2017 bootstrap'i; klade destek değerleriyle birlikte Newick formatında konsensüs ağacı üretir.

## Sınıflandırma + çapraz doğrulama

Herhangi bir sklearn sınıflandırıcı (Lojistik Regresyon, lineer / RBF SVM, Random Forest, HistGBM) `build_classifier(name)` ile, ayrıca üç stilometri odaklı çapraz doğrulama stratejisiyle `cross_validate_tamga(fm, y, cv_kind=...)`:

- `stratified` — StratifiedKFold; `seed` karıştırmayı denetler
- `loao` — Leave-One-Author-Out (yazar grup olarak LeaveOneGroupOut)
- `leave_one_text_out` — LeaveOneOut

## Bayesian

- `BayesianAuthorshipAttributor` — Wallace–Mosteller Dirichlet-düzleştirilmiş log-oran sınıflandırıcısı. Ham sayımları kullanır (`MFWExtractor(scale="none")`).
- `HierarchicalGroupComparison` — yazar başına bir grup hiperparametreden türeyen değişken-kesişim PyMC modeli. İki yazar popülasyonunun stilistik bir öznitelik açısından sistematik biçimde farklılaşıp farklılaşmadığını sınamak için kullanışlıdır. `tamga[bayesian]` gerektirir.

## Adli dilbilim yöntemleri

`tamga.forensic` altında:

| Yöntem | Görev | Referans |
|---|---|---|
| `GeneralImpostors` | tek sınıflı yazar doğrulama | Koppel & Winter 2014 |
| `Unmasking` | uzun metin doğrulama (doğruluk-bozunma eğrisi) | Koppel & Schler 2004 |
| `CalibratedScorer` | herhangi bir puanlayıcının Platt / izotonik kalibrasyonu | Platt 1999; Niculescu-Mizil & Caruana 2005 |

Bkz. [Adli dilbilim araç takımı](../forensic/index.md).

## Sonraki adım

- [Sonuçlar ve köken bilgisi](results.md) — her yöntemin döndürdüğü değer.
