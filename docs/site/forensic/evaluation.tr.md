# Değerlendirme (PAN paketi)

Adli yayınlar ve mahkemeler ham doğruluktan fazlasını bekler. tamga, standart PAN doğrulama görevi metrik menüsünü tek bir çağrının arkasında sunar.

## Tek çağrı değerlendirme

```python
from tamga.forensic import compute_pan_report

report = compute_pan_report(
    probs=calibrated_probs,     # CalibratedScorer'dan
    y=ground_truth_labels,
    log_lrs=log_lr_values,      # isteğe bağlı; cllr_bits'i etkinleştirir
)
report.to_dict()
# {
#   "auc": 0.94, "c_at_1": 0.88, "f05u": 0.87,
#   "brier": 0.11, "ece": 0.042, "cllr_bits": 0.31,
#   "n_target": 80, "n_nontarget": 120,
# }
```

## Metrikler

| Metrik | Ölçtüğü | Referans |
|---|---|---|
| `auc` | Sıralama yeteneği. 1.0 mükemmel, 0.5 rastgele. | — |
| `c_at_1` | İlkeli çekinme kredisiyle doğruluk | Peñas & Rodrigo 2011 |
| `f05u` | Yanıtsızlık cezalı hassasiyet ağırlıklı F-ölçüsü | Bevendorff et al. PAN 2022 |
| `brier` | Ortalama kare posterior hatası. 0 mükemmel. | Brier 1950 |
| `ece` | Beklenen Kalibrasyon Hatası (eşit genişlikli bölmeler) | — |
| `cllr` | Log-olabilirlik-oranı maliyeti — adli uygun puanlama kuralı | Brümmer & du Preez 2006 |
| `tippett` | Sınıf başına kümülatif hedef / hedef olmayan LR dağılımları | — |

### c@1

$$
\text{c@1} = \frac{1}{n}\!\left( n_\text{correct} + n_\text{unanswered} \cdot \frac{n_\text{correct}}{n} \right)
$$

*yanıtsız* denemeler, kalibre edilmiş olasılığı `[0.5 − margin, 0.5 + margin]` içinde olan denemelerdir. Margin = 0 (varsayılan) c@1'i ham doğruluğa indirger.

PAN doğrulama paylaşımlı görevi, çekinmesini bilen sistemleri ödüllendirdiği için — doğrudan "yetersiz kanıt" adli kavramıyla örtüşür — 2013'ten bu yana c@1'i birincil metrik olarak kullanmaktadır.

### C_llr

$$
C_\text{llr} = \frac{1}{2}\!\left[
  \frac{1}{|T|}\!\sum_{i \in T} \log_2\!\left(1 + \tfrac{1}{\text{LR}_i}\right)
  +
  \frac{1}{|N|}\!\sum_{i \in N} \log_2\!\left(1 + \text{LR}_i\right)
\right]
$$

$T$ = hedef denemeler (target trials), $N$ = hedef olmayan denemeler (non-target trials). En iyi kalibre edilmiş referans sisteme göre deneme başına ortalama bilgi kaybı (bit cinsinden) olarak yorumlanır.

- **Yalnızca önsel olasılık sistemi** (her log-LR = 0) → C_llr = **1.0** tam.
- **Mükemmel, güvenilir sistem** → C_llr ≈ 0.
- **Yanıltıcı sistem** (yanlış işaret) → C_llr > 1.

C_llr < 1 olan bir sistem yalnızca önsel olasılık sistemini geride bırakır. Adli yayınlar, C_llr'nin ayrım *ve* kalibrasyonu tek bir skalerde yakalaması nedeniyle AUC'nin yanında C_llr'yi düzenli olarak raporlar.

## Tippett grafikleri

`tippett(log_lrs, y)`, doğrudan çizebileceğiniz sınıf başına kümülatif dağılımları döndürür:

```python
import matplotlib.pyplot as plt
from tamga.forensic import tippett

data = tippett(log_lrs, y)
plt.step(data["thresholds"], data["target_cdf"], label="aynı-yazar")
plt.step(data["thresholds"], data["nontarget_cdf"], label="farklı-yazar")
plt.xlabel("log₁₀(LR) eşiği")
plt.ylabel("P(log-LR ≥ eşik | sınıf)")
plt.legend()
```

İyi ayrım yapan bir sistemde hedef CDF sağda (yüksek log-LR'ler ağırlıklı olarak hedef), hedef olmayan CDF solda birikirler.

## Referans

::: tamga.forensic.metrics.compute_pan_report

::: tamga.forensic.metrics.PANReport

::: tamga.forensic.metrics.auc

::: tamga.forensic.metrics.c_at_1

::: tamga.forensic.metrics.f05u

::: tamga.forensic.metrics.cllr

::: tamga.forensic.metrics.ece

::: tamga.forensic.metrics.brier

::: tamga.forensic.metrics.tippett
