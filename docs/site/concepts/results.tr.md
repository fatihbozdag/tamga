# Sonuçlar ve köken bilgisi

Her yöntem, tamga genelinde ortak dönüş türü olan bir `Result` döndürür.

## Result

```python
@dataclass
class Result:
    method_name: str
    params: dict[str, Any]
    values: dict[str, Any]          # JSON-safe (ndarray {"__ndarray__": ...} olarak kodlanır)
    tables: list[pd.DataFrame]      # parquet olarak dışa aktarılır
    figures: list[Any]              # matplotlib figürleri veya ham baytlar
    provenance: Provenance | None
```

### Kalıcılık

```python
result.save("results/demo/pca")   # result.json + table_*.parquet yazar
```

`result.save(directory)` şunları yazar:

- `result.json` — `method_name`, `params`, `values` (numpy kodlanmış), `provenance`
- `table_0.parquet`, `table_1.parquet`, … — `tables` içindeki her DataFrame için birer dosya
- Figürler görselleştirme katmanına ertelenir (`render_figures.py` örnek başına)

`Result.from_json("results/demo/pca/result.json")` ile gidiş-dönüş sağlanır.

## Köken bilgisi

Her Result'ın `.provenance` alanı tam yeniden üretilebilirlik zarfını taşır:

```python
@dataclass
class Provenance:
    tamga_version: str
    python_version: str
    spacy_model: str
    spacy_version: str
    corpus_hash: str
    feature_hash: str | None
    seed: int
    timestamp: datetime
    resolved_config: dict[str, Any]
    # Adli dilbilim (tümü isteğe bağlı):
    questioned_description: str | None
    known_description: str | None
    hypothesis_pair: str | None
    acquisition_notes: str | None
    custody_notes: str | None
    source_hashes: dict[str, str]
```

`Provenance.current(...)` çalışma zamanı ve girdilerinizden bir kayıt oluşturur.
`Provenance.from_dict(...)` kaydedilmiş bir `result.json`'dan gidiş-dönüş sağlar.

### Yeniden üretilebilirlik sözleşmesi

Aynı seed değeriyle aynı corpus üzerinde aynı `study.yaml`'ın iki ayrı çalıştırılması **bayt düzeyinde özdeş** `result.json` üretir. Çalıştırıcı, `cfg.seed` değerini şu bileşenler boyunca iletir:

- herhangi bir örnekleme yönteminde numpy'ın varsayılan RNG'si
- her stokastik tahmincide scikit-learn'ün `random_state` parametresi (k-means, LogReg çapraz doğrulama, RandomForest, …)
- her Bayesian `pm.sample()` çağrısında PyMC'nin `random_seed` parametresi
- Stratified K-Fold karıştırması

Belirleyici olmayan bir durum hata sayılır — lütfen bildirin.

## Çok yöntemli çalıştırmaları yükleme

`tamga run study.yaml`, şu dizin yapısını üretir:

```
results/demo/
├── resolved_config.json
├── burrows/
│   └── result.json
├── pca/
│   ├── result.json
│   └── figure.png          # işlendiyse
└── zeta/
    ├── result.json
    ├── table_0.parquet
    └── table_1.parquet
```

`build_report(results/demo, output="report.html")`, dizin altındaki her `result.json`'ı yükler ve tek bir HTML raporu oluşturur.

## Sonraki adım

- [Adli dilbilim araç takımı](../forensic/index.md) — Provenance'ın adli dilbilim alanlarının devreye girdiği yer.
