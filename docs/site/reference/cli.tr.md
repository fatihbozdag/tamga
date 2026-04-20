# CLI başvurusu

Her tamga CLI komutu. `tamga` giriş noktası aracılığıyla `tamga` olarak kurulur.

## Proje iskeleti

### `tamga init <name>`

Yeni bir proje dizini oluşturur.

```bash
tamga init my-study
```

Oluşturulanlar:

```
my-study/
├── corpus/             # .txt dosyalarını buraya bırakın
│   └── metadata.tsv    # dosya başına bir satır
├── study.yaml          # bildirimsel çalışma yapılandırması
└── README.md           # kısa bir yönlendirme
```

## İçe aktarma

### `tamga ingest <path>`

İsteğe bağlı üst veriyle bir derlem dizinini ayrıştırır.

```bash
tamga ingest corpus/ --metadata corpus/metadata.tsv [--strict|--no-strict]
```

- `--strict` (varsayılan) — herhangi bir belge üst veri satırından yoksunsa hata verir
- `--no-strict` — kısmi kapsamı kabul eder

Çıktı, sonraki komutlar için bir spaCy DocBin olarak önbelleğe alınır.

### `tamga info`

İçe aktarılmış bir derlemi özetler: belge sayısı, üst veri alanları ve değer dağılımları,
toplam simge sayısı.

## Öznitelikler

### `tamga features <path>`

Bir öznitelik matrisi oluşturur ve özeti yazdırır.

```bash
tamga features corpus/ --metadata corpus/metadata.tsv --type mfw --n 500
```

Türler: `mfw`, `word_ngram`, `char_ngram`, `function_word`, `punctuation`,
`lexical_diversity`, `readability`.

## Yöntemler

Tüm yöntem komutları `--metadata`, `--group-by <field>`, `--seed <int>` seçeneklerini kabul eder.

| Komut | İşlev |
|---|---|
| `tamga delta <path> --method {burrows,argamon,eder,cosine,quadratic}` | Delta'yı uygular, yazar başına tahminleri yazdırır |
| `tamga zeta <path> --group-a X --group-b Y` | İki yazar grubu arasında Craig's Zeta karşılaştırması yapar |
| `tamga reduce <path> --method {pca,mds,tsne,umap} --n-components 2` | Boyut indirgeme → parquet |
| `tamga cluster <path> --method {hierarchical,kmeans,hdbscan} --n-clusters N --seed S` | k-means için `--seed` ile kümeleme |
| `tamga consensus <path>` | MFW bantları üzerinde önyükleme fikir birliği ağacı |
| `tamga classify <path> --estimator {logreg,svm_linear,svm_rbf,rf,hgbm} --cv-kind {stratified,loao,leave_one_text_out}` | sklearn sınıflandırıcısı + stilometri uyumlu çapraz doğrulama |
| `tamga embed <path>` | Cümle veya bağlamsal gömme (ek: `tamga[embeddings]`) |
| `tamga bayesian <path>` | Wallace–Mosteller yazar tespiti + hiyerarşik grup karşılaştırması (ek: `tamga[bayesian]`) |

## Düzenleme

### `tamga run <study.yaml>`

Bildirimsel bir çalışmayı uçtan uca yürütür.

```bash
tamga run study.yaml --name demo [--output-dir results/]
```

Her yöntemin `Result` nesnesini kendi alt dizinine ve bir `resolved_config.json` dosyasına yazar.

### `tamga report <run-dir>`

Bir çalıştırma dizininden Jinja2 HTML veya Markdown raporu oluşturur.

```bash
tamga report results/demo --output results/demo/report.html [--format html|md]
```

### `tamga plot <run-dir>`

Kaydedilmiş Result nesnelerinden yönteme özgü şekiller (PCA dağılım grafiği, Ward dendrogramı, Zeta tercih grafiği, …) oluşturur.

### `tamga shell`

Bir çalışma kurulumunda size eşlik eden etkileşimli Rich tabanlı sihirbaz.

## Önbellek

### `tamga cache <cmd>`

`tamga ingest` tarafından üretilen spaCy DocBin önbelleğini yönetir:

- `tamga cache info` — özetler
- `tamga cache clear` — kaldırır

## Yardım alma

Her komut `--help` seçeneğini destekler:

```bash
tamga --help
tamga run --help
```
