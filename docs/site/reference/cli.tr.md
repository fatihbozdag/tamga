# CLI başvurusu

Her bitig CLI komutu. `bitig` giriş noktası aracılığıyla `bitig` olarak kurulur.

Her komut `--help` seçeneğini destekler; aşağıdaki bayraklar en yaygın olanlardır, ancak
`bitig <cmd> --help` her zaman yetkili kaynaktır.

## Proje iskeleti

### `bitig init <name>`

Yeni bir proje dizini oluşturur.

```bash
bitig init my-study [--target DIR] [--language en] [--force]
```

- `--target, -t` — projenin oluşturulacağı üst dizin (varsayılan: geçerli dizin)
- `--language, -l` — oluşturulan `study.yaml` için derlem dili (varsayılan: `en`)
- `--force` — mevcut bir dizinin üzerine yazar

Oluşturulanlar:

```
my-study/
├── corpus/             # .txt dosyalarını buraya bırakın
│   └── metadata.tsv    # dosya başına bir satır
├── study.yaml          # bildirimsel çalışma yapılandırması
└── README.md           # kısa bir yönlendirme
```

Oluşturulan `study.yaml`, bir MFW öznitelik kümesi (ilk 1000 sözcük) ve bir Burrows Delta
yöntemi bildirir. Analizi genişletmek için başka öznitelik/yöntem blokları ekleyin
(bkz. [study.yaml şeması](config.md)).

## İçe aktarma

### `bitig ingest <path>`

İsteğe bağlı üst veriyle bir derlem dizinini ayrıştırır.

```bash
bitig ingest corpus/ --metadata corpus/metadata.tsv [--strict|--no-strict]
```

- `--metadata, -m` — dosya adı → yazar, grup, yıl, … eşlemesi yapan TSV
- `--strict` (varsayılan) — herhangi bir belge üst veri satırından yoksunsa hata verir
- `--no-strict` — kısmi kapsamı kabul eder
- `--language, -l` — derlem dili (spaCy arka ucunu belirler; varsayılan `en`)
- `--spacy-model` — spaCy ardışık düzen adını geçersiz kılar
- `--cache-dir` — önbelleğe alınmış ayrıştırmaların yazılacağı yer
- `--exclude` — devre dışı bırakılacak spaCy ardışık düzen bileşenleri

Çıktı, sonraki komutlar için bir spaCy DocBin olarak önbelleğe alınır.

### `bitig info`

Çalışma zamanı bilgisini yazdırır: bitig / Python / platform / spaCy sürümleri ve çalışma
dizininde bir `study.yaml` varsa yapılandırılmış dil. Argüman almaz. (İçe aktarılmış bir
derlemi **özetlemez** — yüklenen/ayrıştırılan belge sayılarını `bitig ingest` raporlar.)

## Öznitelikler

### `bitig features <path>`

Bir öznitelik matrisi oluşturur ve parquet'e kaydeder.

```bash
bitig features corpus/ --metadata corpus/metadata.tsv --type mfw --n 1000 --output features.parquet
```

- `--type` — `mfw` (varsayılan), `char_ngram`, `word_ngram`, `function_word`, `punctuation`
- `--n` — MFW/n-gram türleri için öznitelik sayısı (varsayılan: 1000)
- `--min-df`, `--scale`, `--lowercase` — MFW ayar düğmeleri
- `--metadata, -m`, `--output, -o`

!!! note
    `lexical_diversity`, `readability`, POS/bağımlılık n-gramları ve gömmeler Python API
    aracılığıyla kullanılabilir (bkz. [Öznitelikler](../concepts/features.md)) ancak
    `bitig features` CLI komutuna bağlanmamıştır.

## Yöntemler

`--metadata`, `delta`, `zeta`, `classify` ve `bayesian` için **zorunludur**; `reduce`,
`cluster`, `consensus` ve `embed` için isteğe bağlıdır. `--group-by`, `--seed` ve `--mfw`
bazı komutlar tarafından kabul edilir ancak hepsi tarafından değil — kesin küme için
`bitig <cmd> --help` çalıştırın.

| Komut | İşlev |
|---|---|
| `bitig delta <path> --method {burrows,eder,eder_simple,argamon,cosine,quadratic}` | Delta'yı uygular, yazar başına tahminleri yazdırır. Ayrıca: `--mfw`, `--mfw-min`, `--group-by`, `--test-filter` |
| `bitig zeta <path> [--group-a X --group-b Y]` | İki yazar grubu arasında Craig's Zeta karşılaştırması. Ayrıca: `--variant {classic,eder}`, `--top-k`, `--group-by`. Atlanırsa gruplar otomatik seçilir |
| `bitig reduce <path> --method {pca,mds,tsne,umap} --n-components 2` | Boyut indirgeme → parquet. Ayrıca: `--mfw`, `--output/-o`. (UMAP `bitig[cluster]` gerektirir) |
| `bitig cluster <path> --method {hierarchical,kmeans,hdbscan} --n-clusters N` | Kümeleme. Ayrıca: `--linkage`, `--mfw`, `--seed`, `--output/-o`. (HDBSCAN `bitig[cluster]` gerektirir) |
| `bitig consensus <path>` | MFW bantları üzerinde önyükleme fikir birliği ağacı. Ayrıca: `--bands`, `--replicates`, `--subsample`, `--support-threshold`, `--seed`, `--output/-o` |
| `bitig classify <path> --estimator {logreg,svm_linear,svm_rbf,rf,hgbm} --cv-kind {stratified,loao,leave_one_text_out}` | sklearn sınıflandırıcısı + stilometri uyumlu çapraz doğrulama. Ayrıca: `--groups-by`, `--folds`, `--mfw`, `--seed`. (`loao`, `--groups-by` gerektirir) |
| `bitig embed <path>` | Cümle veya bağlamsal gömme (ek: `bitig[embeddings]`). Ayrıca: `--model`, `--pool`, `--output/-o` |
| `bitig bayesian <path>` | Wallace–Mosteller yazar tespiti + hiyerarşik grup karşılaştırması. Ayrıca: `--group-by`, `--test-filter`, `--mfw`, `--prior-alpha` |

## Düzenleme

### `bitig run <study.yaml>`

Bildirimsel bir çalışmayı uçtan uca yürütür.

```bash
bitig run study.yaml --name demo [--output results/]
```

- `--name` — çalıştırma adı (çıktı dizini altındaki alt dizin)
- `--output, -o` — çıktı dizini (varsayılan, çalışmanın `output.dir` değerinden)

Her yöntemin `Result` nesnesini kendi alt dizinine ve çözümlenmiş bir yapılandırma kaydına yazar.

### `bitig report <run-dir>`

Bir çalıştırma dizininden Jinja2 HTML veya Markdown raporu oluşturur.

```bash
bitig report results/demo --output results/demo/report.html [--format html|md] [--title "Raporum"]
```

### `bitig plot <run-dir>`

!!! warning "Taslak"
    `bitig plot` şu anda bir taslaktır ve henüz şekil **oluşturmaz** (`--format`, `--dpi`
    kabul edilir ancak işlevsizdir). Şekilleri bugün oluşturmak için `bitig.viz.plot_*`
    işlevlerini doğrudan Python'dan çağırın ya da `study.yaml` içindeki `viz` bloğunu
    `bitig run` ile kullanın.

### `bitig shell [<corpus>]`

Bir çalışma kurulumunda size eşlik eden etkileşimli Rich tabanlı sihirbaz.

```bash
bitig shell [corpus/ --metadata corpus/metadata.tsv]
```

## Adli vakalar

### `bitig case <cmd>`

Adli Lab Vakalarını yönetir — tek bir inceleme için delil zinciri izlenen, imzalanabilir bir
çalışma alanı. Vaka deposunu göstermek için `--cases-dir` geçirin.

- `bitig case new` — yeni bir Vaka dizini oluşturur
- `bitig case list` — her Vakayı bir Rich tablosunda listeler
- `bitig case open` — vaka yolunu + tek satırlık özeti yazdırır
- `bitig case status` — tam durum: kayıt alanları, delil envanteri, zincir denetimi
- `bitig case fork` — bir Vakayı, üzerinde çalışmaya devam etmek için imzasız bir torun olarak klonlar
- `bitig case sign` — bir Vakayı imzalar ve kilitler (sonrasında salt okunur)
- `bitig case verify` — imzalı bir Vakanın delil zinciri mührünü doğrular

## Önbellek

### `bitig cache <cmd>`

`bitig ingest` tarafından üretilen spaCy DocBin önbelleğini yönetir:

- `bitig cache size` — saklanan toplam bayt sayısını gösterir
- `bitig cache list` — önbellek anahtarlarını listeler
- `bitig cache clear` — her girişi siler

## Yardım alma

Her komut `--help` seçeneğini destekler:

```bash
bitig --help
bitig run --help
```
