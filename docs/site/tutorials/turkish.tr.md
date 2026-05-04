# Türkçe stilometri öğreticisi

Küçük bir Türkçe kısa hikaye derleminde MFW + Ateşman okunabilirlik + Burrows Delta
kullanarak yazar tespiti yapan, baştan sona çalıştırılabilir bir örnek. Hikayeler, Türkçe
Wikisource'dan alınan, telif hakkı süresi dolmuş Ömer Seyfettin metinleridir.

## Kurulum

```bash
uv pip install 'bitig[turkish]'
python -c "import stanza; stanza.download('tr')"
bitig init seyfettin --language tr
cd seyfettin
```

Bu komut, Türkçe için önceden yapılandırılmış `study.yaml` içeren bir proje dizini oluşturur.
Şununla doğrulayın:

```bash
bitig info
```

`language` satırı `tr` gösterir.

## Derlem

`corpus/` dizinine UTF-8 `.txt` dosyası olarak 3-5 Türkçe kısa hikaye ekleyin. İyi bir
telif hakkı süresi dolmuş kaynak, [Türkçe Vikiskaynak'taki Ömer Seyfettin](https://tr.wikisource.org/wiki/Yazar:%C3%96mer_Seyfettin)
sayfasıdır — 20. yüzyıl başına ait düzinelerce kısa hikaye orada zaten yazıya geçirilmiş durumdadır.

`corpus/metadata.tsv` dosyasını ekleyin:

```tsv
filename	author	year
bomba.txt	Omer_Seyfettin	1910
kesik_biyik.txt	Omer_Seyfettin	1911
forsa.txt	Omer_Seyfettin	1913
pembe_incili_kaftan.txt	Omer_Seyfettin	1917
```

Gerçek bir çalışmada birden fazla yazar olması gerekir. Tek yazarlı demo için Seyfettin'i
birkaç Refik Halit Karay hikayesiyle eşleştirin (o da telif hakkı süresi dolmuş) — böylece
Delta'nın ayırt edecek bir şeyi olur.

## Çalışmayı çalıştırın

```bash
bitig ingest corpus/ --language tr --metadata corpus/metadata.tsv
bitig run study.yaml --name first-run
```

`bitig ingest`, Stanza'yı `spacy-stanza` aracılığıyla çalıştırır. İlk çalıştırma her belgeyi
ayrıştırır ve DocBin'leri önbelleğe alır; sonraki çalıştırmalar önbellekten okur ve saniyeler
içinde tamamlanır.

## Çıktılar

Varsayılan bir Türkçe çalışma şunları hesaplar:

- **MFW** (ilk 1000 belirteç, z-skorlanmış göreli sıklıklar)
- **Türkçe işlev sözcükleri** — UD Turkish BOUN kapalı sınıf belirteçlerinden türetilen
  `resources/languages/tr/function_words.txt` dosyasından yüklenir
- **Ateşman ve Bezirci-Yılmaz** okunabilirlik endeksleri
- **Burrows Delta** + PCA/MDS indirgeme grafikleri

`results/first-run/` çıktı klasörü şunları içerir:

- Delta skorları ve köken bilgisi içeren `result.json`
- `table_*.parquet` öznitelik matrisleri
- PNG / PDF şekiller (mesafe ısı haritası, PCA dağılım grafiği)
- Derlem özetini, seed değerini ve tam çözümlenmiş yapılandırmayı kaydeden `provenance.json`

## Özelleştirme

Öznitelikleri veya yöntemleri değiştirmek için `study.yaml` dosyasını düzenleyin. Örneğin,
MFW yerine bağlamsal gömme kullanmak için:

```yaml
features:
  - id: bert_tr
    type: contextual_embedding
    # model auto-resolves to `dbmdz/bert-base-turkish-cased` via the language registry
    pool: mean
```

Daha ağır bir Türkçe kodlayıcı için `model:` öğesini herhangi bir HuggingFace denetim
noktasına yönlendirin:

```yaml
features:
  - id: bert5urk
    type: contextual_embedding
    model: stefan-it/bert5urk
    pool: mean
```

## Türkçe'ye özgü notlar

- **Morfoloji.** Türkçe eklemeli bir dildir; `evlerinizden` gibi bir belirteç `ev+ler+iniz+den`
  biçimlerini tek formda bir araya getirir. Stanza'nın BOUN modeli bu biçimleri doğru şekilde
  lemmatize eder ve etiketler; bu durum POS n-gram ve bağımlılık tabanlı öznitelikler için
  önemlidir. Stilometrik çözümleme açısından bu morfolojik zenginlik, sözcük düzeyi MFW listelerini
  doğrudan Türkçe'ye uygulamak yerine lemma tabanlı ya da kök tabanlı öznitelikleri
  tercih etmeyi gerektirebilir.
- **Hece sayımı.** Hem Ateşman hem de Bezirci-Yılmaz, Türkçe yazıma özel bir sesli harf sayacı
  kullanır (`ı`, `ğ`, `ş`, `ç`, `ü`, `ö` dahil).
- **İşlev sözcükleri.** Paketlenmiş liste, Türkçe'nin kapalı sınıf ilgeçlerine, bağlaçlarına ve
  söylem parçacıklarına dayanır (örn. `ile`, `ancak`, `fakat`, `çünkü`, `ki`, `ise`).

## Sorun giderme

- **`ModuleNotFoundError: No module named 'spacy_stanza'`** — şunu çalıştırın:
  `uv pip install 'bitig[turkish]'`.
- **`FileNotFoundError: ... stanza_resources/tr/default.zip`** — şunu çalıştırın:
  `python -c "import stanza; stanza.download('tr')"`. Model yaklaşık 600 MB'tır.
- **MPS'de çok yavaş ilk aktarım.** Stanza'nın Türkçe modeli henüz Apple Silicon MPS'i
  desteklememektedir. İlk çalıştırmada CPU ayrıştırma hızları beklenir; sonraki çalıştırmalar
  önbellek isabetleridir.
