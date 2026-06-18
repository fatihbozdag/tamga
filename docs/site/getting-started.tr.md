# Başlarken

## Kurulum

bitig Python 3.11+ gerektirir.

=== "uv (önerilen)"

    ```bash
    uv pip install bitig
    python -m spacy download en_core_web_trf
    ```

=== "pip"

    ```bash
    pip install bitig
    python -m spacy download en_core_web_trf
    ```

### İsteğe bağlı eklentiler

```bash
uv pip install "bitig[bayesian]"    # PyMC + arviz for hierarchical models
uv pip install "bitig[embeddings]"  # sentence-transformers + contextual BERT
uv pip install "bitig[viz]"         # plotly, kaleido, ete3
uv pip install "bitig[reports]"     # weasyprint for PDF report export
uv pip install "bitig[docs]"        # mkdocs + material theme (build this site)
```

## Beş komutla bir çalışma

```bash
bitig init my-study          # (1) scaffold a project directory
cd my-study
# (2) drop .txt files into corpus/
# (3) fill in corpus/metadata.tsv — one row per file with filename → author, group, year, ...
bitig ingest corpus/ --metadata corpus/metadata.tsv  # (4) parse + cache
bitig info                    # (5a) bitig / Python / spaCy sürümlerini + yapılandırılmış dili yazdırır
bitig run study.yaml --name demo   # (5b) run the declared study
bitig report results/demo --output results/demo/report.html
```

`bitig init` ile oluşturulan proje iskeleti, tek bir en sık sözcük özellik kümesi
(ilk 1000 sözcük) ve bir Burrows Delta yöntemi içeren çalışan bir `study.yaml` içerir;
dolayısıyla yukarıdaki adım dizisi, aldığınız herhangi bir derlemde baştan sona çalışır.
Analizi genişletmek için `study.yaml` dosyasına PCA/Zeta/kümeleme blokları ekleyin
([Yöntemler](concepts/methods.md) ve [study.yaml şeması](reference/config.md) sayfalarına bakın).

## İlk Python oturumunuz

```python
from bitig import (
    Corpus, Document,
    MFWExtractor, BurrowsDelta,
    PCAReducer, plot_scatter_2d,
)

corpus = Corpus(documents=[
    Document(id="d1", text=open("doc1.txt").read(), metadata={"author": "Alice"}),
    Document(id="d2", text=open("doc2.txt").read(), metadata={"author": "Alice"}),
    Document(id="d3", text=open("doc3.txt").read(), metadata={"author": "Bob"}),
    Document(id="d4", text=open("doc4.txt").read(), metadata={"author": "Bob"}),
    Document(id="q",  text=open("questioned.txt").read(), metadata={"author": "?"}),
])

# 1. Extract the most-frequent-word feature matrix.
fm = MFWExtractor(n=200, scale="zscore", lowercase=True).fit_transform(corpus)

# 2. Train Burrows Delta on the known docs and attribute the questioned one.
import numpy as np
y = np.array(corpus.metadata_column("author"))
train_mask = y != "?"

clf = BurrowsDelta().fit(fm.X[train_mask], y[train_mask])  # sklearn-compatible: fit(X, y)
prediction = clf.predict(fm.X[~train_mask])                # e.g. array(['Alice'])
print(prediction)
```

## Örnek veri: Federalist örneği

Depo ile birlikte çalıştırmaya hazır iki örnek gelir:

- [`examples/quickstart/`](https://github.com/fatihbozdag/bitig/tree/main/examples/quickstart)
  — tartışmalı 50. makale dahil 9 makale kullanan başlangıç dostu bir kılavuz.
- [`examples/federalist/`](https://github.com/fatihbozdag/bitig/tree/main/examples/federalist)
  — Mosteller & Wallace (1964) sonucunu yeniden üreten 85 makalenin tam analizi.

Hızlı başlangıç, ilk çalıştırmada şu PCA grafiğini üretir:

<p align="center">
  <img src="https://raw.githubusercontent.com/fatihbozdag/bitig/main/examples/quickstart/results/demo/pca/pca.png" alt="Hamilton ve Madison'ın PCA grafiği" style="max-width: 82%;">
</p>

## Sonraki adımlar

- Ortak zihinsel modeli [Kavramlar](concepts/index.md) bölümünde öğrenin.
- Doğrulama, LR çıktısı ve PAN tarzı değerlendirme için [Adli dilbilim araç takımı](forensic/index.md)na geçin.
- Mosteller & Wallace'ı [Federalist öğreticisinde](tutorials/federalist.md) yeniden üretin.
