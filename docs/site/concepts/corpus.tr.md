# Corpus

`Corpus`, bir `Document` listesi ve (örtük olarak) bu belgelerin üst verisidir. Her ikisi de `tamga.corpus` içinde tanımlanmış veri sınıflarıdır.

## Derlem oluşturma

### Dosya sisteminden

```bash
tamga ingest corpus/ --metadata corpus/metadata.tsv
```

`metadata.tsv`, ilk sütunu `filename` olan (değerler `corpus/` dizinindeki dosya adlarıyla eşleştirilir) ve diğer her sütunun `Document.metadata` alanına dönüştüğü sekmeyle ayrılmış bir dosyadır:

```tsv
filename	author	year	genre
fed_01.txt	Hamilton	1787	political essay
fed_10.txt	Madison	1787	political essay
fed_50.txt	Unknown	1788	political essay
```

### Python ile

```python
from tamga.io import load_corpus
corpus = load_corpus("corpus/", metadata="corpus/metadata.tsv", strict=True)
```

- `strict=True` (varsayılan): Üst veri satırı eksik olan her belge için hata fırlatır.
- `strict=False`: Kısmi kapsama izin verir — yeni etiketlenmemiş bir belge geldiğinde kullanışlıdır.

### Programatik olarak

```python
from tamga.corpus import Corpus, Document

corpus = Corpus(documents=[
    Document(id="q", text=q_text, metadata={"role": "questioned"}),
    Document(id="k1", text=k1_text, metadata={"author": "Alice"}),
    Document(id="k2", text=k2_text, metadata={"author": "Alice"}),
])
```

## Filtreleme ve gruplama

`Corpus.filter(**query)`, belirtilen her anahtar-değer çiftini karşılayan belgelerden oluşan yeni bir Corpus döndürür:

```python
hamiltonian = corpus.filter(author="Hamilton")
train_only = corpus.filter(role="train")
```

Birden fazla değerle eşleştirmek için bir liste kullanılabilir:

```python
two_authors = corpus.filter(author=["Hamilton", "Madison"])
```

`Corpus.groupby(field)`, alt derlemlerin bulunduğu bir sözlük döndürür:

```python
grouped = corpus.groupby("author")
# {"Hamilton": Corpus(...), "Madison": Corpus(...), "Jay": Corpus(...)}
```

## Özetleme

`Corpus.hash()`, her belgenin metninin sıralanmış SHA-256 özetlerinden ve sıralanmış üst veri girişlerinden türetilen kararlı bir SHA-256 özeti üretir. Bu özet, her `Provenance` kaydına eklenir; böylece aynı derlemle çalışan iki çalışma, dosya sistemi yolundan, çalışma zamanından veya giriş dizinindeki belge sıralamasından bağımsız olarak aynı özeti paylaşır.

!!! note "Sıra duyarlılığı"
    `Corpus.hash()` tasarım gereği sıra-değişmezdir (aynı metinler + üst veri = aynı özet). Öznitelik matrisi düzenini etkileyen satır yeniden sıralamalarını tespit etmek gibi sıra-duyarlı bir özete ihtiyaç duyarsanız, `[d.id for d in corpus.documents]` listesini ayrıca özetleyin.

## Sonraki adım

- [Features](features.md) — derlemin sayısal matrise dönüştürülmesi.
