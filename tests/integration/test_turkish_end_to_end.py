"""End-to-end Turkish pipeline: Stanza parse -> features -> readability.

Skipped unless:
  - ``spacy-stanza`` and ``stanza`` are installed (``pip install 'tamga[turkish]'``)
  - Stanza's Turkish model has been downloaded (``stanza.download('tr')``)

Runs under the ``slow`` marker only, so the default ``-m "not slow"`` CI grid stays green on
machines without the Turkish stack. The ``tests-multilang`` workflow opts in explicitly.
"""

from __future__ import annotations

import pytest

# ``importorskip`` collects cleanly on dev machines that lack the Turkish extras. If the import
# succeeds but the Stanza model is missing, the pipeline init below raises RuntimeError with a
# download hint — we catch that and convert it into a skip so the suite still reports green.
pytest.importorskip("spacy_stanza")
pytest.importorskip("stanza")

pytestmark = pytest.mark.slow


def test_turkish_pipeline_smoke(tmp_path) -> None:
    from tamga.corpus import Corpus, Document
    from tamga.features.function_words import FunctionWordExtractor
    from tamga.features.readability import ReadabilityExtractor
    from tamga.preprocess.pipeline import SpacyPipeline

    texts = [
        "Ali topu tuttu. Kedi uyudu. Hava güzeldi.",
        "Ahmet kitabı okudu. Öğretmen sordu. Öğrenciler dinledi.",
        "Mehmet eve gitti. Kardeşi bekledi. Yemek yediler.",
    ]
    corpus = Corpus(
        documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)],
        language="tr",
    )

    pipe = SpacyPipeline(language="tr", cache_dir=tmp_path / "cache")
    try:
        parsed = pipe.parse(corpus)
    except RuntimeError as exc:
        # Stanza Turkish model not downloaded on this machine.
        pytest.skip(f"Stanza Turkish model unavailable: {exc}")

    assert len(parsed) == 3

    fw = FunctionWordExtractor(scale="none").fit_transform(corpus)
    assert fw.X.shape[0] == 3

    rb = ReadabilityExtractor().fit_transform(corpus)
    # Two Turkish indices: Ateşman + Bezirci-Yılmaz.
    assert rb.X.shape == (3, 2)
