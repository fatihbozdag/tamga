"""Smoke tests for tamga's top-level public API."""


def test_languages_re_exported_from_top_level() -> None:
    import tamga

    assert "LANGUAGES" in tamga.__all__
    assert "LanguageSpec" in tamga.__all__
    assert "get_language" in tamga.__all__
    assert set(tamga.LANGUAGES) == {"en", "tr", "de", "es", "fr"}
    assert tamga.get_language("tr").backend == "spacy_stanza"
