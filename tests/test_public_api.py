"""Smoke tests for bitig's top-level public API."""


def test_languages_re_exported_from_top_level() -> None:
    import bitig

    assert "LANGUAGES" in bitig.__all__
    assert "LanguageSpec" in bitig.__all__
    assert "get_language" in bitig.__all__
    assert set(bitig.LANGUAGES) == {"en", "tr", "de", "es", "fr"}
    assert bitig.get_language("tr").backend == "spacy_stanza"
