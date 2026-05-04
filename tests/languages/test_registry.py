"""Tests for the language registry."""

import dataclasses

import pytest

from bitig.languages import LANGUAGES, get_language


def test_registry_contains_five_first_class_languages() -> None:
    assert set(LANGUAGES) == {"en", "tr", "de", "es", "fr"}


def test_language_spec_is_frozen() -> None:
    spec = get_language("en")
    assert dataclasses.is_dataclass(spec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.code = "xx"  # type: ignore[misc]


def test_get_language_is_case_insensitive() -> None:
    assert get_language("EN") is get_language("en")


def test_get_language_unknown_raises_with_supported_list() -> None:
    with pytest.raises(ValueError, match="Unknown language code"):
        get_language("xx")


def test_english_spec_matches_current_defaults() -> None:
    spec = get_language("en")
    assert spec.code == "en"
    assert spec.name == "English"
    assert spec.default_model == "en_core_web_trf"
    assert spec.backend == "spacy"
    assert set(spec.readability_indices) >= {"flesch", "flesch_kincaid", "gunning_fog"}


def test_turkish_spec_uses_spacy_stanza_backend() -> None:
    spec = get_language("tr")
    assert spec.backend == "spacy_stanza"
    assert spec.default_model == "tr"
    assert spec.readability_indices == ("atesman", "bezirci_yilmaz")


def test_every_spec_declares_embedding_defaults() -> None:
    for code, spec in LANGUAGES.items():
        assert spec.contextual_embedding_default, f"{code} missing contextual default"
        assert spec.sentence_embedding_default, f"{code} missing sentence default"
