"""Tests for the StudyConfig pydantic schema."""

import pytest
from pydantic import ValidationError

from tamga.config.schema import StudyConfig

VALID_MINIMAL = {
    "name": "demo",
    "seed": 42,
    "corpus": {"path": "corpus/"},
    "preprocess": {},
    "features": [],
    "methods": [],
    "viz": {},
    "report": {},
    "cache": {},
    "output": {},
}


def test_minimal_config_validates():
    cfg = StudyConfig(**VALID_MINIMAL)
    assert cfg.name == "demo"
    assert cfg.seed == 42
    assert cfg.corpus.path == "corpus/"


def test_feature_config_requires_id_and_type():
    bad = dict(VALID_MINIMAL, features=[{"type": "mfw"}])
    with pytest.raises(ValidationError):
        StudyConfig(**bad)


def test_feature_config_accepts_full_entry():
    cfg = StudyConfig(
        **dict(
            VALID_MINIMAL,
            features=[{"id": "mfw1000", "type": "mfw", "n": 1000, "min_df": 2, "scale": "zscore"}],
        )
    )
    assert cfg.features[0].id == "mfw1000"
    assert cfg.features[0].params["n"] == 1000


def test_method_config_validates_kind():
    cfg = StudyConfig(
        **dict(
            VALID_MINIMAL,
            methods=[{"id": "d1", "kind": "delta", "method": "burrows", "features": "mfw1000"}],
        )
    )
    assert cfg.methods[0].kind == "delta"
    assert cfg.methods[0].params["method"] == "burrows"


def test_viz_config_defaults():
    cfg = StudyConfig(**VALID_MINIMAL)
    assert cfg.viz.dpi == 300
    assert "pdf" in cfg.viz.format
    assert "png" in cfg.viz.format


def test_report_offline_default_false():
    cfg = StudyConfig(**VALID_MINIMAL)
    assert cfg.report.offline is False


def test_seed_default_is_42():
    cfg = StudyConfig(**dict(VALID_MINIMAL, seed=None))
    # pydantic should use the default
    assert cfg.seed == 42


def test_round_trip_dict_json():
    cfg = StudyConfig(**VALID_MINIMAL)
    redump = cfg.model_dump()
    again = StudyConfig(**redump)
    assert again == cfg


def test_preprocess_language_defaults_to_english() -> None:
    from tamga.config.schema import PreprocessConfig

    cfg = PreprocessConfig()
    assert cfg.language == "en"


def test_preprocess_language_accepts_registered_code() -> None:
    from tamga.config.schema import PreprocessConfig

    cfg = PreprocessConfig(language="tr")
    assert cfg.language == "tr"


def test_preprocess_language_rejects_unknown_code() -> None:
    import pytest
    from pydantic import ValidationError

    from tamga.config.schema import PreprocessConfig

    with pytest.raises(ValidationError, match="Unknown language code"):
        PreprocessConfig(language="xx")


def test_preprocess_language_case_insensitive() -> None:
    from tamga.config.schema import PreprocessConfig

    cfg = PreprocessConfig(language="TR")
    assert cfg.language == "tr"


def test_spacy_config_model_now_optional() -> None:
    from tamga.config.schema import SpacyConfig

    cfg = SpacyConfig()
    assert cfg.model is None
    assert cfg.backend is None
