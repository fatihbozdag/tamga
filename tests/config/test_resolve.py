"""Tests for config YAML loading and precedence layering."""

from pathlib import Path

import pytest

from tamga.config import StudyConfig, load_config, resolve_config


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


def test_load_config_parses_yaml(tmp_path: Path) -> None:
    cfg_file = _write(
        tmp_path / "study.yaml",
        """
name: t1
seed: 7
corpus: {path: corpus/}
""",
    )
    cfg = load_config(cfg_file)
    assert isinstance(cfg, StudyConfig)
    assert cfg.name == "t1"
    assert cfg.seed == 7


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_resolve_config_cli_overrides_file(tmp_path: Path) -> None:
    cfg_file = _write(
        tmp_path / "study.yaml",
        """
name: t1
seed: 7
corpus: {path: corpus/}
""",
    )
    resolved = resolve_config(config_file=cfg_file, cli_overrides={"seed": 99})
    assert resolved.seed == 99
    assert resolved.name == "t1"


def test_resolve_config_deep_merges_nested_overrides(tmp_path: Path) -> None:
    cfg_file = _write(
        tmp_path / "study.yaml",
        """
name: t1
seed: 7
corpus: {path: corpus/, metadata: meta.tsv}
viz: {dpi: 300, format: [pdf, png]}
""",
    )
    resolved = resolve_config(config_file=cfg_file, cli_overrides={"viz": {"dpi": 600}})
    assert resolved.viz.dpi == 600
    # unspecified nested keys are preserved
    assert "pdf" in resolved.viz.format


def test_resolve_config_with_no_file_uses_defaults(tmp_path: Path) -> None:
    resolved = resolve_config(
        config_file=None,
        cli_overrides={"name": "cli-only", "corpus": {"path": "some/"}},
    )
    assert resolved.name == "cli-only"
    assert resolved.corpus.path == "some/"
