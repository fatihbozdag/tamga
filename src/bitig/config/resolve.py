"""YAML loading + deep-merge config resolution.

Precedence (highest wins): `cli_overrides` > `config_file` > package defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bitig.config.schema import StudyConfig


def load_config(path: Path) -> StudyConfig:
    """Parse a `study.yaml` file into a StudyConfig."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _validate(data)


def resolve_config(
    *,
    config_file: Path | None,
    cli_overrides: dict[str, Any] | None = None,
) -> StudyConfig:
    """Resolve the effective StudyConfig given a config file and CLI overrides.

    CLI overrides are deep-merged on top of the file contents. Keys unset at both layers fall
    back to the package defaults declared in `StudyConfig`.
    """
    base: dict[str, Any] = {}
    if config_file is not None:
        with Path(config_file).open("r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
    merged = _deep_merge(base, cli_overrides or {})
    return _validate(merged)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, overlay_value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(overlay_value, dict):
            out[key] = _deep_merge(out[key], overlay_value)
        else:
            out[key] = overlay_value
    return out


def _validate(data: dict[str, Any]) -> StudyConfig:
    # StudyConfig requires `corpus`; if absent, fill in a placeholder the caller is expected to
    # override via CLI / subsequent validation. This allows `resolve_config` to be reused for
    # partial inspection commands like `bitig config show`.
    data.setdefault("corpus", {"path": ""})
    return StudyConfig.model_validate(data)
