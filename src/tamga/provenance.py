"""The Provenance record — captured on every Result so re-runs are fully reproducible."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from tamga._version import __version__


@dataclass
class Provenance:
    tamga_version: str
    python_version: str
    spacy_model: str
    spacy_version: str
    corpus_hash: str
    feature_hash: str | None
    seed: int
    timestamp: datetime
    resolved_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        raw_ts = data["timestamp"]
        ts = datetime.fromisoformat(raw_ts) if isinstance(raw_ts, str) else raw_ts
        return cls(
            tamga_version=data["tamga_version"],
            python_version=data["python_version"],
            spacy_model=data["spacy_model"],
            spacy_version=data["spacy_version"],
            corpus_hash=data["corpus_hash"],
            feature_hash=data.get("feature_hash"),
            seed=int(data["seed"]),
            timestamp=ts,
            resolved_config=dict(data.get("resolved_config") or {}),
        )

    @classmethod
    def current(
        cls,
        *,
        spacy_model: str,
        spacy_version: str,
        corpus_hash: str,
        feature_hash: str | None,
        seed: int,
        resolved_config: dict[str, Any],
    ) -> Provenance:
        return cls(
            tamga_version=__version__,
            python_version=platform.python_version(),
            spacy_model=spacy_model,
            spacy_version=spacy_version,
            corpus_hash=corpus_hash,
            feature_hash=feature_hash,
            seed=seed,
            timestamp=datetime.now(),
            resolved_config=resolved_config,
        )
