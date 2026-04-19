"""Config-driven orchestrator — executes all methods declared in a `study.yaml`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import spacy

from tamga.config import StudyConfig, load_config
from tamga.features import (
    CharNgramExtractor,
    FeatureMatrix,
    FunctionWordExtractor,
    LexicalDiversityExtractor,
    MFWExtractor,
    PunctuationExtractor,
    ReadabilityExtractor,
    WordNgramExtractor,
)
from tamga.io import load_corpus
from tamga.methods.classify import build_classifier, cross_validate_tamga
from tamga.methods.cluster import HierarchicalCluster
from tamga.methods.consensus import BootstrapConsensus
from tamga.methods.delta import BurrowsDelta
from tamga.methods.reduce import PCAReducer
from tamga.methods.zeta import ZetaClassic
from tamga.plumbing.logging import get_logger
from tamga.preprocess.pipeline import SpacyPipeline
from tamga.provenance import Provenance
from tamga.result import Result

_log = get_logger(__name__)

_FEATURE_BUILDERS = {
    "mfw": MFWExtractor,
    "word_ngram": WordNgramExtractor,
    "char_ngram": CharNgramExtractor,
    "function_word": FunctionWordExtractor,
    "punctuation": PunctuationExtractor,
    "lexical_diversity": LexicalDiversityExtractor,
    "readability": ReadabilityExtractor,
}


def run_study(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
) -> Path:
    """Execute a full study from a `study.yaml` file and save all results.

    Returns the path to the run directory (e.g., `results/2026-04-17T10-15-30/`).
    """
    cfg: StudyConfig = load_config(Path(config_path))
    run_dir = _make_run_dir(cfg, output_dir, run_name)
    _log.info("run directory: %s", run_dir)

    corpus = load_corpus(
        Path(cfg.corpus.path), metadata=Path(cfg.corpus.metadata) if cfg.corpus.metadata else None
    )
    if cfg.corpus.filter:
        corpus = corpus.filter(**cfg.corpus.filter)
    _log.info("loaded %d documents", len(corpus))

    # Build all feature matrices by id.
    features_by_id: dict[str, FeatureMatrix] = {}
    for feat_cfg in cfg.features:
        extractor_cls = _FEATURE_BUILDERS.get(feat_cfg.type)
        if extractor_cls is None:
            _log.warning(
                "skipping feature %s: type %s not yet supported by runner",
                feat_cfg.id,
                feat_cfg.type,
            )
            continue
        extractor = extractor_cls(**feat_cfg.params)
        features_by_id[feat_cfg.id] = extractor.fit_transform(corpus)
        _log.info("built features %s: %s", feat_cfg.id, features_by_id[feat_cfg.id].X.shape)

    # SpacyPipeline resolves `language` → default model/backend via the languages registry.
    # Explicit model/backend on SpacyConfig override the registry defaults.
    pipe = SpacyPipeline(
        language=cfg.preprocess.language,
        model=cfg.preprocess.spacy.model,
        backend=cfg.preprocess.spacy.backend,
        exclude=list(cfg.preprocess.spacy.exclude),
    )

    # Execute each method.
    for method_cfg in cfg.methods:
        method_dir = run_dir / method_cfg.id
        method_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = _dispatch_method(method_cfg, corpus, features_by_id, seed=cfg.seed)
            result.provenance = Provenance.current(
                spacy_model=pipe.model,
                spacy_version=spacy.__version__,
                corpus_hash=corpus.hash(),
                feature_hash=None,
                seed=cfg.seed,
                resolved_config=cfg.model_dump(),
            )
            result.save(method_dir)
            _log.info("wrote %s", method_dir)
        except Exception as exc:
            _log.error("method %s failed: %s", method_cfg.id, exc)
            (method_dir / "error.txt").write_text(str(exc))

    (run_dir / "resolved_config.json").write_text(
        json.dumps(cfg.model_dump(), indent=2, default=str)
    )
    return run_dir


def _make_run_dir(cfg: StudyConfig, output_dir: str | Path | None, run_name: str | None) -> Path:
    base = Path(output_dir or cfg.output.dir)
    if run_name:
        run_dir = base / run_name
    elif cfg.output.timestamp:
        run_dir = base / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    else:
        run_dir = base
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _dispatch_method(
    method_cfg: Any,
    corpus: Any,
    features_by_id: dict[str, FeatureMatrix],
    *,
    seed: int,
) -> Result:
    kind = method_cfg.kind

    if kind == "delta":
        # Only Burrows for now in the runner; other Delta variants can be wired later.
        feat_id = (
            method_cfg.features if isinstance(method_cfg.features, str) else method_cfg.features[0]
        )
        fm = features_by_id[feat_id]
        y = np.array(corpus.metadata_column(method_cfg.group_by))
        clf = BurrowsDelta().fit(fm, y)
        preds = clf.predict(fm)
        return Result(
            method_name="burrows_delta",
            params=dict(method_cfg.params),
            values={"predictions": preds, "accuracy": float((preds == y).mean())},
        )

    if kind == "zeta":
        return ZetaClassic(
            group_by=method_cfg.group_by,
            top_k=int(method_cfg.params.get("top_k", 20)),
        ).fit_transform(corpus)

    if kind == "reduce":
        feat_id = (
            method_cfg.features if isinstance(method_cfg.features, str) else method_cfg.features[0]
        )
        fm = features_by_id[feat_id]
        return PCAReducer(n_components=int(method_cfg.params.get("n_components", 2))).fit_transform(
            fm
        )

    if kind == "cluster":
        feat_id = (
            method_cfg.features if isinstance(method_cfg.features, str) else method_cfg.features[0]
        )
        fm = features_by_id[feat_id]
        return HierarchicalCluster(
            n_clusters=int(method_cfg.params.get("n_clusters", 2)),
            linkage=method_cfg.params.get("linkage", "ward"),
        ).fit_transform(fm)

    if kind == "consensus":
        return BootstrapConsensus(
            mfw_bands=method_cfg.params.get("mfw_bands", [100, 200, 300]),
            replicates=int(method_cfg.params.get("replicates", 20)),
        ).fit_transform(corpus)

    if kind == "classify":
        feat_id = (
            method_cfg.features if isinstance(method_cfg.features, str) else method_cfg.features[0]
        )
        fm = features_by_id[feat_id]
        y = np.array(corpus.metadata_column(method_cfg.group_by))
        cv_kind = method_cfg.cv.kind if method_cfg.cv else "stratified"
        clf = build_classifier(method_cfg.params.get("estimator", "logreg"))
        report = cross_validate_tamga(
            clf,
            fm,
            y,
            cv_kind=cv_kind,
            groups_from=y if cv_kind == "loao" else None,
            seed=seed,
        )
        return Result(
            method_name=f"classify_{method_cfg.params.get('estimator', 'logreg')}",
            params=dict(method_cfg.params),
            values={"accuracy": report["accuracy"]},
        )

    raise ValueError(f"runner does not support method kind: {kind!r}")
