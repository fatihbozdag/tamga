"""tamga — next-generation computational stylometry."""

from tamga._version import __version__
from tamga.config import StudyConfig, load_config, resolve_config
from tamga.corpus import Corpus, Document
from tamga.io import load_corpus, load_metadata
from tamga.preprocess.pipeline import ParsedCorpus, SpacyPipeline
from tamga.provenance import Provenance

__all__ = [
    "__version__",
    "Corpus",
    "Document",
    "ParsedCorpus",
    "Provenance",
    "SpacyPipeline",
    "StudyConfig",
    "load_config",
    "load_corpus",
    "load_metadata",
    "resolve_config",
]
