"""tamga — next-generation computational stylometry."""

from tamga._version import __version__
from tamga.config import StudyConfig, load_config, resolve_config
from tamga.corpus import Corpus, Document
from tamga.features import (
    BaseFeatureExtractor,
    CharNgramExtractor,
    DependencyBigramExtractor,
    FeatureMatrix,
    FunctionWordExtractor,
    LexicalDiversityExtractor,
    MFWExtractor,
    PosNgramExtractor,
    PunctuationExtractor,
    ReadabilityExtractor,
    SentenceLengthExtractor,
    WordNgramExtractor,
)
from tamga.io import load_corpus, load_metadata
from tamga.methods.delta import (
    ArgamonLinearDelta,
    BurrowsDelta,
    CosineDelta,
    EderDelta,
    EderSimpleDelta,
    QuadraticDelta,
)
from tamga.preprocess.pipeline import ParsedCorpus, SpacyPipeline
from tamga.provenance import Provenance

__all__ = [
    "ArgamonLinearDelta",
    "BaseFeatureExtractor",
    "BurrowsDelta",
    "CharNgramExtractor",
    "Corpus",
    "CosineDelta",
    "DependencyBigramExtractor",
    "Document",
    "EderDelta",
    "EderSimpleDelta",
    "FeatureMatrix",
    "FunctionWordExtractor",
    "LexicalDiversityExtractor",
    "MFWExtractor",
    "ParsedCorpus",
    "PosNgramExtractor",
    "Provenance",
    "PunctuationExtractor",
    "QuadraticDelta",
    "ReadabilityExtractor",
    "SentenceLengthExtractor",
    "SpacyPipeline",
    "StudyConfig",
    "WordNgramExtractor",
    "__version__",
    "load_config",
    "load_corpus",
    "load_metadata",
    "resolve_config",
]
