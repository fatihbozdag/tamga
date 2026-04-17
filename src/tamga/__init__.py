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
from tamga.methods.classify import build_classifier, cross_validate_tamga
from tamga.methods.cluster import HDBSCANCluster, HierarchicalCluster, KMeansCluster
from tamga.methods.consensus import BootstrapConsensus
from tamga.methods.delta import (
    ArgamonLinearDelta,
    BurrowsDelta,
    CosineDelta,
    EderDelta,
    EderSimpleDelta,
    QuadraticDelta,
)
from tamga.methods.reduce import MDSReducer, PCAReducer, TSNEReducer, UMAPReducer
from tamga.methods.zeta import ZetaClassic, ZetaEder
from tamga.preprocess.pipeline import ParsedCorpus, SpacyPipeline
from tamga.provenance import Provenance
from tamga.result import Result

__all__ = [
    "ArgamonLinearDelta",
    "BaseFeatureExtractor",
    "BootstrapConsensus",
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
    "HDBSCANCluster",
    "HierarchicalCluster",
    "KMeansCluster",
    "LexicalDiversityExtractor",
    "MDSReducer",
    "MFWExtractor",
    "PCAReducer",
    "ParsedCorpus",
    "PosNgramExtractor",
    "Provenance",
    "PunctuationExtractor",
    "QuadraticDelta",
    "ReadabilityExtractor",
    "Result",
    "SentenceLengthExtractor",
    "SpacyPipeline",
    "StudyConfig",
    "TSNEReducer",
    "UMAPReducer",
    "WordNgramExtractor",
    "ZetaClassic",
    "ZetaEder",
    "__version__",
    "build_classifier",
    "cross_validate_tamga",
    "load_config",
    "load_corpus",
    "load_metadata",
    "resolve_config",
]
