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
from tamga.forensic import (
    CalibratedScorer,
    CategorizedCharNgramExtractor,
    GeneralImpostors,
    distort_corpus,
    distort_text,
    log_lr_from_probs,
    log_lr_from_probs_with_priors,
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
from tamga.report import build_report
from tamga.result import Result
from tamga.runner import run_study
from tamga.viz import (
    apply_publication_style,
    figure_size,
    plot_confusion_matrix,
    plot_dendrogram,
    plot_distance_heatmap,
    plot_feature_importance,
    plot_scatter_2d,
    plot_zeta,
)

__all__ = [
    "ArgamonLinearDelta",
    "BaseFeatureExtractor",
    "BootstrapConsensus",
    "BurrowsDelta",
    "CalibratedScorer",
    "CategorizedCharNgramExtractor",
    "CharNgramExtractor",
    "Corpus",
    "CosineDelta",
    "DependencyBigramExtractor",
    "Document",
    "EderDelta",
    "EderSimpleDelta",
    "FeatureMatrix",
    "FunctionWordExtractor",
    "GeneralImpostors",
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
    "apply_publication_style",
    "build_classifier",
    "build_report",
    "cross_validate_tamga",
    "distort_corpus",
    "distort_text",
    "figure_size",
    "load_config",
    "load_corpus",
    "load_metadata",
    "log_lr_from_probs",
    "log_lr_from_probs_with_priors",
    "plot_confusion_matrix",
    "plot_dendrogram",
    "plot_distance_heatmap",
    "plot_feature_importance",
    "plot_scatter_2d",
    "plot_zeta",
    "resolve_config",
    "run_study",
]

# Optional extras — available only when tamga[embeddings] / tamga[bayesian] installed.
try:
    from tamga.features.embeddings import (
        ContextualEmbeddingExtractor,
        SentenceEmbeddingExtractor,
    )

    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

try:
    from tamga.methods.bayesian import (
        BayesianAuthorshipAttributor,
        HierarchicalGroupComparison,
    )

    _BAYESIAN_AVAILABLE = True
except ImportError:
    _BAYESIAN_AVAILABLE = False

if _EMBEDDINGS_AVAILABLE:
    __all__ = [*__all__, "ContextualEmbeddingExtractor", "SentenceEmbeddingExtractor"]
if _BAYESIAN_AVAILABLE:
    __all__ = [*__all__, "BayesianAuthorshipAttributor", "HierarchicalGroupComparison"]
