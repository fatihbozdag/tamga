"""bitig — next-generation computational stylometry."""

from bitig._version import __version__
from bitig.config import StudyConfig, load_config, resolve_config
from bitig.corpus import Corpus, Document
from bitig.features import (
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
from bitig.forensic import (
    CalibratedScorer,
    CategorizedCharNgramExtractor,
    GeneralImpostors,
    Unmasking,
    distort_corpus,
    distort_text,
    log_lr_from_probs,
    log_lr_from_probs_with_priors,
)
from bitig.io import load_corpus, load_metadata
from bitig.languages import LANGUAGES, LanguageSpec, get_language
from bitig.methods.classify import build_classifier, cross_validate_bitig
from bitig.methods.cluster import HDBSCANCluster, HierarchicalCluster, KMeansCluster
from bitig.methods.consensus import BootstrapConsensus
from bitig.methods.delta import (
    ArgamonLinearDelta,
    BurrowsDelta,
    CosineDelta,
    EderDelta,
    EderSimpleDelta,
    QuadraticDelta,
)
from bitig.methods.reduce import MDSReducer, PCAReducer, TSNEReducer, UMAPReducer
from bitig.methods.zeta import ZetaClassic, ZetaEder
from bitig.preprocess.pipeline import ParsedCorpus, SpacyPipeline
from bitig.provenance import Provenance
from bitig.report import build_report
from bitig.result import Result
from bitig.runner import run_study
from bitig.viz import (
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
    "LANGUAGES",
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
    "LanguageSpec",
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
    "Unmasking",
    "WordNgramExtractor",
    "ZetaClassic",
    "ZetaEder",
    "__version__",
    "apply_publication_style",
    "build_classifier",
    "build_report",
    "cross_validate_bitig",
    "distort_corpus",
    "distort_text",
    "figure_size",
    "get_language",
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

# Optional extras — available only when bitig[embeddings] / bitig[bayesian] installed.
try:
    from bitig.features.embeddings import (
        ContextualEmbeddingExtractor,
        SentenceEmbeddingExtractor,
    )

    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

try:
    from bitig.methods.bayesian import (
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
