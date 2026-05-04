"""Feature extractors producing FeatureMatrix objects."""

from bitig.features.base import BaseFeatureExtractor, FeatureMatrix
from bitig.features.dependency import DependencyBigramExtractor
from bitig.features.function_words import FunctionWordExtractor
from bitig.features.lexical_diversity import LexicalDiversityExtractor
from bitig.features.mfw import MFWExtractor
from bitig.features.ngrams import CharNgramExtractor, WordNgramExtractor
from bitig.features.pos import PosNgramExtractor
from bitig.features.punctuation import PunctuationExtractor
from bitig.features.readability import ReadabilityExtractor
from bitig.features.sentence_length import SentenceLengthExtractor

__all__ = [
    "BaseFeatureExtractor",
    "CharNgramExtractor",
    "DependencyBigramExtractor",
    "FeatureMatrix",
    "FunctionWordExtractor",
    "LexicalDiversityExtractor",
    "MFWExtractor",
    "PosNgramExtractor",
    "PunctuationExtractor",
    "ReadabilityExtractor",
    "SentenceLengthExtractor",
    "WordNgramExtractor",
]

# Optional `bitig[embeddings]` extractors — available only when the extra is installed.
try:
    from bitig.features.embeddings import (
        ContextualEmbeddingExtractor,
        SentenceEmbeddingExtractor,
    )

    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

if _EMBEDDINGS_AVAILABLE:
    __all__ = [*__all__, "ContextualEmbeddingExtractor", "SentenceEmbeddingExtractor"]
