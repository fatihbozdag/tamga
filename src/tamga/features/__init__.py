"""Feature extractors producing FeatureMatrix objects."""

from tamga.features.base import BaseFeatureExtractor, FeatureMatrix
from tamga.features.dependency import DependencyBigramExtractor
from tamga.features.function_words import FunctionWordExtractor
from tamga.features.mfw import MFWExtractor
from tamga.features.ngrams import CharNgramExtractor, WordNgramExtractor
from tamga.features.pos import PosNgramExtractor
from tamga.features.punctuation import PunctuationExtractor

__all__ = [
    "BaseFeatureExtractor",
    "CharNgramExtractor",
    "DependencyBigramExtractor",
    "FeatureMatrix",
    "FunctionWordExtractor",
    "MFWExtractor",
    "PosNgramExtractor",
    "PunctuationExtractor",
    "WordNgramExtractor",
]
