"""Feature extractors producing FeatureMatrix objects."""

from tamga.features.base import BaseFeatureExtractor, FeatureMatrix
from tamga.features.mfw import MFWExtractor
from tamga.features.ngrams import CharNgramExtractor, WordNgramExtractor

__all__ = [
    "BaseFeatureExtractor",
    "CharNgramExtractor",
    "FeatureMatrix",
    "MFWExtractor",
    "WordNgramExtractor",
]
