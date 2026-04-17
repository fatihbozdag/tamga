"""Feature extractors producing FeatureMatrix objects."""

from tamga.features.base import BaseFeatureExtractor, FeatureMatrix
from tamga.features.dependency import DependencyBigramExtractor
from tamga.features.function_words import FunctionWordExtractor
from tamga.features.lexical_diversity import LexicalDiversityExtractor
from tamga.features.mfw import MFWExtractor
from tamga.features.ngrams import CharNgramExtractor, WordNgramExtractor
from tamga.features.pos import PosNgramExtractor
from tamga.features.punctuation import PunctuationExtractor
from tamga.features.readability import ReadabilityExtractor
from tamga.features.sentence_length import SentenceLengthExtractor

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
