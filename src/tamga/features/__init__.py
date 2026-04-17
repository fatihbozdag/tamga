"""Feature extractors producing FeatureMatrix objects."""

from tamga.features.base import BaseFeatureExtractor, FeatureMatrix
from tamga.features.mfw import MFWExtractor

__all__ = ["BaseFeatureExtractor", "FeatureMatrix", "MFWExtractor"]
