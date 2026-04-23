"""Delta family of distance-based nearest-author-centroid classifiers."""

from tamga.methods.delta.argamon import ArgamonLinearDelta, QuadraticDelta
from tamga.methods.delta.burrows import BurrowsDelta
from tamga.methods.delta.cosine import CosineDelta
from tamga.methods.delta.eder import EderDelta, EderSimpleDelta

__all__ = [
    "ArgamonLinearDelta",
    "BurrowsDelta",
    "CosineDelta",
    "EderDelta",
    "EderSimpleDelta",
    "QuadraticDelta",
]
