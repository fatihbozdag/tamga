"""Delta family of distance-based nearest-author-centroid classifiers."""

from bitig.methods.delta.argamon import ArgamonLinearDelta, QuadraticDelta
from bitig.methods.delta.burrows import BurrowsDelta
from bitig.methods.delta.cosine import CosineDelta
from bitig.methods.delta.eder import EderDelta, EderSimpleDelta

__all__ = [
    "ArgamonLinearDelta",
    "BurrowsDelta",
    "CosineDelta",
    "EderDelta",
    "EderSimpleDelta",
    "QuadraticDelta",
]
